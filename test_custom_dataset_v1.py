"""Testing script for adapter on custom dataset with prefix-based evaluation."""

import argparse
import json
import os
import re
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from tabulate import tabulate
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

import adapterlib
from adapterlib import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from dataset import Dataset, PromptDataset
from tools import Evaluator, get_logger, get_transform, setup_seed
from tools.utils import normalize


def normalize_map_global(x, eps=1e-6):
    min_v = x.min()
    max_v = x.max()
    return (x - min_v) / (max_v - min_v + eps)
def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def mask_to_boxes(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w - 1, y + h - 1])
    return boxes

def filter_boxes_by_area(boxes, H, W, min_area_ratio=0.002):
    """过滤掉面积太小的box
    
    Args:
        boxes: list of boxes, each box is [x1, y1, x2, y2]
        H: 图像高度
        W: 图像宽度
        min_area_ratio: 最小面积比例，默认0.002（0.2%）
    
    Returns:
        过滤后的boxes列表
    """
    if len(boxes) == 0:
        return boxes
    
    MIN_BOX_AREA = min_area_ratio * H * W  # 0.2% 图像面积
    filtered_boxes = [
        b for b in boxes
        if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA
    ]
    return filtered_boxes

def compute_precision_recall_at_threshold(
    gt_masks,
    pr_maps,
    gt_anomalys,
    box_thresh=0.24,
    iou_thresh=0.2,
    return_all_scores_labels=False  # 新增：是否返回所有的分数和标签用于mAP计算
):
    """
    在指定阈值下计算precision和recall
    
    Args:
        gt_masks: GT mask列表
        pr_maps: 预测map列表
        gt_anomalys: GT异常标签列表
        box_thresh: 检测阈值
        iou_thresh: IoU阈值
        return_all_scores_labels: 是否返回所有的分数和标签（用于mAP计算）
    
    Returns:
        precision, recall, TP, FP, FN 或 (precision, recall, TP, FP, FN, all_scores, all_labels)
    """
    TP, FP, FN = 0, 0, 0
    all_scores = []
    all_labels = []
    
    # 将gt_anomalys转换为numpy数组以便索引
    if torch.is_tensor(gt_anomalys):
        gt_anomalys_np = gt_anomalys.cpu().numpy()
    else:
        gt_anomalys_np = np.array(gt_anomalys)

    for i in range(len(gt_masks)):
        if int(gt_anomalys_np[i]) == 0:
            continue

        gt_mask = gt_masks[i]
        pr_map = pr_maps[i]

        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.cpu().numpy()
        if torch.is_tensor(pr_map):
            pr_map = pr_map.cpu().numpy()

        gt_mask = gt_mask.squeeze()
        pr_map = pr_map.squeeze()

        H, W = gt_mask.shape[:2]

        # ================= GT boxes =================
        gt_boxes = mask_to_boxes(gt_mask)
        gt_boxes = filter_boxes_by_area(gt_boxes, H, W, min_area_ratio=0.002)

        # ================= Pred boxes =================
        # 使用指定阈值
        pred_bin = (pr_map >= box_thresh).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

        pred_boxes = mask_to_boxes(pred_bin)
        pred_boxes = filter_boxes_by_area(pred_boxes, H, W, min_area_ratio=0.002)

        # ★ box score：用 mean
        pred_scores = []
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            region = pr_map[y1:y2+1, x1:x2+1]
            pred_scores.append(float(region.mean()))

        # ★ NMS
        keep = nms(pred_boxes, pred_scores, iou_thresh=iou_thresh)
        pred_boxes = [pred_boxes[k] for k in keep]
        pred_scores = [pred_scores[k] for k in keep]

        matched_gt = set()
        order = np.argsort(pred_scores)[::-1]

        for idx in order:
            pb = pred_boxes[idx]
            score = pred_scores[idx]

            best_iou = 0.0
            best_gt = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            if best_iou >= iou_thresh:
                TP += 1
                matched_gt.add(best_gt)
                if return_all_scores_labels:
                    all_scores.append(score)
                    all_labels.append(1)
            else:
                FP += 1
                if return_all_scores_labels:
                    all_scores.append(score)
                    all_labels.append(0)

        FN += (len(gt_boxes) - len(matched_gt))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if return_all_scores_labels:
        return precision, recall, TP, FP, FN, all_scores, all_labels
    else:
        return precision, recall, TP, FP, FN


def compute_map_from_pr_curve(recalls, precisions, logger=None):
    """
    从PR曲线计算mAP
    
    Args:
        recalls: recall列表
        precisions: precision列表
        logger: 日志记录器
    
    Returns:
        mAP: 平均精度
        best_f1_precision: F1最大时的precision
        best_f1_recall: F1最大时的recall
        best_f1: 最大F1值
    """
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # 计算F1分数
    f1_scores = np.zeros_like(precisions)
    valid_mask = (precisions + recalls) > 0
    f1_scores[valid_mask] = 2 * (precisions[valid_mask] * recalls[valid_mask]) / (precisions[valid_mask] + recalls[valid_mask])
    
    # 找到F1最大的点
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precisions[best_f1_idx]
    best_f1_recall = recalls[best_f1_idx]
    
    # 计算mAP (Average Precision) - 使用11点插值法
    ap_11_points = []
    for r in np.arange(0, 1.1, 0.1):
        mask = recalls >= r
        if np.any(mask):
            max_precision = np.max(precisions[mask])
            ap_11_points.append(max_precision)
        else:
            ap_11_points.append(0.0)
    
    mAP_11 = np.mean(ap_11_points)
    
    # 使用梯形法则计算面积（更精确的方法）
    if len(recalls) > 1:
        # 对recall进行排序（从小到大）
        sorted_idx = np.argsort(recalls)
        sorted_recalls = recalls[sorted_idx]
        sorted_precisions = precisions[sorted_idx]
        
        # 对precision进行单调递减处理
        monotonic_precisions = np.zeros_like(sorted_precisions)
        for i in range(len(sorted_precisions) - 1, -1, -1):
            if i == len(sorted_precisions) - 1:
                monotonic_precisions[i] = sorted_precisions[i]
            else:
                monotonic_precisions[i] = max(sorted_precisions[i], monotonic_precisions[i + 1])
        
        # 添加端点
        all_recalls = np.concatenate([[0.0], sorted_recalls])
        if len(monotonic_precisions) > 0:
            all_precisions = np.concatenate([[monotonic_precisions[0]], monotonic_precisions])
        else:
            all_precisions = np.array([1.0, 0.0])
        
        # 如果最后一个recall小于1，添加点(1, 最后一个precision)
        if all_recalls[-1] < 1.0:
            all_recalls = np.concatenate([all_recalls, [1.0]])
            all_precisions = np.concatenate([all_precisions, [all_precisions[-1]]])
        
        # 使用梯形法则计算面积
        mAP_trapezoid = np.trapz(all_precisions, all_recalls)
    else:
        mAP_trapezoid = mAP_11
    
    return mAP_trapezoid, best_f1_precision, best_f1_recall, best_f1


def compute_map_with_multiple_thresholds_and_plot(
    gt_masks,
    pr_maps,
    gt_anomalys,
    threshold_range=(0.0, 1.0),
    threshold_step=0.01,
    iou_thresh=0.2,
    save_path=None,
    logger=None,
    specified_threshold=None  # 新增：指定的阈值（用于标记在图上）
):
    """
    在不同阈值下计算precision和recall，绘制PR曲线并计算mAP
    
    Args:
        gt_masks: GT mask列表
        pr_maps: 预测map列表
        gt_anomalys: GT异常标签列表
        threshold_range: 阈值范围 (min, max)
        threshold_step: 阈值步长
        iou_thresh: IoU阈值
        save_path: 保存路径，如果提供则保存图像
        logger: 日志记录器
        specified_threshold: 指定的阈值（用于在图上标记对应的点）
    
    Returns:
        mAP: 平均精度
        best_f1_precision: F1最大时的precision
        best_f1_recall: F1最大时的recall
        best_f1: 最大F1值
        pr_data: (recalls, precisions, thresholds) 元组
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    precisions = []
    recalls = []
    
    if logger:
        logger.info(f"开始在不同阈值下计算precision和recall...")
        logger.info(f"阈值范围: [{threshold_range[0]:.3f}, {threshold_range[1]:.3f}], 步长: {threshold_step:.3f}")
        logger.info(f"共 {len(thresholds)} 个阈值")
    
    # 收集所有阈值下的预测框分数和标签用于mAP计算
    all_scores_all_thresholds = []
    all_labels_all_thresholds = []
    
    # 在不同阈值下计算precision和recall
    for thresh in tqdm(thresholds, desc="计算不同阈值下的PR", disable=logger is None):
        precision, recall, _, _, _, scores, labels = compute_precision_recall_at_threshold(
            gt_masks, pr_maps, gt_anomalys, box_thresh=thresh, iou_thresh=iou_thresh,
            return_all_scores_labels=True
        )
        precisions.append(precision)
        recalls.append(recall)
        all_scores_all_thresholds.extend(scores)
        all_labels_all_thresholds.extend(labels)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # 使用average_precision_score计算mAP
    if len(all_scores_all_thresholds) > 0 and len(all_labels_all_thresholds) > 0:
        # 转换为numpy数组
        all_scores_np = np.array(all_scores_all_thresholds)
        all_labels_np = np.array(all_labels_all_thresholds)
        
        # 检查是否有正样本和负样本
        if len(np.unique(all_labels_np)) >= 2:
            mAP = average_precision_score(all_labels_np, all_scores_np)
        else:
            mAP = 0.0
            if logger:
                logger.warning("所有标签都是同一类别，无法使用average_precision_score计算mAP，设为0.0")
    else:
        mAP = 0.0
        if logger:
            logger.warning("没有预测框，无法计算mAP，设为0.0")
    
    # 计算F1最大点（仍然使用多阈值PR曲线）
    # 注意：这里传入recalls和precisions只是为了计算F1，mAP已经用average_precision_score计算
    _, best_f1_precision, best_f1_recall, best_f1 = compute_map_from_pr_curve(
        recalls, precisions, logger=logger
    )
    
    # 绘制PR曲线
    if save_path:
        plt.figure(figsize=(10, 8))
        
        # 按照threshold顺序绘制原始阈值点的连接线（这是用户期望看到的）
        # 阈值从高到低：随着阈值降低，通常recall增加，precision可能降低
        threshold_order_idx = np.argsort(thresholds)[::-1]  # 从高阈值到低阈值排序
        threshold_ordered_recalls = recalls[threshold_order_idx]
        threshold_ordered_precisions = precisions[threshold_order_idx]
        threshold_ordered_thresholds = thresholds[threshold_order_idx]
        plt.plot(threshold_ordered_recalls, threshold_ordered_precisions, 'b-', linewidth=2.5, 
                label=f'PR Curve (by Threshold Order)', alpha=0.7, linestyle='-', zorder=1)
        
        # 在蓝色曲线上标注阈值值（稀疏标注，避免过于密集）
        # 根据点数决定标注间隔：如果点很多，每隔几个点标注一次
        num_points = len(threshold_ordered_recalls)
        if num_points > 20:
            # 如果点数很多，只标注部分点（例如每5个点标注一次，或者标注关键阈值）
            # 选择标注间隔：确保标注的点数不超过15个
            annotate_step = max(1, num_points // 15)
            annotate_indices = list(range(0, num_points, annotate_step))
            # 确保标注第一个和最后一个点
            if annotate_indices[-1] != num_points - 1:
                annotate_indices.append(num_points - 1)
        else:
            # 如果点数不多，标注所有点
            annotate_indices = list(range(num_points))
        
        # 添加阈值标注
        for idx in annotate_indices:
            r = threshold_ordered_recalls[idx]
            p = threshold_ordered_precisions[idx]
            thresh = threshold_ordered_thresholds[idx]
            # 标注阈值值，稍微偏移位置避免重叠
            plt.annotate(f'{thresh:.2f}', 
                        xy=(r, p), 
                        xytext=(5, 5),  # 偏移5个像素
                        textcoords='offset points',
                        fontsize=7,
                        alpha=0.7,
                        color='blue',
                        ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='blue', linewidth=0.5),
                        zorder=6)
        
        # 对recall进行排序（用于计算mAP的标准方法）
        sorted_idx = np.argsort(recalls)
        sorted_recalls = recalls[sorted_idx]
        sorted_precisions = precisions[sorted_idx]
        sorted_thresholds = thresholds[sorted_idx]
        
        # 对precision进行单调递减处理（用于mAP计算的标准方法）
        # 这确保了PR曲线是单调递减的，符合标准PR曲线定义
        monotonic_precisions = np.zeros_like(sorted_precisions)
        for i in range(len(sorted_precisions) - 1, -1, -1):
            if i == len(sorted_precisions) - 1:
                monotonic_precisions[i] = sorted_precisions[i]
            else:
                monotonic_precisions[i] = max(sorted_precisions[i], monotonic_precisions[i + 1])
        
        # 绘制单调递减的PR曲线（用于mAP计算，这是标准的PR曲线绘制方法）
        plt.plot(sorted_recalls, monotonic_precisions, 'r--', linewidth=2, 
                label=f'Monotonic PR Curve (for mAP calculation)', alpha=0.8, zorder=2)
        
        # 绘制原始点
        plt.scatter(recalls, precisions, c='blue', s=25, alpha=0.7, label='Threshold Points', 
                   zorder=4, edgecolors='darkblue', linewidths=0.8)
        
        # 标记F1最大的点
        plt.plot(best_f1_recall, best_f1_precision, 'ro', markersize=14, 
                label=f'Best F1 Point (P={best_f1_precision:.4f}, R={best_f1_recall:.4f}, F1={best_f1:.4f})',
                zorder=5)
        
        # 如果指定了阈值，标记该阈值对应的点
        specified_precision_plot = None
        specified_recall_plot = None
        if specified_threshold is not None and threshold_range[0] <= specified_threshold <= threshold_range[1]:
            # 找到最接近指定阈值的点
            closest_idx = np.argmin(np.abs(thresholds - specified_threshold))
            specified_precision_plot = precisions[closest_idx]
            specified_recall_plot = recalls[closest_idx]
            plt.plot(specified_recall_plot, specified_precision_plot, 'go', markersize=14, 
                    label=f'Specified Threshold={specified_threshold:.3f} (P={specified_precision_plot:.4f}, R={specified_recall_plot:.4f})',
                    zorder=5, markerfacecolor='none', markeredgewidth=2.5)
        
        # 添加11点插值曲线
        recall_points_11 = np.arange(0, 1.1, 0.1)
        precision_points_11 = []
        for r in recall_points_11:
            mask = recalls >= r
            if np.any(mask):
                precision_points_11.append(np.max(precisions[mask]))
            else:
                precision_points_11.append(0.0)
        plt.plot(recall_points_11, precision_points_11, 'g--', linewidth=1.5, alpha=0.6, 
                label='11-point interpolation', marker='s', markersize=5, zorder=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (Multi-Threshold mAP Calculation)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left', fontsize=9)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # 添加文本信息
        info_text = f'mAP (average_precision_score): {mAP:.4f}\n'
        info_text += f'Best F1: {best_f1:.4f}\n'
        info_text += f'Best Precision: {best_f1_precision:.4f}\n'
        info_text += f'Best Recall: {best_f1_recall:.4f}\n'
        if specified_threshold is not None and specified_precision_plot is not None:
            info_text += f'\nSpecified Threshold: {specified_threshold:.3f}\n'
            info_text += f'Specified Precision: {specified_precision_plot:.4f}\n'
            info_text += f'Specified Recall: {specified_recall_plot:.4f}\n'
        info_text += f'\nThreshold Range: [{threshold_range[0]:.2f}, {threshold_range[1]:.2f}]\n'
        info_text += f'Threshold Step: {threshold_step:.3f}\n'
        info_text += f'Number of Thresholds: {len(thresholds)}\n'
        info_text += f'Total Predictions: {len(all_scores_all_thresholds)}'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, 'pr_curve_map.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"PR曲线图已保存到: {plot_path}")
    
    return mAP, best_f1_precision, best_f1_recall, best_f1, (recalls, precisions, thresholds)


def compute_box_map(
    gt_masks,
    pr_maps,
    gt_anomalys,
    box_thresh=0.24,      # ★ 统一阈值，0.3~0.5 都可调
    iou_thresh=0.2,
    logger=None,
    save_path=None  # 新增参数，用于保存PR曲线
):
    """
    返回：
        mAP, precision, recall, TP, FP, FN
    """

    all_scores = []
    all_labels = []

    TP, FP, FN = 0, 0, 0
    num_anomaly_images = 0
    num_images_with_gt_boxes = 0
    num_images_with_pred_boxes = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    pr_map_max_values = []

    for i in range(len(gt_masks)):
        if gt_anomalys[i] == 0:
            continue

        num_anomaly_images += 1
        gt_mask = gt_masks[i]
        pr_map = pr_maps[i]

        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.cpu().numpy()
        if torch.is_tensor(pr_map):
            pr_map = pr_map.cpu().numpy()

        gt_mask = gt_mask.squeeze()
        pr_map = pr_map.squeeze()

        # 不使用归一化，直接使用原始预测值，使用绝对阈值0.24
        H, W = gt_mask.shape[:2]
        pr_map_max_values.append(float(pr_map.max()))

        # ================= GT boxes =================
        gt_boxes = mask_to_boxes(gt_mask)
        gt_boxes = filter_boxes_by_area(gt_boxes, H, W, min_area_ratio=0.002)
        total_gt_boxes += len(gt_boxes)
        if len(gt_boxes) > 0:
            num_images_with_gt_boxes += 1

        # ================= Pred boxes =================
        # 使用绝对阈值0.24（不归一化，直接使用原始预测值）
        pred_bin = (pr_map >= box_thresh).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

        pred_boxes = mask_to_boxes(pred_bin)
        pred_boxes = filter_boxes_by_area(pred_boxes, H, W, min_area_ratio=0.002)

        if len(pred_boxes) > 0:
            num_images_with_pred_boxes += 1
            total_pred_boxes += len(pred_boxes)

        # ★ box score：用 mean（比 max 稳定，AP 更高）
        pred_scores = []
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            region = pr_map[y1:y2+1, x1:x2+1]
            pred_scores.append(float(region.mean()))

        # ★ NMS：FP 大幅下降
        keep = nms(pred_boxes, pred_scores, iou_thresh=0.5)
        pred_boxes = [pred_boxes[k] for k in keep]
        pred_scores = [pred_scores[k] for k in keep]

        matched_gt = set()
        order = np.argsort(pred_scores)[::-1]

        for idx in order:
            pb = pred_boxes[idx]
            score = pred_scores[idx]

            best_iou = 0.0
            best_gt = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            if best_iou >= iou_thresh:
                TP += 1
                matched_gt.add(best_gt)
                all_scores.append(score)
                all_labels.append(1)
            else:
                FP += 1
                all_scores.append(score)
                all_labels.append(0)

        FN += (len(gt_boxes) - len(matched_gt))

    # ================= Diagnostics =================
    if logger is not None:
        logger.info("诊断信息:")
        logger.info(f"  异常图像数量: {num_anomaly_images}")
        logger.info(f"  有GT框的图像数: {num_images_with_gt_boxes}")
        logger.info(f"  有预测框的图像数: {num_images_with_pred_boxes}")
        logger.info(f"  总GT框数: {total_gt_boxes}")
        logger.info(f"  总预测框数: {total_pred_boxes}")
        if len(pr_map_max_values) > 0:
            logger.info(
                f"  预测图最大值范围(原始值，未归一化): "
                f"[{min(pr_map_max_values):.3f}, {max(pr_map_max_values):.3f}]"
            )

    # 在指定阈值下计算的precision和recall（用于日志显示）
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # 确定阈值范围（用于多阈值PR曲线计算）：0.05 到 0.5
    min_thresh = 0.05
    max_thresh = 0.5
    
    # 使用多阈值方法计算mAP并绘制PR曲线
    if num_anomaly_images == 0:
        mAP = 0.0
        best_f1_precision = 0.0
        best_f1_recall = 0.0
        best_f1 = 0.0
        if logger is not None:
            logger.warning("没有异常样本，无法计算 mAP")
    else:
        # 使用多阈值方法计算mAP（阈值范围：0.05 到 0.5）
        mAP, best_f1_precision, best_f1_recall, best_f1, pr_data = compute_map_with_multiple_thresholds_and_plot(
            gt_masks,
            pr_maps,
            gt_anomalys,
            threshold_range=(min_thresh, max_thresh),
            threshold_step=0.01,
            iou_thresh=iou_thresh,
            save_path=save_path,
            logger=logger,
            specified_threshold=box_thresh  # 传入指定阈值，用于在图上标记
        )
        
        # 获取指定box_thresh对应的precision和recall
        # 如果在阈值范围内，从pr_data中获取；否则直接计算
        recalls_all, precisions_all, thresholds_all = pr_data
        if min_thresh <= box_thresh <= max_thresh and len(thresholds_all) > 0:
            # 在阈值范围内，从已计算的数据中找到最接近的点
            closest_idx = np.argmin(np.abs(thresholds_all - box_thresh))
            specified_precision = precisions_all[closest_idx]
            specified_recall = recalls_all[closest_idx]
        else:
            # 不在阈值范围内，直接计算（使用已计算的precision和recall）
            specified_precision = precision
            specified_recall = recall
        
        if logger is not None:
            logger.info("\n" + "="*60)
            logger.info("PR曲线分析结果（基于多阈值计算，阈值范围：0.05-0.5）:")
            logger.info("="*60)
            logger.info(f"mAP (使用average_precision_score计算): {mAP * 100:.4f}%")
            logger.info("")
            logger.info("F1最大时的指标:")
            logger.info(f"  Precision: {best_f1_precision * 100:.4f}%")
            logger.info(f"  Recall: {best_f1_recall * 100:.4f}%")
            logger.info(f"  F1分数: {best_f1:.4f}")
            logger.info("")
            logger.info(f"指定阈值 (--box_thresh={box_thresh}) 下的指标:")
            logger.info(f"  Precision: {specified_precision * 100:.4f}%")
            logger.info(f"  Recall: {specified_recall * 100:.4f}%")
            logger.info("="*60)

    return mAP, precision, recall, TP, FP, FN


def apply_heatmap(image, scoremap, alpha=0.5):
    """将热力图叠加到图像上
    Args:
        image: BGR格式的图像
        scoremap: 分数图（可以是任意范围，函数内部会归一化到0-1用于显示）
        alpha: 叠加透明度
    Returns:
        BGR格式的叠加图像
    """
    np_image = np.asarray(image, dtype=float)
    # 归一化到0-1范围（仅用于显示）
    scoremap_min = scoremap.min()
    scoremap_max = scoremap.max()
    if scoremap_max > scoremap_min:
        scoremap_norm = (scoremap - scoremap_min) / (scoremap_max - scoremap_min)
    else:
        scoremap_norm = scoremap
    scoremap = (scoremap_norm * 255).astype(np.uint8)
    # cv2.applyColorMap返回的是BGR格式，不需要转换
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    # 直接混合（都是BGR格式）
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize_results(results_eval, dataset_dir, save_path, img_size, logger, box_thresh=0.24):
    """生成并保存可视化结果：按照reference、query、预测结果的顺序拼接为一张图片"""
    logger.info("\n" + "="*80)
    logger.info("开始生成可视化结果...")
    
    query_paths_array = results_eval['query_paths']
    pr_masks_tensor = results_eval['pr_masks']
    gt_masks_tensor = results_eval['gt_masks']
    pr_anomalys_tensor = results_eval.get('pr_anomalys', None)
    
    # 创建可视化保存目录
    vis_save_dir = os.path.join(save_path, 'visualizations')
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # 为每个图像生成可视化
    num_samples = len(query_paths_array)
    
    for idx in tqdm(range(num_samples), desc="生成可视化图像"):
        img_path = query_paths_array[idx]
        
        # 从原始路径读取query图像（获取真实尺寸）
        full_img_path = os.path.join(dataset_dir, str(img_path))
        if not os.path.exists(full_img_path):
            logger.warning(f"图像文件不存在: {full_img_path}，跳过可视化")
            continue
        
        try:
            query_img_pil = Image.open(full_img_path).convert('RGB')
            query_img = np.array(query_img_pil)
            query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
            orig_h, orig_w = query_img.shape[:2]
        except Exception as e:
            logger.warning(f"无法读取图像 {full_img_path}: {e}，跳过可视化")
            continue
        
        # 获取预测热力图
        if torch.is_tensor(pr_masks_tensor):
            pred_map = pr_masks_tensor[idx].cpu().numpy()
        else:
            pred_map = pr_masks_tensor[idx]
        
        # 获取GT mask
        if torch.is_tensor(gt_masks_tensor):
            gt_mask = gt_masks_tensor[idx].cpu().numpy()
        else:
            gt_mask = gt_masks_tensor[idx]
        
        # 如果mask是多维的，squeeze
        if len(pred_map.shape) > 2:
            pred_map = pred_map.squeeze()
        if len(gt_mask.shape) > 2:
            gt_mask = gt_mask.squeeze()
        
        # 获取异常分数
        image_anomaly_score = None
        if pr_anomalys_tensor is not None:
            if torch.is_tensor(pr_anomalys_tensor):
                image_anomaly_score = float(pr_anomalys_tensor[idx].cpu().numpy())
            else:
                image_anomaly_score = float(pr_anomalys_tensor[idx])
        
        # 计算像素级最大异常值（使用原始pred_map，未调整大小）
        pixel_max_anomaly = float(np.max(pred_map))
        pixel_mean_anomaly = float(np.mean(pred_map))
        
        # 调整热力图大小以匹配原始图像
        pred_map_resized = cv2.resize(pred_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        gt_mask_resized = cv2.resize(gt_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        # （可视化）直接使用插值后的预测图，不再额外归一化
        pred_vis = apply_heatmap(query_img_bgr.copy(), pred_map_resized, alpha=0.5)
        
        # 在预测结果上绘制GT轮廓（绿色）
        if gt_mask_resized.max() > 0:
            gt_mask_uint8 = (gt_mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(gt_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cv2.drawContours(pred_vis, contours, -1, (0, 255, 0), 2)
        
        # ===== Box-level visualization =====
        # 使用统一阈值（不归一化，直接使用原始预测值）
        pred_bin = (pred_map_resized >= box_thresh).astype(np.uint8)

        pred_boxes = mask_to_boxes(pred_bin)
        gt_boxes = mask_to_boxes(gt_mask_resized)
        
        # 过滤掉面积太小的boxes（用于可视化）
        pred_boxes = filter_boxes_by_area(pred_boxes, orig_h, orig_w, min_area_ratio=0.002)
        gt_boxes = filter_boxes_by_area(gt_boxes, orig_h, orig_w, min_area_ratio=0.002)

        # GT box：绿色
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(pred_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Pred box：红色
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(pred_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 构建保存路径
        # 从img_path提取类别和文件名
        path_str = str(img_path)
        path_parts = path_str.replace('\\', '/').split('/')
        if 'anomaly_query' in path_parts:
            cls_name_idx = path_parts.index('anomaly_query') - 1
            cls_name = path_parts[cls_name_idx] if cls_name_idx >= 0 else 'unknown'
        elif 'normal_query' in path_parts:
            cls_name_idx = path_parts.index('normal_query') - 1
            cls_name = path_parts[cls_name_idx] if cls_name_idx >= 0 else 'unknown'
        else:
            cls_name = 'unknown'
        
        filename = os.path.basename(path_str)
        filename_base = os.path.splitext(filename)[0]
        
        # 加载reference图像
        reference_dir = os.path.join(dataset_dir, cls_name, 'reference')
        reference_img_bgr = None
        if os.path.exists(reference_dir):
            ref_files = [f for f in os.listdir(reference_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(ref_files) > 0:
                ref_files.sort()
                ref_path = os.path.join(reference_dir, ref_files[0])
                try:
                    ref_img_pil = Image.open(ref_path).convert('RGB')
                    ref_img = np.array(ref_img_pil)
                    reference_img_bgr = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
                    # 调整reference图像大小以匹配query图像
                    reference_img_bgr = cv2.resize(reference_img_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.warning(f"无法读取reference图像 {ref_path}: {e}")
        
        # 如果reference图像不存在，使用黑色图像占位
        if reference_img_bgr is None:
            reference_img_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            # 在黑色图像上添加文字提示
            cv2.putText(reference_img_bgr, 'No Reference', (orig_w//4, orig_h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # 在每张图片上方添加标签
        label_height = 30
        # 创建带标签的图片
        labeled_img = np.zeros((orig_h + label_height, orig_w * 3, 3), dtype=np.uint8)
        labeled_img[label_height:, :orig_w] = reference_img_bgr
        labeled_img[label_height:, orig_w:2*orig_w] = query_img_bgr
        labeled_img[label_height:, 2*orig_w:] = pred_vis
        
        # 计算文字位置（居中）
        def get_text_size(text, font, scale, thickness):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            return text_width, text_height
        
        # 添加标签文字（居中显示）
        ref_text = 'Reference'
        query_text = 'Query'
        pred_text = 'Prediction'
        
        ref_w, _ = get_text_size(ref_text, font, font_scale, thickness)
        query_w, _ = get_text_size(query_text, font, font_scale, thickness)
        pred_w, _ = get_text_size(pred_text, font, font_scale, thickness)
        
        cv2.putText(labeled_img, ref_text, (orig_w//2 - ref_w//2, 20), font, font_scale, color, thickness)
        cv2.putText(labeled_img, query_text, (orig_w + orig_w//2 - query_w//2, 20), font, font_scale, color, thickness)
        cv2.putText(labeled_img, pred_text, (2*orig_w + orig_w//2 - pred_w//2, 20), font, font_scale, color, thickness)
        
        # 在预测结果图片上标注异常分数
        score_font = cv2.FONT_HERSHEY_SIMPLEX
        score_font_scale = 0.6
        score_thickness = 2
        score_color = (0, 255, 255)  # 黄色 (BGR格式)
        
        # 在预测结果图片的左上角添加异常分数信息
        score_y_offset = label_height + 25
        score_x_start = 2 * orig_w + 10
        
        if image_anomaly_score is not None:
            score_text = f'Image Score: {image_anomaly_score:.4f}'
            cv2.putText(labeled_img, score_text, (score_x_start, score_y_offset), 
                       score_font, score_font_scale, score_color, score_thickness)
        
        # 添加像素级最大异常值
        max_text = f'Pixel Max: {pixel_max_anomaly:.4f}'
        cv2.putText(labeled_img, max_text, (score_x_start, score_y_offset + 25), 
                   score_font, score_font_scale, score_color, score_thickness)
        
        # 添加像素级平均异常值
        mean_text = f'Pixel Mean: {pixel_mean_anomaly:.4f}'
        cv2.putText(labeled_img, mean_text, (score_x_start, score_y_offset + 50), 
                   score_font, score_font_scale, score_color, score_thickness)
        
        # 按类别组织目录
        cls_save_dir = os.path.join(vis_save_dir, cls_name)
        os.makedirs(cls_save_dir, exist_ok=True)
        
        # 保存拼接后的可视化结果
        vis_filename = f"{filename_base}_visualization.png"
        vis_filepath = os.path.join(cls_save_dir, vis_filename)
        cv2.imwrite(vis_filepath, labeled_img)
    
    logger.info(f"可视化结果已保存到: {vis_save_dir}")
    logger.info(f"共生成 {num_samples} 张可视化图像")


def extract_prefix_from_path(img_path):
    """从图片路径中提取前缀"""
    # 处理路径，可能是绝对路径或相对路径
    if isinstance(img_path, bytes):
        img_path = img_path.decode('utf-8')
    elif isinstance(img_path, np.ndarray):
        img_path = str(img_path)
    
    filename = os.path.basename(img_path)
    basename = os.path.splitext(filename)[0]
    # 匹配模式: 前缀_result_数字
    match = re.match(r'^(.+?)_result_\d+$', basename)
    if match:
        return match.group(1)
    return None


def prompt_association(image_memory, patch_memory, target_class_name):
    patch_level_num = len(patch_memory[target_class_name[0]])
    retrive_image = []
    retrive_patch = [[] for i in range(patch_level_num)]

    for class_name in target_class_name:
        retrive_image.append(image_memory[class_name])
        for l in range(patch_level_num):
            retrive_patch[l].append(patch_memory[class_name][l])

    retrive_image = torch.stack(retrive_image)
    for l in range(patch_level_num):
        retrive_patch[l] = torch.stack(retrive_patch[l])
    return retrive_image, retrive_patch


def build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, features_list, DPAM_layer):
    """Build few-shot prompt memory."""
    feats_scale_num = len(features_list)
    prompt_image_memory = {}
    prompt_patch_memory = {}

    image_temp = []
    patch_temp = [[] for i in range(feats_scale_num)]
    cls_names_temp = []
    view_ids_temp = []

    for idx, items in enumerate(tqdm(prompt_dataloader)):
        cls_name = items['cls_name']
        prompt_image = items['img'].to(device)
        prompt_mask = items['img_mask'].to(device)
        view_id = items['view_id']

        with torch.no_grad():
            image_feat, patch_feat = model.encode_image(prompt_image, features_list, DPAM_layer=DPAM_layer)

        cls_names_temp.extend(cls_name)
        image_temp.append(image_feat)
        view_ids_temp.extend(view_id)

        for i in range(feats_scale_num):
            patch_temp[i].append(patch_feat[i])

    image_temp = torch.cat(image_temp, dim=0)
    for i in range(feats_scale_num):
        patch_temp[i] = torch.cat(patch_temp[i], dim=0)

    for obj in obj_list:
        if len(view_list) > 1:
            for view_id in view_list:
                indice = (np.array(cls_names_temp) == obj) & (np.array(view_ids_temp) == view_id)
                obj_name = obj + '_' + view_id

                prompt_image_memory[obj_name] = image_temp[indice]
                prompt_patch_memory[obj_name] = []

                for i in range(feats_scale_num):
                    prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])
        else:
            indice = (np.array(cls_names_temp) == obj)
            obj_name = obj

            prompt_image_memory[obj_name] = image_temp[indice]
            prompt_patch_memory[obj_name] = []

            for i in range(feats_scale_num):
                prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])

    return prompt_image_memory, prompt_patch_memory


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.test_data_path
    save_path = args.save_path
    dataset_name = args.dataset
    batch_size = args.batch_size
    k_shots = args.k_shots
    seed = args.seed
    vl_reduction = args.vl_reduction
    pq_mid_dim = args.pq_mid_dim
    pq_context = args.pq_context
    eval_metrics = args.eval_metrics
    mode = 'test'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger = get_logger(save_path, log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = adapterlib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer=20)
        patch_size = 14
        input_dim = 768
        DPAM_layer = 20
    elif args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = adapterlib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer=10)
        patch_size = 16
        input_dim = 640
        DPAM_layer = 10

    preprocess, target_transform = get_transform(image_size=args.image_size)
    
    prompt_data = PromptDataset(root=dataset_dir, transform=preprocess, target_transform=target_transform,
                                dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed)
    test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=target_transform,
                        dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed)
    
    sample_level = False
    prompt_dataloader = torch.utils.data.DataLoader(prompt_data, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    obj_list = test_data.obj_list
    view_list = test_data.view_list

    # ====================== Init Adapters ======================
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualAdapter(img_size, patch_size, input_dim=input_dim, reduction=vl_reduction)
    pq_learner = PQAdapter(img_size, patch_size, context=pq_context, input_dim=input_dim, mid_dim=pq_mid_dim, layers_num=len(features_list))

    logger.info('\n' + "loading model from: " + args.checkpoint_path)
    checkpoint_adapter = torch.load(args.checkpoint_path)
    textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
    visual_learner.load_state_dict(checkpoint_adapter["visual_learner"])
    pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])

    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)

    model.eval()
    textual_learner.eval()
    visual_learner.eval()
    pq_learner.eval()

    # ====================== Initialize Evaluation Metrics ======================
    evaluator = Evaluator(device, metrics=eval_metrics, sample_level=sample_level)

    # ======================Text Encoder forward ======================
    textual_learner.prepare_static_text_feature(model)
    static_text_features = textual_learner.static_text_features

    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()

    # ====================== Few-shot Prompt Memory ======================
    if k_shots > 0:
        prompt_image_memory, prompt_patch_memory = build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, args.features_list, DPAM_layer)

    # ====================== Visual and Learner forward ======================
    sample_ids, gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], [], []
    
    for idx, items in enumerate(tqdm(test_dataloader)):
        query_image = items['img'].to(device)
        current_batchsize = query_image.shape[0]
        query_path = items['img_path']

        cls_name = items['cls_name']
        cls_id = items['cls_id']
        sample_id = items['sample_id']

        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['img_mask'][:, 0]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        gt_mask = gt_mask.to(device)

        with torch.no_grad():
            query_feats, query_patch_feats = model.encode_image(query_image, args.features_list, DPAM_layer=DPAM_layer)

        if k_shots > 0:
            if len(view_list) > 1:
                target_cls_name = [cls_name + '_' + view_id for cls_name, view_id in zip(cls_name, items['view_id'])]
            else:
                target_cls_name = cls_name
            prompt_feats, prompt_patch_feats = prompt_association(prompt_image_memory, prompt_patch_memory, target_cls_name)

        # ====================== visual_adapter ======================
        if args.visual_learner:
            global_vl_logit, local_vl_map = visual_learner(query_feats, query_patch_feats, static_text_features)
            local_vl_map = local_vl_map[:, 1].detach()

            global_vl_score = global_vl_logit.softmax(-1)
            global_vl_score = global_vl_score[:, 1].detach()

        # ====================== textual_adapter ======================
        if args.textual_learner:
            global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)
            local_tl_map = local_tl_map[:, 1].detach()

            global_tl_score = global_tl_logit.softmax(-1)
            global_tl_score = global_tl_score[:, 1].detach()

        # ====================== pq_adapter ======================
        if args.pq_learner and k_shots > 0:
            global_pq_logit, local_pq_map_list, align_score_list = pq_learner(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)

            local_pq_map_list = [x[:, 1].unsqueeze(1) for x in local_pq_map_list]
            local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
            align_score = fusion_fun(align_score_list, fusion_type='harmonic_mean')[:, 0]

            if isinstance(global_pq_logit, list):
                global_pq_score = [x.softmax(-1).unsqueeze(-1) for x in global_pq_logit]
                global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1).detach()
                global_pq_score = global_pq_score[:, 1].detach()
            else:
                global_pq_score = global_pq_logit.softmax(-1)
                global_pq_score = global_pq_score[:, 1].detach()

        if k_shots > 0:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map], fusion_type=args.fusion_type)
            pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type='harmonic_mean')
            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in pixel_anomaly_map.cpu()], dim=0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, global_pq_score], fusion_type=args.fusion_type)
            image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type="harmonic_mean")
        else:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map], fusion_type=args.fusion_type)
            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in pixel_anomaly_map.cpu()], dim=0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, anomaly_map_max], fusion_type=args.fusion_type)

        pixel_anomaly_map = torch.nan_to_num(pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
        image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)

        sample_ids.append(np.array(sample_id))
        cls_names.append(np.array(cls_name))
        query_paths.append(np.array(query_path))
        gt_masks.append(gt_mask.int())
        pr_masks.append(pixel_anomaly_map)
        gt_anomalys.append(gt_anomaly.int())
        pr_anomalys.append(image_anomaly_pred)

    # ====================== Evaluation ======================
    results_eval = dict(
        sample_ids=sample_ids,
        gt_masks=gt_masks,
        pr_masks=pr_masks,
        cls_names=cls_names,
        gt_anomalys=gt_anomalys,
        pr_anomalys=pr_anomalys,
        query_paths=query_paths,
    )
    results_eval = {
        k: np.concatenate(v, axis=0) if k in ['cls_names', 'query_paths', 'sample_ids']
        else torch.cat(v, dim=0)
        for k, v in results_eval.items()
    }




    # ====================== 保存分布图所需的分数和标签 ======================
    # image-level anomaly scores
    if torch.is_tensor(results_eval['pr_anomalys']):
        anomaly_scores_np = results_eval['pr_anomalys'].detach().cpu().numpy().ravel()
    else:
        anomaly_scores_np = np.array(results_eval['pr_anomalys']).ravel()

    # image-level ground-truth labels (0: normal, 1: anomaly)
    if torch.is_tensor(results_eval['gt_anomalys']):
        labels_np = results_eval['gt_anomalys'].detach().cpu().numpy().ravel()
    else:
        labels_np = np.array(results_eval['gt_anomalys']).ravel()

    np.save(os.path.join(save_path, "anomaly_scores.npy"), anomaly_scores_np)
    np.save(os.path.join(save_path, "labels.npy"), labels_np)
    
    # ====================== Prefix-based Evaluation ======================
    # 按前缀分组统计（只统计anomaly_query的前缀）
    prefix_groups = defaultdict(lambda: {'indices': [], 'prefix': None})
    
    query_paths_array = results_eval['query_paths']
    gt_anomalys_array = results_eval['gt_anomalys']
    
    # 将tensor转换为numpy数组以便处理
    if torch.is_tensor(gt_anomalys_array):
        gt_anomalys_np = gt_anomalys_array.cpu().numpy()
    else:
        gt_anomalys_np = gt_anomalys_array
    
    for idx in range(len(query_paths_array)):
        img_path = query_paths_array[idx]
        anomaly = int(gt_anomalys_np[idx])
        
        if anomaly == 1:  # 只统计异常样本
            prefix = extract_prefix_from_path(img_path)
            if prefix:
                prefix_groups[prefix]['indices'].append(idx)
                prefix_groups[prefix]['prefix'] = prefix

    # 计算各前缀的seg AUC
    prefix_seg_aucs = {}
    for prefix, group in prefix_groups.items():
        indices = np.array(group['indices'])
        if len(indices) == 0:
            continue
        
        gt_px = results_eval['gt_masks'][indices]
        pr_px = results_eval['pr_masks'][indices]
        
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        
        # 转换为numpy数组
        if torch.is_tensor(gt_px):
            gt_px_np = gt_px.cpu().numpy().ravel()
        else:
            gt_px_np = gt_px.ravel()
            
        if torch.is_tensor(pr_px):
            pr_px_np = pr_px.cpu().numpy().ravel()
        else:
            pr_px_np = pr_px.ravel()
        
        # 检查数据有效性
        if len(gt_px_np) == 0 or len(pr_px_np) == 0:
            logger.warning(f"前缀 {prefix} 的数据为空，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(gt_px_np)) or np.any(np.isnan(pr_px_np)):
            logger.warning(f"前缀 {prefix} 的数据包含NaN，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
            
        if np.any(np.isinf(gt_px_np)) or np.any(np.isinf(pr_px_np)):
            logger.warning(f"前缀 {prefix} 的数据包含Inf，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 检查是否所有标签都是同一个值
        unique_labels = np.unique(gt_px_np)
        if len(unique_labels) < 2:
            logger.warning(f"前缀 {prefix} 的所有像素标签都是同一类别（{unique_labels[0]}），无法计算AUC")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 计算像素级AUC
        try:
            seg_auc = roc_auc_score(gt_px_np, pr_px_np)
            # 检查结果是否为nan
            if np.isnan(seg_auc):
                logger.warning(f"前缀 {prefix} 的seg AUC计算结果为nan（可能是预测值全相同）")
                prefix_seg_aucs[prefix] = 0.0
            else:
                prefix_seg_aucs[prefix] = seg_auc
        except ValueError as e:
            logger.warning(f"无法计算前缀 {prefix} 的seg AUC: {e}")
            prefix_seg_aucs[prefix] = 0.0

    # 计算overall average seg AUC（所有异常样本）
    gt_anomalys_tensor = results_eval['gt_anomalys']
    if torch.is_tensor(gt_anomalys_tensor):
        all_anomaly_indices = np.where(gt_anomalys_tensor.cpu().numpy() == 1)[0]
    else:
        all_anomaly_indices = np.where(gt_anomalys_tensor == 1)[0]
    
    if len(all_anomaly_indices) > 0:
        gt_px_all = results_eval['gt_masks'][all_anomaly_indices]
        pr_px_all = results_eval['pr_masks'][all_anomaly_indices]
        
        if len(gt_px_all.shape) == 4:
            gt_px_all = gt_px_all.squeeze(1)
        if len(pr_px_all.shape) == 4:
            pr_px_all = pr_px_all.squeeze(1)
        
        # 转换为numpy数组
        if torch.is_tensor(gt_px_all):
            gt_px_all_np = gt_px_all.cpu().numpy().ravel()
        else:
            gt_px_all_np = gt_px_all.ravel()
            
        if torch.is_tensor(pr_px_all):
            pr_px_all_np = pr_px_all.cpu().numpy().ravel()
        else:
            pr_px_all_np = pr_px_all.ravel()
        
        # 检查数据有效性
        if len(gt_px_all_np) == 0 or len(pr_px_all_np) == 0:
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC数据为空，设为0.0")
        elif np.any(np.isnan(gt_px_all_np)) or np.any(np.isnan(pr_px_all_np)):
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC数据包含NaN，设为0.0")
        elif np.any(np.isinf(gt_px_all_np)) or np.any(np.isinf(pr_px_all_np)):
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC数据包含Inf，设为0.0")
        elif len(np.unique(gt_px_all_np)) < 2:
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC所有标签都是同一类别，设为0.0")
        else:
            try:
                overall_seg_auc = roc_auc_score(gt_px_all_np, pr_px_all_np)
                if np.isnan(overall_seg_auc):
                    overall_seg_auc = 0.0
                    logger.warning("overall seg AUC计算结果为nan，设为0.0")
            except ValueError as e:
                overall_seg_auc = 0.0
                logger.warning(f"无法计算overall seg AUC: {e}")
    else:
        overall_seg_auc = 0.0

    # 计算overall average image AUC（所有样本，包括正常和异常）
    try:
        if torch.is_tensor(results_eval['gt_anomalys']):
            gt_anomalys_np = results_eval['gt_anomalys'].cpu().numpy()
            pr_anomalys_np = results_eval['pr_anomalys'].cpu().numpy()
        else:
            gt_anomalys_np = results_eval['gt_anomalys']
            pr_anomalys_np = results_eval['pr_anomalys']
        overall_image_auc = roc_auc_score(gt_anomalys_np, pr_anomalys_np)
    except ValueError:
        overall_image_auc = 0.0
        logger.warning("无法计算overall image AUC")

    # ====================== 输出结果 ======================
    logger.info("\n" + "="*80)
    logger.info("Prefix-based Evaluation Results")
    logger.info("="*80)
    
    # 输出各前缀的seg AUC
    msg = {}
    msg['Prefix'] = []
    msg['Seg-AUC'] = []
    
    for prefix in sorted(prefix_seg_aucs.keys()):
        msg['Prefix'].append(prefix)
        msg['Seg-AUC'].append(prefix_seg_aucs[prefix] * 100)
    
    # 计算平均seg AUC
    if len(prefix_seg_aucs) > 0:
        avg_seg_auc = np.mean(list(prefix_seg_aucs.values())) * 100
        msg['Prefix'].append('Average')
        msg['Seg-AUC'].append(avg_seg_auc)
    
    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.2f', numalign="center", stralign="center")
    logger.info('\n' + tab)
    
    # 输出Overall结果
    logger.info("\n" + "="*80)
    logger.info("Overall Results")
    logger.info("="*80)
    logger.info(f"Overall Average Seg AUC: {overall_seg_auc * 100:.2f}%")
    logger.info(f"Overall Average Image AUC: {overall_image_auc * 100:.2f}%")
    
    if len(prefix_seg_aucs) > 0:
        logger.info(f"Average Seg AUC (across prefixes): {avg_seg_auc:.2f}%")
    
    # ====================== Box-level mAP ======================
    logger.info("\n" + "="*80)
    logger.info("Box-level Detection Evaluation (mAP)")
    logger.info("="*80)
    
    # 统计异常样本数量
    if torch.is_tensor(results_eval['gt_anomalys']):
        num_anomaly_samples = int((results_eval['gt_anomalys'] == 1).sum().item())
    else:
        num_anomaly_samples = int((results_eval['gt_anomalys'] == 1).sum())
    logger.info(f"异常样本数量: {num_anomaly_samples}")
    
    if num_anomaly_samples == 0:
        logger.warning("警告: 没有异常样本，无法计算Box mAP")
        mAP, box_precision, box_recall, TP, FP, FN = 0.0, 0.0, 0.0, 0, 0, 0
    else:
        mAP, box_precision, box_recall, TP, FP, FN = compute_box_map(
            results_eval['gt_masks'],
            results_eval['pr_masks'],
            results_eval['gt_anomalys'],
            box_thresh=args.box_thresh,
            iou_thresh=0.2,
            logger=logger,
            save_path=save_path  # 传入save_path以保存PR曲线图
        )
    
    logger.info(f"Box mAP@0.2: {mAP * 100:.2f}%")
    logger.info(f"Box Precision: {box_precision * 100:.2f}%")
    logger.info(f"Box Recall:    {box_recall * 100:.2f}%")
    logger.info(f"TP={TP}, FP={FP}, FN={FN}")
    
    if TP == 0 and FP == 0 and FN == 0:
        logger.warning("警告: TP=FP=FN=0，可能原因：")
        logger.warning("  1. 没有异常样本")
        logger.warning(f"  2. 预测图最大值 < {args.box_thresh}（box_thresh），导致没有预测框")
        logger.warning("  3. GT mask中没有异常区域，导致没有GT框")
    
    # ====================== Prefix-based Box-level Evaluation ======================
    logger.info("\n" + "="*80)
    logger.info("Prefix-based Box-level Detection Evaluation")
    logger.info("="*80)
    
    prefix_box_metrics = {}
    for prefix, group in prefix_groups.items():
        indices = np.array(group['indices'])
        if len(indices) == 0:
            continue
        
        # 提取该前缀对应的数据
        prefix_gt_masks = results_eval['gt_masks'][indices]
        prefix_pr_masks = results_eval['pr_masks'][indices]
        prefix_gt_anomalys = results_eval['gt_anomalys'][indices]
        
        # 检查是否有异常样本
        if torch.is_tensor(prefix_gt_anomalys):
            num_anomaly = int((prefix_gt_anomalys == 1).sum().item())
        else:
            num_anomaly = int((prefix_gt_anomalys == 1).sum())
        
        if num_anomaly == 0:
            logger.warning(f"前缀 {prefix} 没有异常样本，跳过Box指标计算")
            prefix_box_metrics[prefix] = {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'TP': 0,
                'FP': 0,
                'FN': 0
            }
            continue
        
        # 计算该前缀的box指标（不保存PR曲线图）
        prefix_mAP, prefix_precision, prefix_recall, prefix_TP, prefix_FP, prefix_FN = compute_box_map(
            prefix_gt_masks,
            prefix_pr_masks,
            prefix_gt_anomalys,
            box_thresh=args.box_thresh,
            iou_thresh=0.2,
            logger=None,  # 不输出详细日志
            save_path=None  # 不保存PR曲线图
        )
        
        prefix_box_metrics[prefix] = {
            'mAP': prefix_mAP,
            'precision': prefix_precision,
            'recall': prefix_recall,
            'TP': prefix_TP,
            'FP': prefix_FP,
            'FN': prefix_FN
        }
    
    # 输出各前缀的box指标表格
    if len(prefix_box_metrics) > 0:
        msg = {}
        msg['Prefix'] = []
        msg['mAP@0.2'] = []
        msg['Precision'] = []
        msg['Recall'] = []
        msg['TP'] = []
        msg['FP'] = []
        msg['FN'] = []
        
        for prefix in sorted(prefix_box_metrics.keys()):
            metrics = prefix_box_metrics[prefix]
            msg['Prefix'].append(prefix)
            msg['mAP@0.2'].append(metrics['mAP'] * 100)
            msg['Precision'].append(metrics['precision'] * 100)
            msg['Recall'].append(metrics['recall'] * 100)
            msg['TP'].append(metrics['TP'])
            msg['FP'].append(metrics['FP'])
            msg['FN'].append(metrics['FN'])
        
        # 计算平均指标
        avg_mAP = np.mean([metrics['mAP'] for metrics in prefix_box_metrics.values()]) * 100
        avg_precision = np.mean([metrics['precision'] for metrics in prefix_box_metrics.values()]) * 100
        avg_recall = np.mean([metrics['recall'] for metrics in prefix_box_metrics.values()]) * 100
        total_TP = sum([metrics['TP'] for metrics in prefix_box_metrics.values()])
        total_FP = sum([metrics['FP'] for metrics in prefix_box_metrics.values()])
        total_FN = sum([metrics['FN'] for metrics in prefix_box_metrics.values()])
        
        msg['Prefix'].append('Average')
        msg['mAP@0.2'].append(avg_mAP)
        msg['Precision'].append(avg_precision)
        msg['Recall'].append(avg_recall)
        msg['TP'].append(total_TP)
        msg['FP'].append(total_FP)
        msg['FN'].append(total_FN)
        
        tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.2f', numalign="center", stralign="center")
        logger.info('\n' + tab)
    
    # ====================== 生成可视化结果 ======================
    if args.visualize:
        visualize_results(results_eval, dataset_dir, save_path, img_size, logger, box_thresh=args.box_thresh)
    
    # 确保所有日志被写入文件
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("adapter Custom Dataset Testing", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, required=True, help="path to custom dataset root")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='custom_dataset')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'P-AUROC'], help='evaluation metrics')
    parser.add_argument("--fusion_type", type=str, default="average_mean", help='fusion type')
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization results with heatmaps")
    parser.add_argument("--box_thresh", type=float, default=0.24, help="Box detection threshold (absolute value, no normalization)")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    try:
        test(args)
    finally:
        # 确保所有日志被写入文件
        import logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.flush()
            handler.close()
        
        test_logger = logging.getLogger('test')
        for handler in test_logger.handlers[:]:
            handler.flush()
            handler.close()

