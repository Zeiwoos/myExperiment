"""Testing script for adapter on custom dataset with prefix-based evaluation."""

import argparse
import os
import re
from collections import defaultdict
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, label as cc_label
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple

import adapterlib
from adapterlib import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from dataset import Dataset, PromptDataset
from tools import Evaluator, get_logger, get_transform, setup_seed
from tools.utils import normalize


# ============================================================================
# MVTec AD 2 Metrics Implementation (AU-PRO & Pixel-level F1)
# ============================================================================

def _as_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array has shape (N, H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W) or (N,H,W), got shape={arr.shape}")
    return arr


def compute_pro_curve(
    anomaly_maps: np.ndarray,
    gt_maps: np.ndarray,
    *,
    connectivity: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the PRO-vs-FPR curve.

    PRO(t) = mean over GT connected components C of |P(t) ∩ C| / |C|
    FPR(t) = FP(t) / N_ok_pixels

    Efficient exact curve:
    - GT=0 pixel contributes FP += 1
    - Pixel in GT component C contributes PRO += 1/|C|
    Sort pixels by anomaly score (desc) and cumulative-sum contributions.
    """
    scores = _as_3d(np.asarray(anomaly_maps, dtype=np.float64))
    gt = _as_3d(np.asarray(gt_maps, dtype=np.uint8))
    if scores.shape != gt.shape:
        raise ValueError(f"Shape mismatch: scores={scores.shape}, gt={gt.shape}")

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    elif connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        raise ValueError("connectivity must be 4 or 8")

    fp_changes = np.zeros_like(gt, dtype=np.uint32)
    pro_changes = np.zeros_like(scores, dtype=np.float64)

    num_ok_pixels = 0
    num_gt_regions = 0

    # Build per-pixel "change" tensors.
    for i in range(gt.shape[0]):
        labeled, n_comp = cc_label(gt[i], structure=structure)
        num_gt_regions += int(n_comp)

        ok_mask = labeled == 0
        num_ok_pixels += int(ok_mask.sum())
        fp_changes[i][ok_mask] = 1

        for comp_id in range(1, n_comp + 1):
            region = labeled == comp_id
            region_size = int(region.sum())
            if region_size > 0:
                pro_changes[i][region] = 1.0 / region_size

    if num_ok_pixels == 0:
        raise ValueError("No GT=0 pixels found; FPR is undefined.")
    if num_gt_regions == 0:
        raise ValueError("No GT anomaly regions found; PRO/AU-PRO is undefined.")

    # Flatten & sort by score (descending).
    scores_flat = scores.ravel()
    fp_flat = fp_changes.ravel()
    pro_flat = pro_changes.ravel()

    order = np.argsort(scores_flat)[::-1]
    scores_sorted = scores_flat[order]
    fp_sorted = fp_flat[order]
    pro_sorted = pro_flat[order]

    # Cumulative sums give curve values.
    fprs = np.cumsum(fp_sorted, dtype=np.float64) / num_ok_pixels
    pros = np.cumsum(pro_sorted, dtype=np.float64) / num_gt_regions

    # Keep only the last point for each unique threshold (score value).
    keep = np.append(scores_sorted[:-1] != scores_sorted[1:], True)
    fprs = fprs[keep]
    pros = pros[keep]

    fprs = np.clip(fprs, 0.0, 1.0)
    pros = np.clip(pros, 0.0, 1.0)

    # Start at (0,0), end at (1,1)
    fprs = np.concatenate([[0.0], fprs, [1.0]])
    pros = np.concatenate([[0.0], pros, [1.0]])
    return fprs, pros


def _trapezoid_auc(x: np.ndarray, y: np.ndarray, *, x_max: float | None = None) -> float:
    """Trapezoidal area under curve y(x), optionally truncated at x_max."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x_max is None:
        return float(np.trapz(y, x))

    x_max = float(x_max)
    if x_max <= x[0]:
        return 0.0
    if x_max >= x[-1]:
        return float(np.trapz(y, x))

    idx = int(np.searchsorted(x, x_max, side="right") - 1)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    y_max = y1 if x1 == x0 else (y0 + (y1 - y0) * (x_max - x0) / (x1 - x0))

    x_trunc = np.concatenate([x[: idx + 1], [x_max]])
    y_trunc = np.concatenate([y[: idx + 1], [y_max]])
    return float(np.trapz(y_trunc, x_trunc))


def au_pro(anomaly_maps: np.ndarray, gt_maps: np.ndarray, *, max_fpr: float) -> float:
    """Compute normalized AU-PRO@max_fpr (range ~[0,1])."""
    fprs, pros = compute_pro_curve(anomaly_maps, gt_maps)
    area = _trapezoid_auc(fprs, pros, x_max=max_fpr)
    return area / float(max_fpr)


@dataclass
class F1Result:
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def threshold_mean_plus_3std(defect_free_val_maps: np.ndarray) -> float:
    """Paper baseline: mean + 3*std over ALL pixels of defect-free validation maps."""
    x = np.asarray(defect_free_val_maps, dtype=np.float64)
    return float(x.mean() + 3.0 * x.std(ddof=0))


def pixel_f1(anomaly_maps: np.ndarray, gt_maps: np.ndarray, *, threshold: float) -> F1Result:
    """Pixel-level F1 computed over all pixels of all images."""
    scores = _as_3d(np.asarray(anomaly_maps, dtype=np.float64))
    gt = _as_3d(np.asarray(gt_maps, dtype=np.uint8)).astype(bool)

    pred = scores >= float(threshold)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return F1Result(f1=f1, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)


# ============================================================================
# Visualization Functions
# ============================================================================

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


def visualize_results(results_eval, dataset_dir, save_path, img_size, logger):
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
        
        # 可视化：直接使用插值后的预测图，不再额外归一化
        pred_vis = apply_heatmap(query_img_bgr.copy(), pred_map_resized, alpha=0.5)
        
        # 在预测结果上绘制GT轮廓（绿色）
        if gt_mask_resized.max() > 0:
            gt_mask_uint8 = (gt_mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(gt_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cv2.drawContours(pred_vis, contours, -1, (0, 255, 0), 2)
        
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

        # Check if required variables are defined
        if not args.visual_learner:
            raise ValueError("--visual_learner is required but not enabled")
        if not args.textual_learner:
            raise ValueError("--textual_learner is required but not enabled")

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

    # ====================== Convert data for evaluation ======================
    # Convert to numpy arrays
    if torch.is_tensor(results_eval['gt_masks']):
        gt_masks_np = results_eval['gt_masks'].cpu().numpy().astype(np.uint8)
    else:
        gt_masks_np = np.array(results_eval['gt_masks']).astype(np.uint8)
    
    if torch.is_tensor(results_eval['pr_masks']):
        pr_masks_np = results_eval['pr_masks'].cpu().numpy().astype(np.float64)
    else:
        pr_masks_np = np.array(results_eval['pr_masks']).astype(np.float64)
    
    if torch.is_tensor(results_eval['gt_anomalys']):
        gt_anomalys_np = results_eval['gt_anomalys'].cpu().numpy()
    else:
        gt_anomalys_np = np.array(results_eval['gt_anomalys'])
    
    if torch.is_tensor(results_eval['pr_anomalys']):
        pr_anomalys_np = results_eval['pr_anomalys'].cpu().numpy()
    else:
        pr_anomalys_np = np.array(results_eval['pr_anomalys'])
    
    # ====================== Prefix-based Evaluation ======================
    # 按前缀分组统计（只统计anomaly_query的前缀）
    prefix_groups = defaultdict(lambda: {'indices': [], 'prefix': None})
    
    query_paths_array = results_eval['query_paths']
    
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
        
        gt_px = gt_masks_np[indices]
        pr_px = pr_masks_np[indices]
        
        gt_px_flat = gt_px.ravel()
        pr_px_flat = pr_px.ravel()
        
        # 检查数据有效性
        if len(gt_px_flat) == 0 or len(pr_px_flat) == 0:
            logger.warning(f"前缀 {prefix} 的数据为空，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(gt_px_flat)) or np.any(np.isnan(pr_px_flat)):
            logger.warning(f"前缀 {prefix} 的数据包含NaN，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
            
        if np.any(np.isinf(gt_px_flat)) or np.any(np.isinf(pr_px_flat)):
            logger.warning(f"前缀 {prefix} 的数据包含Inf，跳过AUC计算")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 检查是否所有标签都是同一个值
        unique_labels = np.unique(gt_px_flat)
        if len(unique_labels) < 2:
            logger.warning(f"前缀 {prefix} 的所有像素标签都是同一类别（{unique_labels[0]}），无法计算AUC")
            prefix_seg_aucs[prefix] = 0.0
            continue
        
        # 计算像素级AUC
        try:
            seg_auc = roc_auc_score(gt_px_flat, pr_px_flat)
            # 检查结果是否为nan
            if np.isnan(seg_auc):
                logger.warning(f"前缀 {prefix} 的seg AUC计算结果为nan（可能是预测值全相同）")
                prefix_seg_aucs[prefix] = 0.0
            else:
                prefix_seg_aucs[prefix] = seg_auc
        except ValueError as e:
            logger.warning(f"无法计算前缀 {prefix} 的seg AUC: {e}")
            prefix_seg_aucs[prefix] = 0.0

    # 计算overall metrics（所有测试集样本，包括正常和异常）
    logger.info("\n" + "="*80)
    logger.info("基于所有测试集样本计算像素级指标...")
    
    # 使用所有测试集样本
    gt_px_all = gt_masks_np
    pr_px_all = pr_masks_np
    
    gt_px_all_flat = gt_px_all.ravel()
    pr_px_all_flat = pr_px_all.ravel()
    
    # 检查是否有异常像素
    if len(gt_px_all_flat) > 0 and len(pr_px_all_flat) > 0:
        
        # 检查数据有效性并计算Pixel-level AUC (Seg AUC)
        if np.any(np.isnan(gt_px_all_flat)) or np.any(np.isnan(pr_px_all_flat)):
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC数据包含NaN，设为0.0")
        elif np.any(np.isinf(gt_px_all_flat)) or np.any(np.isinf(pr_px_all_flat)):
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC数据包含Inf，设为0.0")
        elif len(np.unique(gt_px_all_flat)) < 2:
            overall_seg_auc = 0.0
            logger.warning("overall seg AUC所有标签都是同一类别，设为0.0")
        else:
            try:
                overall_seg_auc = roc_auc_score(gt_px_all_flat, pr_px_all_flat)
                if np.isnan(overall_seg_auc):
                    overall_seg_auc = 0.0
                    logger.warning("overall seg AUC计算结果为nan，设为0.0")
                else:
                    logger.info(f"Pixel-level AUC (Seg AUC): {overall_seg_auc * 100:.2f}%")
            except ValueError as e:
                overall_seg_auc = 0.0
                logger.warning(f"无法计算overall seg AUC: {e}")
        
        # 计算AU-PRO指标
        logger.info("\n" + "="*80)
        logger.info("计算AU-PRO指标...")
        try:
            au_pro_005 = au_pro(pr_px_all, gt_px_all, max_fpr=0.05)
            au_pro_030 = au_pro(pr_px_all, gt_px_all, max_fpr=0.30)
            logger.info(f"AU-PRO@0.05: {au_pro_005 * 100:.2f}%")
            logger.info(f"AU-PRO@0.30: {au_pro_030 * 100:.2f}%")
        except Exception as e:
            au_pro_005 = 0.0
            au_pro_030 = 0.0
            logger.warning(f"无法计算AU-PRO: {e}")
        
        # 计算pixel-level F1（使用指定阈值）
        logger.info("\n" + "="*80)
        logger.info("计算pixel-level F1指标...")
        threshold = args.f1_threshold
        logger.info(f"使用指定阈值: {threshold:.6f}")
        
        # 计算pixel-level F1
        try:
            f1_result = pixel_f1(pr_px_all, gt_px_all, threshold=threshold)
            logger.info(f"Pixel-level F1: {f1_result.f1:.4f}")
            logger.info(f"Precision: {f1_result.precision:.4f}")
            logger.info(f"Recall: {f1_result.recall:.4f}")
            logger.info(f"TP={f1_result.tp}, FP={f1_result.fp}, FN={f1_result.fn}")
        except Exception as e:
            f1_result = None
            logger.warning(f"无法计算pixel-level F1: {e}")
    else:
        # 数据为空或无效
        overall_seg_auc = 0.0
        au_pro_005 = 0.0
        au_pro_030 = 0.0
        f1_result = None
        logger.warning("测试集数据为空，无法计算像素级指标")

    # 计算overall average image AUC（所有样本，包括正常和异常）
    try:
        overall_image_auc = roc_auc_score(gt_anomalys_np, pr_anomalys_np)
    except ValueError:
        overall_image_auc = 0.0
        logger.warning("无法计算overall image AUC")

    # ====================== 输出结果 ======================
    logger.info("\n" + "="*80)
    logger.info("Prefix-based Evaluation Results (仅异常样本的Pixel-level Seg AUC)")
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
    logger.info("Overall Results Summary (基于所有测试集样本)")
    logger.info("="*80)
    logger.info(f"Overall Image-level AUC: {overall_image_auc * 100:.2f}%")
    logger.info(f"Overall Pixel-level AUC (Seg AUC): {overall_seg_auc * 100:.2f}%")
    if len(prefix_seg_aucs) > 0:
        logger.info(f"Average Seg AUC (across anomaly prefixes): {avg_seg_auc:.2f}%")
    
    # 输出AU-PRO结果
    if 'au_pro_005' in locals():
        logger.info(f"\nAU-PRO@0.05: {au_pro_005 * 100:.2f}%")
        logger.info(f"AU-PRO@0.30: {au_pro_030 * 100:.2f}%")
    
    # 输出pixel-level F1结果
    if f1_result is not None:
        logger.info(f"\nPixel-level F1 (threshold={args.f1_threshold}):")
        logger.info(f"  F1: {f1_result.f1:.4f}")
        logger.info(f"  Precision: {f1_result.precision:.4f}")
        logger.info(f"  Recall: {f1_result.recall:.4f}")
        logger.info(f"  TP={f1_result.tp}, FP={f1_result.fp}, FN={f1_result.fn}")
    
    logger.info("="*80)
    
    # ====================== 生成可视化结果 ======================
    if args.visualize:
        visualize_results(results_eval, dataset_dir, save_path, img_size, logger)


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
    parser.add_argument("--f1_threshold", type=float, default=0.2, help="Threshold for pixel-level F1 calculation")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)

