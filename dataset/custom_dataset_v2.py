"""Custom dataset solver v2 for anomaly detection.
改进版本：在生成meta.json时，对每个子文件夹，先从normal_query中补齐reference至4个。
"""

import argparse
import json
import os
import re
from collections import defaultdict


class CustomDatasetSolverV2(object):
    def __init__(self, root='data/custom_dataset', target_reference_count=4):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.target_reference_count = target_reference_count  # 目标reference数量

    def extract_prefix(self, filename):
        """从文件名中提取前缀（例如从 diushi_or_songdong_result_0.png 提取 diushi_or_songdong）"""
        # 移除扩展名
        basename = os.path.splitext(filename)[0]
        # 匹配模式: 前缀_result_数字 或 前缀_mask_数字
        # 提取前缀部分（去除 _result_数字 或 _mask_数字）
        match = re.match(r'^(.+?)_(?:result|mask)_\d+$', basename)
        if match:
            return match.group(1)
        return None

    def run(self):
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0
        normal_query_test_samples = 0
        normal_query_train_samples = 0  # 从normal_query移动到train的样本数
        
        # 获取所有子目录（例如 1_D01-1-1, 1_D01-1-2 等）
        subdirs = [d for d in os.listdir(self.root) 
                   if os.path.isdir(os.path.join(self.root, d)) and not d.startswith('.')]
        subdirs.sort()
        
        # 用于按前缀分组
        prefix_groups = defaultdict(list)
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.root, subdir)
            
            # 读取anomaly_query目录
            anomaly_query_dir = os.path.join(subdir_path, 'anomaly_query')
            mask_dir = os.path.join(subdir_path, 'mask')
            normal_query_dir = os.path.join(subdir_path, 'normal_query')
            reference_dir = os.path.join(subdir_path, 'reference')
            
            if not os.path.exists(anomaly_query_dir):
                continue
            
            # ==================== 处理 reference ====================
            reference_files = []
            if os.path.exists(reference_dir):
                reference_files = [f for f in os.listdir(reference_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                reference_files.sort()
            
            reference_count = len(reference_files)
            
            # ==================== 处理 normal_query ====================
            normal_files = []
            if os.path.exists(normal_query_dir):
                normal_files = [f for f in os.listdir(normal_query_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                normal_files.sort()
            
            normal_query_count = len(normal_files)
            
            # ==================== 计算需要从 normal_query 移动到 reference 的数量 ====================
            move_count = 0
            if reference_count < self.target_reference_count:
                needed = self.target_reference_count - reference_count
                # 确保 normal_query 至少保留 1 个
                if normal_query_count > needed:
                    # 如果 normal_query 数量足够，补齐到目标数量
                    move_count = needed
                else:
                    # 如果 normal_query 数量不足，尽可能多地移动，但至少保留 1 个
                    # 如果 normal_query_count == 0，则 move_count = 0
                    # 如果 normal_query_count == 1，则 move_count = 0（保留1个）
                    # 如果 normal_query_count > 1，则 move_count = normal_query_count - 1
                    move_count = max(0, normal_query_count - 1)
            
            # ==================== 处理 reference 中的图片（作为train数据用于few-shot）====================
            for ref_file in reference_files:
                img_path = f'{subdir}/reference/{ref_file}'
                
                info_img = dict(
                    img_path=img_path,
                    mask_path='',
                    cls_name=subdir,
                    specie_name='',
                    anomaly=0,
                )
                
                if 'train' not in info:
                    info['train'] = {}
                if subdir not in info['train']:
                    info['train'][subdir] = []
                info['train'][subdir].append(info_img)
            
            # ==================== 处理 normal_query 中的图片 ====================
            # 前 move_count 个文件作为 train（补充到 reference）
            # 剩余的文件作为 test
            for idx, normal_file in enumerate(normal_files):
                img_path = f'{subdir}/normal_query/{normal_file}'
                
                info_img = dict(
                    img_path=img_path,
                    mask_path='',
                    cls_name=subdir,
                    specie_name='',
                    anomaly=0,
                )
                
                if idx < move_count:
                    # 前 move_count 个文件放入 train（作为 reference 的补充）
                    if 'train' not in info:
                        info['train'] = {}
                    if subdir not in info['train']:
                        info['train'][subdir] = []
                    info['train'][subdir].append(info_img)
                    normal_query_train_samples += 1
                else:
                    # 剩余文件放入 test
                    if 'test' not in info:
                        info['test'] = {}
                    if subdir not in info['test']:
                        info['test'][subdir] = []
                    info['test'][subdir].append(info_img)
                    normal_query_test_samples += 1
                    normal_samples += 1
            
            # ==================== 处理 anomaly_query 中的图片 ====================
            if os.path.exists(anomaly_query_dir):
                anomaly_files = [f for f in os.listdir(anomaly_query_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                anomaly_files.sort()
                
                for anomaly_file in anomaly_files:
                    prefix = self.extract_prefix(anomaly_file)
                    if prefix is None:
                        continue
                    
                    # 构建mask文件路径（将result改为mask）
                    mask_filename = anomaly_file.replace('_result_', '_mask_')
                    mask_path = os.path.join(mask_dir, mask_filename) if os.path.exists(os.path.join(mask_dir, mask_filename)) else ''
                    
                    img_path = f'{subdir}/anomaly_query/{anomaly_file}'
                    mask_path_relative = f'{subdir}/mask/{mask_filename}' if mask_path else ''
                    
                    info_img = dict(
                        img_path=img_path,
                        mask_path=mask_path_relative,
                        cls_name=subdir,  # 使用子目录名作为类别名
                        specie_name=prefix,  # 使用前缀作为specie_name，用于后续分组
                        anomaly=1,
                    )
                    
                    # 将所有数据放入test阶段
                    if 'test' not in info:
                        info['test'] = {}
                    if subdir not in info['test']:
                        info['test'][subdir] = []
                    info['test'][subdir].append(info_img)
                    
                    prefix_groups[prefix].append(info_img)
                    anomaly_samples += 1
            
            # 输出每个子文件夹的统计信息
            final_reference_count = reference_count + move_count
            print(f"子文件夹 {subdir}:")
            print(f"  原始 reference: {reference_count}, 原始 normal_query: {normal_query_count}")
            print(f"  从 normal_query 移动到 train: {move_count}")
            print(f"  最终 reference (train): {final_reference_count}, 最终 normal_query (test): {normal_query_count - move_count}")
        
        # 保存meta.json
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        
        print(f'\n总体统计:')
        print(f'  anomaly_samples: {anomaly_samples}')
        print(f'  normal_query_test_samples: {normal_query_test_samples}')
        print(f'  normal_query_train_samples (从normal_query移动到train): {normal_query_train_samples}')
        print(f'  normal_samples (test中的normal): {normal_samples}')
        print(f'Found {len(prefix_groups)} unique prefixes: {sorted(prefix_groups.keys())}')
        for prefix, items in prefix_groups.items():
            print(f'  {prefix}: {len(items)} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare custom dataset v2 for adapter (with reference augmentation)')
    parser.add_argument('--root', type=str, required=True, help='Path to custom dataset root directory')
    parser.add_argument('--target_reference_count', type=int, default=4, 
                       help='Target number of reference samples per subfolder (default: 4)')
    args = parser.parse_args()
    
    runner = CustomDatasetSolverV2(root=args.root, target_reference_count=args.target_reference_count)
    runner.run()

