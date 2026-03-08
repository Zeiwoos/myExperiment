"""Custom dataset solver for anomaly detection."""

import argparse
import json
import os
import re
from collections import defaultdict


class CustomDatasetSolver(object):
    def __init__(self, root='data/custom_dataset'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']

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
            
            # 处理anomaly_query中的图片
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
            
            # 处理normal_query中的图片
            if os.path.exists(normal_query_dir):
                normal_files = [f for f in os.listdir(normal_query_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                normal_files.sort()
                
                for normal_file in normal_files:
                    img_path = f'{subdir}/normal_query/{normal_file}'
                    
                    info_img = dict(
                        img_path=img_path,
                        mask_path='',
                        cls_name=subdir,
                        specie_name='',
                        anomaly=0,
                    )
                    
                    if 'test' not in info:
                        info['test'] = {}
                    if subdir not in info['test']:
                        info['test'][subdir] = []
                    info['test'][subdir].append(info_img)
                    normal_samples += 1
            
            # 处理reference中的图片（作为train数据用于few-shot）
            if os.path.exists(reference_dir):
                reference_files = [f for f in os.listdir(reference_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                reference_files.sort()
                
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
        
        # 保存meta.json
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        
        print(f'normal_samples: {normal_samples}, anomaly_samples: {anomaly_samples}')
        print(f'Found {len(prefix_groups)} unique prefixes: {sorted(prefix_groups.keys())}')
        for prefix, items in prefix_groups.items():
            print(f'  {prefix}: {len(items)} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare custom dataset for adapter')
    parser.add_argument('--root', type=str, required=True, help='Path to custom dataset root directory')
    args = parser.parse_args()
    
    runner = CustomDatasetSolver(root=args.root)
    runner.run()

