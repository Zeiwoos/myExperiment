#!/bin/bash

# 训练自定义数据集的脚本
# 使用方法: bash scripts/train_custom_dataset_copy.sh

device=0
# /data1/GYS/HXZY_dataset_v4/traindata
# 数据集配置
data_root=/data1/GYS/HXZY_dataset_v4/train_visual_adapter_dataset  # <请设置您的数据集根目录路径>
n_ctx=12
vl_reduction=4
pq_mid_dim=128

# 训练配置
dataset_name=hxzy_v4_train_4shot_nosongdong_half_plus
epochs=15
learning_rate=0.001
batch_size=8
image_size=518
k_shots=1  # few-shot数量
# checkpoint_path: 可选，用于从已有checkpoint继续训练
# 如果为空字符串，则从头开始训练；如果填写路径，则从该checkpoint加载权重
checkpoint_path="adapter_checkpoints/train_on_hxzy_v4_train_4shot_nosongdong_half_3adapters_batch8_v2/epoch_15.pth"  # 示例: adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth

# 创建保存目录
base_dir=train_on_${dataset_name}_3adapters_batch${batch_size}_v2
save_dir=./adapter_checkpoints/${base_dir}
mkdir -p ${save_dir}

首先准备数据集（生成meta.json）
echo "准备数据集..."
根据数据集名称自动选择数据准备脚本
if [[ "${dataset_name}" == "visa" ]]; then
    python dataset/visa.py --root ${data_root}
elif [[ "${dataset_name}" == "mvtec" ]]; then
    python dataset/mvtec.py --root ${data_root}
else
    # HXZY或其他自定义数据集使用custom_dataset.py
    python dataset/custom_dataset.py --root ${data_root}
fi

# 运行训练
echo "开始训练..."
CUDA_VISIBLE_DEVICES=${device} python train_custom_dataset.py \
    --train_data_path ${data_root} \
    --save_path ${save_dir} \
    --dataset ${dataset_name} \
    --pretrained_model ViT-L/14@336px \
    --n_ctx ${n_ctx} \
    --features_list 6 12 18 24 \
    --epoch ${epochs} \
    --learning_rate ${learning_rate} \
    --batch_size ${batch_size} \
    --image_size ${image_size} \
    --seed 10 \
    --k_shots ${k_shots} \
    --visual_learner \
    --textual_learner \
    --pq_learner \
    --vl_reduction ${vl_reduction} \
    --pq_mid_dim ${pq_mid_dim} \
    --pq_context \
    --print_freq 1 \
    --save_freq 1

echo "训练完成！模型保存在: ${save_dir}"







