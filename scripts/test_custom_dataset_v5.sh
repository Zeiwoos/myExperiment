#!/bin/bash

# 使用 test_custom_dataset_v5.py 对自定义数据集进行测试（包含 A/B/C 及困难项统计）
# 使用方法: bash scripts/test_custom_dataset_v5.sh

device=0

# 数据集路径映射
declare -A dataset_map
dataset_map[hxzy_part1]=/data1/GYS/HXZY_traindata_v3
dataset_map[hxzy_part2]=/data1/GYS/HXZY_testdata_v3
dataset_map[hxzy]=/data1/GYS/HXZY_testdata_v3
dataset_map[hxzy_v4_test]=/data1/GYS/HXZY_dataset_v4/testdata

# 模型checkpoint路径映射
declare -A model_map
model_map[visa]=adapter_checkpoints/train_on_visa_3adapters_batch8/epoch_15.pth
model_map[mvtec]=adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth
model_map[hxzy]=adapter_checkpoints/hxzy_full/epoch_15.pth
model_map[hxzy_part1]=adapter_checkpoints/train_on_hxzy_part1_3adapters_batch8/epoch_15.pth
model_map[hxzy_part2]=adapter_checkpoints/train_on_hxzy_part2_3adapters_batch8/epoch_15.pth
model_map[hxzy_part1_v2]=adapter_checkpoints/train_on_hxzy_part1_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train]=adapter_checkpoints/train_on_hxzy_v4_train_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_nosongdong]=adapter_checkpoints/train_on_hxzy_v4_train_nosongdong_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_half]=adapter_checkpoints/train_on_hxzy_v4_train_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_nosongdong_half]=adapter_checkpoints/train_on_hxzy_v4_train_nosongdong_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_controlnet]=adapter_checkpoints/train_on_hxzy_v4_train_visual_only_batch8_v4/visual_only_epoch_10.pth
model_map[hxzy_v4_train_controlnet_nofreeze]=adapter_checkpoints/train_on_hxzy_v4_train_nofreeze_controlnet_batch8/epoch_10.pth
model_map[hxzy_v4_train_controlnet_freeze]=adapter_checkpoints/train_on_hxzy_v4_train_freeze_controlnet_batch8/epoch_10.pth
model_map[hxzy_v4_train_controlnet_fix]=adapter_checkpoints/train_on_hxzy_v4_train_controlnet_fix_controlnet_batch8_freeze/epoch_10.pth
model_map[hxzy_v4_train_plus]=adapter_checkpoints/train_on_hxzy_v4_train_plus_3adapters_batch8_v2/epoch_10.pth

model_map[hxzy_v4_train_4shot]=adapter_checkpoints/train_on_hxzy_v4_train_4shot_3adapters_batch8_v2/epoch_15.pth


# 其他配置
n_ctx=12
vl_reduction=4
pq_mid_dim=128

# 测试配置
train_dataset=hxzy_v4_train_controlnet_nofreeze  # 训练模型使用的数据集
test_dataset=hxzy_v4_test     # 测试数据集
k_shots=1                     # 0: zero-shot, >0: few-shot
test_script=test_custom_dataset_v5.py

# 可视化配置
enable_visualize=false         # true: 开启可视化, false: 关闭可视化
enable_controlnet=true       # true: 测试时启用ControlNet
control_hint_source=image      # gt / image
control_hint_channels=1
control_scales=(1.0)

# F1 阈值配置（可指定多个阈值，用空格分隔）
f1_thresholds=(0.2 0.15 0.1)  # F1 计算使用的阈值列表

# 根据配置获取实际路径
data_root=${dataset_map[$test_dataset]}
checkpoint_path=${model_map[$train_dataset]}

# 校验路径
if [ -z "$data_root" ]; then
    echo "错误: 未找到 test_dataset '${test_dataset}' 的数据路径映射"
    exit 1
fi

if [ -z "$checkpoint_path" ]; then
    echo "错误: 未找到 train_dataset '${train_dataset}' 的模型 checkpoint 映射"
    exit 1
fi

# 将 checkpoint 路径转换为绝对路径（如果当前是相对路径）
if [[ ! "$checkpoint_path" = /* ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    project_root="$(cd "$script_dir/.." && pwd)"
    checkpoint_path="${project_root}/${checkpoint_path}"
fi

echo "数据集路径: ${data_root}"
echo "模型 checkpoint: ${checkpoint_path}"
echo "使用测试脚本: ${test_script}"

# 种子设置
if [ ${k_shots} -eq 0 ]; then
    seeds="10"   # zero-shot: 一个 seed 即可
else
    seeds="20"   # few-shot: 当前场景使用单一 seed=20（可自行扩展）
fi

base_dir=train_on_${train_dataset}_test_on_${test_dataset}_3adapters_batch8_v5
save_dir=./results/${base_dir}

# 准备数据集（生成 meta.json）
echo "准备数据集 meta.json ..."
if [[ "${test_dataset}" == "visa" ]]; then
    python dataset/visa.py --root ${data_root}
elif [[ "${test_dataset}" == "mvtec" ]]; then
    python dataset/mvtec.py --root ${data_root}
else
    # HXZY 或其他自定义数据集
    python dataset/custom_dataset.py --root ${data_root}
fi

# 运行测试
for seed in ${seeds}; do
    echo "Running test with seed=${seed}, k_shots=${k_shots}"

    # 构建可视化参数
    visualize_arg=""
    if [ "${enable_visualize}" = "true" ]; then
        visualize_arg="--visualize"
    fi

    # 构建ControlNet参数
    controlnet_arg=""
    if [ "${enable_controlnet}" = "true" ]; then
        controlnet_arg="--enable_controlnet --control_hint_source ${control_hint_source} --control_hint_channels ${control_hint_channels} --control_scales ${control_scales[@]}"
    fi

    CUDA_VISIBLE_DEVICES=${device} \
    python ${test_script} \
        --dataset "${test_dataset}" \
        --test_data_path "${data_root}" \
        --seed "${seed}" \
        --k_shots "${k_shots}" \
        --checkpoint_path "${checkpoint_path}" \
        --save_path "${save_dir}" \
        --features_list 6 12 18 24 \
        --image_size 518 \
        --batch_size 8 \
        --n_ctx "${n_ctx}" \
        --vl_reduction "${vl_reduction}" \
        --pq_mid_dim "${pq_mid_dim}" \
        --visual_learner \
        --textual_learner \
        --pq_learner \
        --pq_context \
        --f1_thresholds "${f1_thresholds[@]}" \
        ${visualize_arg} \
        ${controlnet_arg}

    wait
done

echo "测试完成！结果保存在: ${save_dir}"