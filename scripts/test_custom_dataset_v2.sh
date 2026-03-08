#!/bin/bash

# 测试自定义数据集的脚本
# 使用方法: bash scripts/test_custom_dataset_v2.sh

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

# 其他配置
n_ctx=12
vl_reduction=4
pq_mid_dim=128

 
# 测试配置
train_dataset=hxzy_v4_train  # 训练的模型使用的数据集
test_dataset=hxzy_v4_test
k_shots=1  # 0表示zero-shot（不使用reference样本，只使用预训练模型），>0表示few-shot（从reference目录采样k个正常样本构建prompt memory）
test_script=test_custom_dataset_v2.py

# 根据配置获取实际路径
data_root=${dataset_map[$test_dataset]}
checkpoint_path=${model_map[$train_dataset]}

# 验证路径是否存在
if [ -z "$data_root" ]; then
    echo "错误: 未找到test_dataset '${test_dataset}' 的映射配置"
    exit 1
fi

if [ -z "$checkpoint_path" ]; then
    echo "错误: 未找到train_dataset '${train_dataset}' 的模型checkpoint配置"
    exit 1
fi

# 将checkpoint路径转换为绝对路径（如果它是相对路径）
if [[ ! "$checkpoint_path" = /* ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    project_root="$(cd "$script_dir/.." && pwd)"
    checkpoint_path="${project_root}/${checkpoint_path}"
fi




echo "数据集路径: ${data_root}"
echo "模型checkpoint: ${checkpoint_path}"
echo "使用测试脚本: ${test_script}"

# 种子设置
# seed的作用：
# 1. 设置全局随机种子，确保结果可重复
# 2. 当k_shots>0时，seed控制从reference目录中采样哪些样本构建few-shot prompt memory
#    - 不同seed会采样不同的reference样本，从而影响测试结果
#    - 使用多个seed（10, 20, 30）可以评估模型在不同few-shot样本组合下的稳定性
# 3. 当k_shots=0时，不使用reference样本，只需一个seed即可
if [ ${k_shots} -eq 0 ]; then
    seeds="10"  # zero-shot模式下只需一个seed
else
    seeds="20"  # few-shot模式下使用多个seed评估稳定性（但目前reference总共只有一张，无需seeds）
fi

base_dir=train_on_${train_dataset}_test_on_${test_dataset}_3adapters_batch8_v2

save_dir=./results/${base_dir}

# 首先准备数据集（生成meta.json）
echo "准备数据集..."
# 根据测试数据集名称自动选择数据准备脚本
if [[ "${test_dataset}" == "visa" ]]; then
    python dataset/visa.py --root ${data_root}
elif [[ "${test_dataset}" == "mvtec" ]]; then
    python dataset/mvtec.py --root ${data_root}
else
    # HXZY或其他自定义数据集使用custom_dataset.py
    python dataset/custom_dataset.py --root ${data_root}
fi

# 运行测试
for seed in ${seeds}; do
    echo "Running test with seed=${seed}, k_shots=${k_shots}"

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
        --visualize

    wait
done

echo "测试完成！结果保存在: ${save_dir}"

