#!/bin/bash

# 自动运行不同超参数配置下的 test_custom_dataset_v4.sh
# 支持显存不足时自动重试
# 使用方法: bash scripts/auto.sh

# ==================== 配置区域 ====================

# 设备配置
device=0

# 重试配置
retry_wait_seconds=300  # 显存不足后等待时间（秒），默认5分钟
max_retries=3           # 每个配置的最大重试次数

# 测试配置列表（每个配置是一个关联数组）
# 格式: train_dataset:test_dataset:enable_visualize:description

declare -a test_configs=(
    # 格式: "train_dataset:test_dataset:enable_visualize:k_shots:description"
    # "hxzy_v4_train_nosongdong_half_plus:hxzy_v4_test:false:1:nosongdong_half_plus"

    # "hxzy_v4_train_1shot_half:hxzy_v4_test:false:1:1shot_half"
    "hxzy_v4_train_1shot_half:hxzy_v4_test:false:4:1shot_half_4"

    # "hxzy_v4_train_1shot_nosongdong_half:hxzy_v4_test:false:1:1shot_nosongdong_half"
    # "hxzy_v4_train_1shot_nosongdong_half:hxzy_v4_test:false:4:1shot_nosongdong_half_4"

    # "hxzy_v4_train_1shot_nosongdong_half_plus:hxzy_v4_test:false:1:1shot_nosongdong_half_plus"
    # "hxzy_v4_train_1shot_nosongdong_half_plus:hxzy_v4_test:false:4:1shot_nosongdong_half_plus_4"

    # "hxzy_v4_train_4shot_nosongdong_half:hxzy_v4_test:false:1:4shot_nosongdong_half"
    # "hxzy_v4_train_4shot_nosongdong_half:hxzy_v4_test:false:4:4shot_nosongdong_half_4"

    # "hxzy_v4_train_4shot_nosongdong_half_plus:hxzy_v4_test:false:1:4shot_nosongdong_half_plus"
    # "hxzy_v4_train_4shot_nosongdong_half_plus:hxzy_v4_test:false:4:4shot_nosongdong_half_plus_4"
)

# 其他固定配置
test_dataset_default=hxzy_v4_test
k_shots_default=1  # 默认值（当配置中未指定时使用）
n_ctx=12
vl_reduction=4
pq_mid_dim=128
f1_thresholds=(0.2 0.15 0.1)

# ==================== 辅助函数 ====================

# 检查是否为显存不足错误
is_oom_error() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        # 检查常见的OOM错误信息
        if grep -qi "out of memory\|CUDA out of memory\|RuntimeError.*CUDA\|OOM" "$log_file"; then
            return 0  # 是OOM错误
        fi
    fi
    return 1  # 不是OOM错误
}

# 检查测试是否成功完成
is_test_successful() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        # 检查是否包含成功完成的标志
        if grep -qi "测试完成\|测试完成！\|Overall Results Summary" "$log_file"; then
            return 0  # 成功
        fi
    fi
    return 1  # 未成功
}

# 运行单个测试配置
run_single_test() {
    local train_dataset="$1"
    local test_dataset="$2"
    local enable_visualize="$3"
    local k_shots="$4"
    local description="$5"
    local config_index="$6"
    local total_configs="$7"
    
    echo ""
    echo "=========================================="
    echo "[$config_index/$total_configs] 运行配置: $description"
    echo "  train_dataset: $train_dataset"
    echo "  test_dataset: $test_dataset"
    echo "  enable_visualize: $enable_visualize"
    echo "  k_shots: $k_shots"
    echo "=========================================="
    
    # 创建临时脚本文件
    local temp_script=$(mktemp)
    local log_file="auto_test_${description}_$(date +%Y%m%d_%H%M%S).log"
    
    # 获取项目根目录（在生成脚本时确定，而不是在临时脚本中计算）
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root="$(cd "$script_dir/.." && pwd)"
    
    # 生成测试脚本内容
    cat > "$temp_script" <<EOF
#!/bin/bash
# 自动生成的测试脚本
# 配置: $description

device=${device}

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
model_map[hxzy_v4_controlnet]=adapter_checkpoints/train_on_hxzy_v4_train_controlnet_batch8/epoch_10.pth
model_map[hxzy_v4_train_4shot]=adapter_checkpoints/train_on_hxzy_v4_train_4shot_3adapters_batch8_v2/epoch_15.pth

model_map[hxzy_v4_train_half]=adapter_checkpoints/train_on_hxzy_v4_train_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_nosongdong_half]=adapter_checkpoints/train_on_hxzy_v4_train_nosongdong_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_nosongdong_half_plus]=adapter_checkpoints/train_on_hxzy_v4_train_nosongdong_half_plus_3adapters_batch8_v2/epoch_15.pth

model_map[hxzy_v4_train_1shot_half]=adapter_checkpoints/train_on_hxzy_v4_train_1shot_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_1shot_nosongdong_half]=adapter_checkpoints/train_on_hxzy_v4_train_1shot_nosongdong_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_1shot_nosongdong_half_plus]=adapter_checkpoints/train_on_hxzy_v4_train_1shot_nosongdong_half_plus_3adapters_batch8_v2/epoch_15.pth

model_map[hxzy_v4_train_4shot_nosongdong_half]=adapter_checkpoints/train_on_hxzy_v4_train_4shot_nosongdong_half_3adapters_batch8_v2/epoch_15.pth
model_map[hxzy_v4_train_4shot_nosongdong_half_plus]=adapter_checkpoints/train_on_hxzy_v4_train_4shot_nosongdong_half_plus_3adapters_batch8_v2/epoch_15.pth

# 其他配置
n_ctx=${n_ctx}
vl_reduction=${vl_reduction}
pq_mid_dim=${pq_mid_dim}

# 测试配置
train_dataset=${train_dataset}
test_dataset=${test_dataset}
k_shots=${k_shots}
test_script=test_custom_dataset_v4.py

# 可视化配置
enable_visualize=${enable_visualize}

# F1 阈值配置
f1_thresholds=(${f1_thresholds[@]})

# 根据配置获取实际路径
data_root=\${dataset_map[\$test_dataset]}
checkpoint_path=\${model_map[\$train_dataset]}

# 校验路径
if [ -z "\$data_root" ]; then
    echo "错误: 未找到 test_dataset '\${test_dataset}' 的数据路径映射"
    exit 1
fi

if [ -z "\$checkpoint_path" ]; then
    echo "错误: 未找到 train_dataset '\${train_dataset}' 的模型 checkpoint 映射"
    exit 1
fi

# 将 checkpoint 路径转换为绝对路径
if [[ ! "\$checkpoint_path" = /* ]]; then
    # 使用预先计算的项目根目录
    project_root="${project_root}"
    # 确保路径拼接正确（移除可能的双斜杠）
    checkpoint_path="\${project_root}/\${checkpoint_path}"
    checkpoint_path=\$(echo "\$checkpoint_path" | sed 's|//|/|g')
fi

echo "数据集路径: \${data_root}"
echo "模型 checkpoint: \${checkpoint_path}"
echo "使用测试脚本: \${test_script}"

# 种子设置
if [ \${k_shots} -eq 0 ]; then
    seeds="10"
else
    seeds="20"
fi

base_dir=train_on_\${train_dataset}_test_on_\${test_dataset}_3adapters_batch8_v4
save_dir=./results/\${base_dir}

# 准备数据集（生成 meta.json）
echo "准备数据集 meta.json ..."
if [[ "\${test_dataset}" == "visa" ]]; then
    python dataset/visa.py --root \${data_root}
elif [[ "\${test_dataset}" == "mvtec" ]]; then
    python dataset/mvtec.py --root \${data_root}
else
    # 根据 k_shots 选择不同的数据集处理脚本
    if [ \${k_shots} -eq 4 ]; then
        python dataset/custom_dataset_v2.py --root \${data_root}
    else
        python dataset/custom_dataset.py --root \${data_root}
    fi
fi

# 运行测试
for seed in \${seeds}; do
    echo "Running test with seed=\${seed}, k_shots=\${k_shots}"

    # 构建可视化参数
    visualize_arg=""
    if [ "\${enable_visualize}" = "true" ]; then
        visualize_arg="--visualize"
    fi

    CUDA_VISIBLE_DEVICES=\${device} \\
    python \${test_script} \\
        --dataset "\${test_dataset}" \\
        --test_data_path "\${data_root}" \\
        --seed "\${seed}" \\
        --k_shots "\${k_shots}" \\
        --checkpoint_path "\${checkpoint_path}" \\
        --save_path "\${save_dir}" \\
        --features_list 6 12 18 24 \\
        --image_size 518 \\
        --batch_size 8 \\
        --n_ctx "\${n_ctx}" \\
        --vl_reduction "\${vl_reduction}" \\
        --pq_mid_dim "\${pq_mid_dim}" \\
        --visual_learner \\
        --textual_learner \\
        --pq_learner \\
        --pq_context \\
        --f1_thresholds "\${f1_thresholds[@]}" \\
        \${visualize_arg}

    wait
done

echo "测试完成！结果保存在: \${save_dir}"
EOF

    chmod +x "$temp_script"
    
    # 运行测试，带重试机制
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $max_retries ]; do
        echo ""
        if [ $retry_count -eq 0 ]; then
            echo "开始运行测试..."
        else
            echo "第 $retry_count 次重试（等待 ${retry_wait_seconds} 秒后）..."
            sleep $retry_wait_seconds
            echo "继续运行..."
        fi
        
        # 运行测试并捕获输出
        bash "$temp_script" 2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
        
        # 检查结果
        if [ $exit_code -eq 0 ] && is_test_successful "$log_file"; then
            echo ""
            echo "✓ 配置 '$description' 运行成功！"
            success=true
            break
        elif is_oom_error "$log_file"; then
            retry_count=$((retry_count + 1))
            echo ""
            echo "✗ 检测到显存不足错误 (OOM)"
            if [ $retry_count -lt $max_retries ]; then
                echo "将在 ${retry_wait_seconds} 秒后重试 (第 $retry_count/$max_retries 次)..."
                # 清理GPU缓存
                python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            else
                echo "已达到最大重试次数 ($max_retries)，跳过此配置"
            fi
        else
            echo ""
            echo "✗ 配置 '$description' 运行失败（退出码: $exit_code）"
            echo "  查看日志: $log_file"
            success=false
            break
        fi
    done
    
    # 清理临时脚本
    rm -f "$temp_script"
    
    # 记录结果
    if [ "$success" = true ]; then
        echo "✓ [$config_index/$total_configs] $description - 成功" >> auto_test_summary.log
    else
        echo "✗ [$config_index/$total_configs] $description - 失败" >> auto_test_summary.log
    fi
    
    return $([ "$success" = true ] && echo 0 || echo 1)
}

# ==================== 主程序 ====================

# 获取脚本所在目录
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"

# 创建日志文件
summary_log="auto_test_summary_$(date +%Y%m%d_%H%M%S).log"
echo "自动测试开始时间: $(date)" > "$summary_log"
echo "==========================================" >> "$summary_log"

# 统计信息
total_configs=${#test_configs[@]}
successful_configs=0
failed_configs=0

echo "=========================================="
echo "自动测试脚本启动"
echo "总配置数: $total_configs"
echo "重试等待时间: ${retry_wait_seconds} 秒"
echo "最大重试次数: $max_retries"
echo "=========================================="
echo ""

# 遍历所有配置
config_index=0
for config in "${test_configs[@]}"; do
    config_index=$((config_index + 1))
    
    # 解析配置
    IFS=':' read -r train_dataset test_dataset enable_visualize k_shots description <<< "$config"
    
    # 如果 test_dataset 为空，使用默认值
    if [ -z "$test_dataset" ]; then
        test_dataset="$test_dataset_default"
    fi
    
    # 如果 k_shots 为空，使用默认值
    if [ -z "$k_shots" ]; then
        k_shots="$k_shots_default"
    fi
    
    # 运行测试
    if run_single_test "$train_dataset" "$test_dataset" "$enable_visualize" "$k_shots" "$description" "$config_index" "$total_configs"; then
        successful_configs=$((successful_configs + 1))
    else
        failed_configs=$((failed_configs + 1))
    fi
    
    # 配置之间的间隔（可选）
    if [ $config_index -lt $total_configs ]; then
        echo ""
        echo "等待 10 秒后继续下一个配置..."
        sleep 10
    fi
done

# 输出总结
echo ""
echo "=========================================="
echo "自动测试完成"
echo "总配置数: $total_configs"
echo "成功: $successful_configs"
echo "失败: $failed_configs"
echo "完成时间: $(date)"
echo "=========================================="

# 追加到总结日志
echo "" >> "$summary_log"
echo "==========================================" >> "$summary_log"
echo "自动测试完成时间: $(date)" >> "$summary_log"
echo "总配置数: $total_configs" >> "$summary_log"
echo "成功: $successful_configs" >> "$summary_log"
echo "失败: $failed_configs" >> "$summary_log"
echo "==========================================" >> "$summary_log"

echo ""
echo "详细日志已保存到: $summary_log"

