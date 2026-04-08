#!/bin/bash

# ==============================================================================
# 脚本：测试不同 K-shot 值对模型效果的影响，并绘制折线图
# 使用方法：bash scripts/test_kshot_impact.sh
# ==============================================================================

set - euo
pipefail

# 固定工作目录到仓库根目录，避免相对路径出错
SCRIPT_DIR = "$(cd "$(dirname "${BASH_SOURCE[0]}")
" && pwd)"
PROJECT_ROOT = "$(cd "$SCRIPT_DIR /.." && pwd)"
cd
"$PROJECT_ROOT"

# --- [1. 配置参数] ---
DEVICE = 0

# 选择测试脚本（v4 / v5）
TEST_SCRIPT = "test_custom_dataset_v4.py"

# 数据集路径映射（和其它测试脚本保持一致）
declare - A
dataset_map
dataset_map[hxzy_part1] = / data1 / GYS / HXZY_traindata_v3
dataset_map[hxzy_part2] = / data1 / GYS / HXZY_testdata_v3
dataset_map[hxzy] = / data1 / GYS / HXZY_testdata_v3
dataset_map[hxzy_v4_test] = / data1 / GYS / HXZY_dataset_v4 / testdata

# 模型checkpoint路径映射（用于“固定模型，只变k-shot”）
declare - A
model_map
model_map[visa] = adapter_checkpoints / train_on_visa_3adapters_batch8 / epoch_15.pth
model_map[mvtec] = adapter_checkpoints / train_on_mvtec_3adapters_batch8 / epoch_15.pth
model_map[hxzy] = adapter_checkpoints / hxzy_full / epoch_15.pth
model_map[hxzy_part1] = adapter_checkpoints / train_on_hxzy_part1_3adapters_batch8 / epoch_15.pth
model_map[hxzy_part2] = adapter_checkpoints / train_on_hxzy_part2_3adapters_batch8 / epoch_15.pth
model_map[hxzy_part1_v2] = adapter_checkpoints / train_on_hxzy_part1_3adapters_batch8_v2 / epoch_15.pth
model_map[hxzy_v4_train] = adapter_checkpoints / train_on_hxzy_v4_train_3adapters_batch8_v2 / epoch_15.pth
model_map[
    hxzy_v4_train_nosongdong] = adapter_checkpoints / train_on_hxzy_v4_train_nosongdong_3adapters_batch8_v2 / epoch_15.pth
model_map[hxzy_v4_train_half] = adapter_checkpoints / train_on_hxzy_v4_train_half_3adapters_batch8_v2 / epoch_15.pth
model_map[
    hxzy_v4_train_nosongdong_half] = adapter_checkpoints / train_on_hxzy_v4_train_nosongdong_half_3adapters_batch8_v2 / epoch_15.pth
model_map[
    hxzy_v4_controlnet] = adapter_checkpoints / train_on_hxzy_v4_train_visual_only_batch8_v4 / visual_only_epoch_10.pth
model_map[
    hxzy_v4_train_controlnet_nofreeze] = adapter_checkpoints / train_on_hxzy_v4_train_nofreeze_controlnet_batch8 / epoch_10.pth
model_map[
    hxzy_v4_train_controlnet_freeze] = adapter_checkpoints / train_on_hxzy_v4_train_freeze_controlnet_batch8 / epoch_10.pth
model_map[
    hxzy_v4_train_controlnet_fix] = adapter_checkpoints / train_on_hxzy_v4_train_controlnet_fix_controlnet_batch8_freeze / epoch_10.pth
model_map[hxzy_v4_train_plus] = adapter_checkpoints / train_on_hxzy_v4_train_plus_3adapters_batch8_v2 / epoch_10.pth
model_map[hxzy_v4_train_4shot] = adapter_checkpoints / train_on_hxzy_v4_train_4shot_3adapters_batch8_v2 / epoch_15.pth

# 你要测试的数据集 key（决定 test_data_path / dataset 名称）
TEST_DATASET_KEY = "hxzy_part1"  # 【请修改】用于生成 meta.json 并评估

# 你要固定使用的训练集 key（决定 checkpoint_path）
TRAIN_DATASET_KEY = "hxzy_part1"  # 【请修改】固定模型，只变 k_shot

# 测试的 K-shot 值列表（当前脚本只针对 >0 的 few-shot）
K_SHOTS = (1 2 4 8)

# 是否开启可视化 (1 开启，0 关闭)
ENABLE_VISUALIZATION = 0

# 日志与结果保存目录
OUTPUT_DIR = "results/kshot_experiment"
mkdir - p
"$OUTPUT_DIR"

CSV_FILE = "${OUTPUT_DIR}/kshot_metrics.csv"
echo
"K_Shot,Image_AUROC,Pixel_AUROC,PRO_AUROC30" > "$CSV_FILE"

# 一些推理默认超参数（保持与其它脚本一致，避免不必要差异）
n_ctx = 12
vl_reduction = 4
pq_mid_dim = 128

# F1 阈值配置
f1_thresholds = (0.2 0.15 0.1)

echo
"=========================================================="
echo
"开始 K-shot 影响测试..."
echo
"脚本: $TEST_SCRIPT"
echo
"固定模型 key: $TRAIN_DATASET_KEY"
echo
"测试数据 key: $TEST_DATASET_KEY"
echo
"测试的 K-shot 列表: ${K_SHOTS[*]}"
echo
"可视化选项: $(if [ $ENABLE_VISUALIZATION -eq 1 ]; then echo '开启'; else echo '关闭'; fi)"
echo
"结果将保存在: $OUTPUT_DIR"
echo
"=========================================================="

# --- [2. 解析映射得到必需路径] ---
DATA_ROOT = "${dataset_map[$TEST_DATASET_KEY]:-}"
CHECKPOINT_PATH = "${model_map[$TRAIN_DATASET_KEY]:-}"

if [-z "$DATA_ROOT"]; then
echo
"错误: 未找到 test_dataset key '${TEST_DATASET_KEY}' 的数据路径映射"
exit
1
fi
if [-z "$CHECKPOINT_PATH"]; then
echo
"错误: 未找到 train_dataset key '${TRAIN_DATASET_KEY}' 的模型 checkpoint 映射"
exit
1
fi

# 将 checkpoint 路径转换为绝对路径（如果当前是相对路径）
if [[ ! "$CHECKPOINT_PATH" = / *]]; then
CHECKPOINT_PATH = "${PROJECT_ROOT}/${CHECKPOINT_PATH}"
fi

echo
"数据集路径: $DATA_ROOT"
echo
"模型 checkpoint: $CHECKPOINT_PATH"

# --- [3. 预处理数据集（生成 meta.json）] ---
echo
"准备数据集 meta.json ..."
if [["${TEST_DATASET_KEY}" == "visa"]]; then
python
dataset / visa.py - -root
"${DATA_ROOT}"
elif [["${TEST_DATASET_KEY}" == "mvtec"]];
then
python
dataset / mvtec.py - -root
"${DATA_ROOT}"
else
# 自定义 HXZY 数据集
python
dataset / custom_dataset.py - -root
"${DATA_ROOT}"
fi

# --- [4. 循环测试不同 K-shot] ---
for K in "${K_SHOTS[@]}"; do
echo
"=========================================================="
echo
"正在测试 K-shot = $K ..."

# v4/v5 要求：可视化用 --visualize，且必须开启 adapter
VIS_ARG = ""
if ["$ENABLE_VISUALIZATION" - eq 1]; then
VIS_ARG = "--visualize"
fi

# 沿用其它测试脚本的 seed 规则：zero-shot 用 10，few-shot 用 20
if ["$K" - eq 0]; then
SEED = 10
else
SEED = 20
fi

RUN_DIR = "${OUTPUT_DIR}/run_k${K}"
mkdir - p
"$RUN_DIR"
RUN_LOG = "${RUN_DIR}/stdout.log"

# 注意：参数名必须和 test_custom_dataset_v4.py / v5.py 的 argparse 一致
CUDA_VISIBLE_DEVICES = "$DEVICE"
python
"$TEST_SCRIPT" \
- -dataset
"$TEST_DATASET_KEY" \
- -test_data_path
"$DATA_ROOT" \
- -seed
"$SEED" \
- -k_shots
"$K" \
- -checkpoint_path
"$CHECKPOINT_PATH" \
- -save_path
"$RUN_DIR" \
- -features_list
6
12
18
24 \
- -image_size
518 \
- -batch_size
8 \
- -n_ctx
"$n_ctx" \
- -vl_reduction
"$vl_reduction" \
- -pq_mid_dim
"$pq_mid_dim" \
- -visual_learner \
- -textual_learner \
- -pq_learner \
- -pq_context \
- -f1_thresholds
"${f1_thresholds[@]}" \
    ${VIS_ARG} \
    2 > & 1 | tee
"$RUN_LOG"

# --- [4. 自动解析指标] ---
# 匹配 v4/v5 的日志输出格式：
# - Overall Image-level AUC: xx.xx%
# - Overall Pixel-level AUC (Seg AUC): xx.xx%
# - AU-PRO@0.30: xx.xx%
IMG_AUROC = "$(grep -E "
Overall
Image - level
AUC: " "$RUN_LOG
" | tail -n 1 | awk '{print $NF}' | tr -d '%')"
PIX_AUROC = "$(grep -E "
Overall
Pixel - level
AUC \\(Seg AUC\\): " "$RUN_LOG
" | tail -n 1 | awk '{print $NF}' | tr -d '%')"
PRO = "$(grep -E "
AU - PRO @ 0.30: " "$RUN_LOG
" | tail -n 1 | awk '{print $NF}' | tr -d '%')"

IMG_AUROC = "${IMG_AUROC:-0}"
PIX_AUROC = "${PIX_AUROC:-0}"
PRO = "${PRO:-0}"

echo
"$K,$IMG_AUROC,$PIX_AUROC,$PRO" >> "$CSV_FILE"
echo
"-> 提取到的指标: K=$K | Image AUROC: $IMG_AUROC | Pixel AUROC: $PIX_AUROC | PRO@0.30: $PRO"
done

# --- [4. 绘制折线图 (使用内嵌 Python 脚本)] ---
echo
"=========================================================="
echo
"所有 K-shot 测试完成！开始读取 $CSV_FILE 绘制指标折线图..."

python - c
"
import pandas as pd
import matplotlib.pyplot as plt

csv_file = '$CSV_FILE'
save_path = '${OUTPUT_DIR}/kshot_impact_curve.png'

try:
    df = pd.read_csv(csv_file)
    # 确保数值列是数字
    for col in ['Image_AUROC', 'Pixel_AUROC', 'PRO_AUROC30']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    plt.figure(figsize=(10, 6))

    # 绘制三条折线
    plt.plot(df['K_Shot'], df['Image_AUROC'], marker='o', linestyle='-', label='Image AUROC', linewidth=2,
             color='#1f77b4')
    plt.plot(df['K_Shot'], df['Pixel_AUROC'], marker='s', linestyle='--', label='Pixel AUROC', linewidth=2,
             color='#ff7f0e')
    plt.plot(df['K_Shot'], df['PRO_AUROC30'], marker='^', linestyle='-.', label='PRO@0.30', linewidth=2,
             color='#2ca02c')

    plt.title('Impact of K-shot on Anomaly Detection Performance', fontsize=16, fontweight='bold')
    plt.xlabel('K-shot (Number of Reference Normal Samples)', fontsize=14)
    plt.ylabel('Metric Score', fontsize=14)
    plt.xticks(df['K_Shot'])

    # 增加网格线和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'-> 折线图已成功生成并保存至: {save_path}')
except Exception as e:
    print(f'-> 绘图失败，请检查 CSV 文件是否正常写入，或确保安装了 pandas 和 matplotlib。错误信息: {e}')
"

echo
"=========================================================="
echo
"全部任务结束！"