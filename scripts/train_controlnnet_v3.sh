#!/usr/bin/env bash
set -euo pipefail

# ControlNet v3 训练脚本
# 用法：
#   bash scripts/train_controlnet_v3.sh
#   DATA_ROOT=/path/to/data DATASET_NAME=my_dataset bash scripts/train_controlnet_v3.sh
#   CHECKPOINT_PATH=adapter_checkpoints/xxx.pth ENABLE_PQ_LEARNER=true bash scripts/train_controlnet_v3.sh

# ==================== 可覆盖配置（通过环境变量） ====================
DEVICE="${DEVICE:-1}"
DATA_ROOT="${DATA_ROOT:-/data1/GYS/HXZY_dataset_v4/train_visual_adapter_dataset}"
DATASET_NAME="${DATASET_NAME:-hxzy_part1}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-ViT-L/14@336px}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/root123/GYS/fine_tuning/adapter_checkpoints/train_on_hxzy_v4_train_nosongdong_half_3adapters_batch8_v2/epoch_15.pth}"

EPOCHS="${EPOCHS:-15}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SEED="${SEED:-10}"
K_SHOTS="${K_SHOTS:-1}"

N_CTX="${N_CTX:-12}"
VL_REDUCTION="${VL_REDUCTION:-4}"
PQ_MID_DIM="${PQ_MID_DIM:-128}"
LAMBDA_KD="${LAMBDA_KD:-0.5}"     #调整以增加新数据的影响 base:1.0

ENABLE_VISUAL_LEARNER="${ENABLE_VISUAL_LEARNER:-true}"
ENABLE_TEXTUAL_LEARNER="${ENABLE_TEXTUAL_LEARNER:-false}"
ENABLE_PQ_LEARNER="${ENABLE_PQ_LEARNER:-false}"
ENABLE_PQ_CONTEXT="${ENABLE_PQ_CONTEXT:-false}"

FEATURES_LIST="${FEATURES_LIST:-6 12 18 24}"

# ==================== 路径准备 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_DIR="train_on_${DATASET_NAME}_v3_batch${BATCH_SIZE}_k${K_SHOTS}_kd${LAMBDA_KD}"
SAVE_DIR="${SAVE_DIR:-./adapter_checkpoints/${BASE_DIR}}"
mkdir -p "${SAVE_DIR}"

# ==================== 配置校验 ====================
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT 不存在: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH 不存在: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ "${ENABLE_VISUAL_LEARNER}" != "true" && "${ENABLE_TEXTUAL_LEARNER}" != "true" && "${ENABLE_PQ_LEARNER}" != "true" ]]; then
  echo "[ERROR] 至少需要启用一个 learner（visual/textual/pq）" >&2
  exit 1
fi

# ==================== 组装参数 ====================
ARGS=(
  --train_data_path "${DATA_ROOT}"
  --save_path "${SAVE_DIR}"
  --dataset "${DATASET_NAME}"
  --pretrained_model "${PRETRAINED_MODEL}"
  --n_ctx "${N_CTX}"
  --features_list ${FEATURES_LIST}
  --epoch "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --image_size "${IMAGE_SIZE}"
  --seed "${SEED}"
  --k_shots "${K_SHOTS}"
  --vl_reduction "${VL_REDUCTION}"
  --pq_mid_dim "${PQ_MID_DIM}"
  --lambda_kd "${LAMBDA_KD}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ "${ENABLE_VISUAL_LEARNER}" == "true" ]]; then
  ARGS+=(--visual_learner)
fi
if [[ "${ENABLE_TEXTUAL_LEARNER}" == "true" ]]; then
  ARGS+=(--textual_learner)
fi
if [[ "${ENABLE_PQ_LEARNER}" == "true" ]]; then
  ARGS+=(--pq_learner)
fi
if [[ "${ENABLE_PQ_CONTEXT}" == "true" ]]; then
  ARGS+=(--pq_context)
fi

# ==================== 执行 ====================
echo "[INFO] Project Root: ${PROJECT_ROOT}"
echo "[INFO] Save Dir: ${SAVE_DIR}"
echo "[INFO] Dataset: ${DATASET_NAME} (${DATA_ROOT})"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "[INFO] Learners: visual=${ENABLE_VISUAL_LEARNER}, textual=${ENABLE_TEXTUAL_LEARNER}, pq=${ENABLE_PQ_LEARNER}"

echo "[INFO] Running: CUDA_VISIBLE_DEVICES=${DEVICE} python train_ControlNet_v3.py ..."
CUDA_VISIBLE_DEVICES="${DEVICE}" python train_ControlNet_v3.py "${ARGS[@]}"

echo "训练完成！模型保存在: ${SAVE_DIR}"
