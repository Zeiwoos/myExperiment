#!/usr/bin/env bash
set -euo pipefail

# ControlNet 训练脚本
# 用法：
#   bash scripts/train_controlnet_v2.sh
#   DATA_ROOT=/path/to/data DATASET_NAME=hxzy_v4_train bash scripts/train_controlnet.sh
#   CONTROLNET_ONLY=true CHECKPOINT_PATH=adapter_checkpoints/xxx.pth bash scripts/train_controlnet.sh

# ==================== 可覆盖配置（通过环境变量） ====================
DEVICE="${DEVICE:-1}"
DATA_ROOT="${DATA_ROOT:-/data1/GYS/HXZY_dataset_v4/train_visual_adapter_dataset}"
DATASET_NAME="${DATASET_NAME:-hxzy_v4_train_controlnet_beta}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-ViT-L/14@336px}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-adapter_checkpoints/train_on_hxzy_v4_train_3adapters_batch8_v2/epoch_15.pth}"

EPOCHS="${EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SEED="${SEED:-10}"
K_SHOTS="${K_SHOTS:-1}"

N_CTX="${N_CTX:-12}"
VL_REDUCTION="${VL_REDUCTION:-4}"
PQ_MID_DIM="${PQ_MID_DIM:-128}"

ENABLE_VISUAL_LEARNER="${ENABLE_VISUAL_LEARNER:-true}"
ENABLE_TEXTUAL_LEARNER="${ENABLE_TEXTUAL_LEARNER:-false}"
ENABLE_PQ_LEARNER="${ENABLE_PQ_LEARNER:-false}"
ENABLE_PQ_CONTEXT="${ENABLE_PQ_CONTEXT:-false}"

ENABLE_CONTROLNET="${ENABLE_CONTROLNET:-true}"
CONTROLNET_ONLY="${CONTROLNET_ONLY:-true}"
CONTROL_HINT_SOURCE="${CONTROL_HINT_SOURCE:-gt}"      # gt / image
CONTROL_HINT_CHANNELS="${CONTROL_HINT_CHANNELS:-1}"
CONTROL_SCALES="${CONTROL_SCALES:-1.0}"              # 例如: "1.0 1.0 0.5 0.5"

FEATURES_LIST="${FEATURES_LIST:-6 12 18 24}"

# ==================== 路径准备 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_DIR="train_on_${DATASET_NAME}_controlnet_batch${BATCH_SIZE}"
if [[ "${CONTROLNET_ONLY}" == "true" ]]; then
  BASE_DIR+="_freeze"
fi
SAVE_DIR="${SAVE_DIR:-./adapter_checkpoints/${BASE_DIR}}"
mkdir -p "${SAVE_DIR}"

# ==================== 配置校验（对齐 train_ControlNet.py） ====================
if [[ "${ENABLE_CONTROLNET}" != "true" ]]; then
  echo "[ERROR] train_controlnet.sh 设计为 ControlNet 训练脚本，ENABLE_CONTROLNET 必须为 true" >&2
  exit 1
fi

if [[ "${ENABLE_VISUAL_LEARNER}" != "true" ]]; then
  echo "[ERROR] ControlNet 输出只在 visual_learner 分支中生效，请将 ENABLE_VISUAL_LEARNER=true" >&2
  exit 1
fi

if [[ "${CONTROLNET_ONLY}" == "true" && -z "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CONTROLNET_ONLY=true 时建议提供 CHECKPOINT_PATH 作为 baseline 初始化" >&2
  echo "        否则被冻结的 textual/visual/pq 仍为随机初始化，训练信号不稳定。" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT 不存在: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH 不存在: ${CHECKPOINT_PATH}" >&2
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
  --enable_controlnet
  --control_hint_source "${CONTROL_HINT_SOURCE}"
  --control_hint_channels "${CONTROL_HINT_CHANNELS}"
  --control_scales ${CONTROL_SCALES}
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
if [[ "${CONTROLNET_ONLY}" == "true" ]]; then
  ARGS+=(--controlnet_only)
fi

# ==================== 执行 ====================
echo "[INFO] Project Root: ${PROJECT_ROOT}"
echo "[INFO] Save Dir: ${SAVE_DIR}"
echo "[INFO] Dataset: ${DATASET_NAME} (${DATA_ROOT})"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "[INFO] ControlNet only: ${CONTROLNET_ONLY}"

echo "[INFO] Running: CUDA_VISIBLE_DEVICES=${DEVICE} python train_ControlNet.py ..."
CUDA_VISIBLE_DEVICES="${DEVICE}" python train_ControlNet.py "${ARGS[@]}"

echo "训练完成！模型保存在: ${SAVE_DIR}"