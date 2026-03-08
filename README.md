## 项目说明：自定义数据集的训练与测试

本仓库主要用于在自定义异常检测数据集上，基于视觉/文本/PQ 多适配器（adapter）的模型进行**训练与测试**。  
核心流程分为三步：

- **数据准备**：利用MVTec与HXZY数据集（v3）
- **训练**：使用 `scripts/train_custom_dataset.sh` 在自定义数据集上微调适配器
- **测试**：
使用 `scripts/test_custom_dataset_v1.sh` 
或 `scripts/test_custom_dataset_v2.sh` 在指定数据集上评估模型

下面对数据集格式和每个脚本的用法进行详细说明。

---

## 1. 数据集结构与准备

### 1.1 自定义数据集目录结构

假设自定义数据集根目录为：

- **训练集示例**：`/data1/GYS/HXZY_traindata_v3`（以1_、2_开头的为part1，例如1_D01-1-1_5059）
- **测试集示例**：`/data1/GYS/HXZY_testdata_v3`（以3_、4_开头的为part2）

根目录下包含多个子目录，每个子目录对应一个部位_车号，例如：

- `1_D01-1-1_5059/`
- `1_D01-1-2_5059/`
- `...`

每个子目录内部推荐组织为（以 `1_D01-1-1_5059` 为例）：

- `1_D01-1-1_5059/anomaly_query/`：异常样本图像
- `1_D01-1-1_5059/mask/`：异常样本对应的像素级标注（mask）
- `1_D01-1-1_5059/normal_query/`：正常样本图像
- `1_D01-1-1_5059/reference/`：用于 few-shot prompt 的参考正常样本

### 1.2 文件命名规则（异常图 & mask）

`dataset/custom_dataset.py` 假设异常图和 mask 遵循如下命名模式：

- 异常图像文件名：`<prefix>_result_<id>.png`  
  例如：`diushi_or_songdong_result_0.png`
- 对应 mask 文件名：`<prefix>_mask_<id>.png`  
  例如：`diushi_or_songdong_mask_0.png`

脚本会：

- 从 `anomaly_query/` 中读取 `*_result_*.png` / `*.jpg` / `*.jpeg`
- 自动把 `_result_` 替换为 `_mask_`，在 `mask/` 目录中寻找对应的标注
- 将：
  - `anomaly_query` 中的图片全部作为 **test 阶段**、`anomaly=1`
  - `normal_query` 中的图片全部作为 **test 阶段**、`anomaly=0`
  - `reference` 中的图片全部作为 **train 阶段**、`anomaly=0`（few-shot 参考）

生成的 `meta.json` 保存在：

- `<dataset_root>/meta.json`

---
## 2. 数据准备：`dataset/custom_dataset.py`

根据数据集生成meta.json
sh脚本中已附带该过程

## 3. 训练脚本：`scripts/train_custom_dataset.sh`

### 3.1 脚本整体流程

脚本主要做三件事：

1. 设置 GPU 和数据/训练超参数
2. **自动选择数据准备脚本**：根据 `dataset_name` 自动选择对应的数据准备脚本生成/更新 `meta.json`
   - `dataset_name="visa"` → 使用 `dataset/visa.py`
   - `dataset_name="mvtec"` → 使用 `dataset/mvtec.py`
   - 其他（如 `hxzy_part1`、`hxzy_part2` 等）→ 使用 `dataset/custom_dataset.py`
3. 调用 `train_custom_dataset.py` 在自定义数据集上训练 adapter

### 3.2 关键配置项

脚本内默认配置如下（可根据需要修改）：

- **硬件**
  - `device=0`：所使用的 GPU id（对应 `CUDA_VISIBLE_DEVICES`）

- **数据集相关**
  - `data_root=/data1/GYS/HXZY_traindata_v3`：数据集根目录
  - `dataset_name=hxzy_part1`：当前训练使用的数据集名称（仅作为标识，用于保存路径命名等）

- **模型与特征**
  - `pretrained_model ViT-L/14@336px`
  - `checkpoint_path=""`：**可选参数**，用于从已有 checkpoint 继续训练
    - **为空字符串**：从头开始训练（adapter 权重随机初始化）
    - **填写路径**：从该 checkpoint 加载 adapter 权重继续训练
    - 示例：`checkpoint_path="adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth"`
    - **注意**：训练代码会自动检查 checkpoint 是否存在，如果路径无效会给出警告并从头训练
  - `n_ctx=12`
  - `features_list 6 12 18 24`
  - `vl_reduction=4`
  - `pq_mid_dim=128`
  - `--visual_learner --textual_learner --pq_learner --pq_context`：开启三种 adapter 及其融合

- **训练超参数**
  - `epochs=15`
  - `learning_rate=0.001`
  - `batch_size=8`
  - `image_size=518`
  - `k_shots=1`：few-shot 数量（训练时使用的参考样本数）
  - `seed=10`

- **保存路径**
  - `base_dir=train_on_${dataset_name}_3adapters_batch${batch_size}_v2`
  - `save_dir=./adapter_checkpoints/${base_dir}`  
    训练好的权重会保存到该目录下，例如：  
    `adapter_checkpoints/train_on_hxzy_part1_3adapters_batch8_v2/epoch_*.pth`

### 3.3 使用方法

1. **确认/修改数据集路径**

   打开 `scripts/train_custom_dataset.sh`，根据实际情况修改：

   - `data_root=/data1/GYS/HXZY_traindata_v3`
   - `dataset_name=hxzy_part1`

2. **确认初始化 checkpoint（可选）**

   - **从头训练**：保持 `checkpoint_path=""` 为空字符串（默认）
   - **从已有 checkpoint 继续训练**：修改 `checkpoint_path` 变量，例如：
     ```bash
     checkpoint_path="adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth"
     ```
   - 脚本会自动判断 checkpoint 是否存在，如果不存在会给出警告并从头训练

3. **运行训练**

```bash
cd /home/root123/GYS/fine_tuning

bash scripts/train_custom_dataset.sh
```

训练完成后会看到：

- `训练完成！模型保存在: ./adapter_checkpoints/train_on_...`

---

## 4. 测试脚本 v1：`scripts/test_custom_dataset_v1.sh`

`test_custom_dataset_v1.sh` 是 **v1 版本**测试脚本，使用 **BOX 评价指标**（mAP@0.2），固定使用 `test_custom_dataset_v1.py` 进行测试。

### 4.1 数据集与模型映射

脚本内部定义了两个映射：

- **数据集路径映射 `dataset_map`**：
  - `hxzy_part1` → `/data1/GYS/HXZY_traindata_v3`
  - `hxzy_part2` → `/data1/GYS/HXZY_testdata_v3`
  - `hxzy`       → `/data1/GYS/HXZY_testdata_v3`
  - `visa`       → （需要根据实际情况添加路径）
  - `mvtec`      → （需要根据实际情况添加路径）

**注意**：脚本会根据 `test_dataset` 的值自动选择数据准备脚本：
- `test_dataset="visa"` → 使用 `dataset/visa.py`
- `test_dataset="mvtec"` → 使用 `dataset/mvtec.py`
- 其他 → 使用 `dataset/custom_dataset.py`

- **模型 checkpoint 映射 `model_map`**：
  - `visa`          → `adapter_checkpoints/train_on_visa_3adapters_batch8/epoch_15.pth`
  - `mvtec`         → `adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth`
  - `hxzy`          → `adapter_checkpoints/hxzy_full/epoch_15.pth`
  - `hxzy_part1`    → `adapter_checkpoints/train_on_hxzy_part1_3adapters_batch8/epoch_15.pth`
  - `hxzy_part2`    → `adapter_checkpoints/train_on_hxzy_part2_3adapters_batch8/epoch_15.pth`
  - `hxzy_part1_v2` → `adapter_checkpoints/train_on_hxzy_part1_3adapters_batch8_v2/epoch_15.pth`

可以根据你的实际训练结果，修改这些映射。

### 4.2 测试配置

脚本顶部关键变量：

- `device=0`
- `train_dataset=hxzy_part1_v2`：**训练该模型时使用的数据集名称**，用于从 `model_map` 中选出对应 checkpoint
- `test_dataset=hxzy_part2`：测试使用的数据集（从 `dataset_map` 中找路径）
- `k_shots=1`：
  - `0`：zero-shot，仅使用预训练模型，不使用 reference 样本
  - `>0`：few-shot，从 `reference/` 中采样 `k` 张正常样本构建 prompt memory
- `test_script=test_custom_dataset_v1.py`：**固定使用 v1 测试脚本**（不可修改）
- `box_thresh=0.16`：Box 检测阈值（v1 测试脚本特有参数）
- **评价指标**：使用 **mAP@0.2**（IoU 阈值为 0.2）进行 BOX 级别的异常检测评估

脚本会根据 `k_shots` 自动设置种子 `seeds`：

- `k_shots = 0` → `seeds="10"`
- `k_shots > 0` → `seeds="20"`（当前 reference 很少，默认只用一个 seed）

结果保存目录：

- `save_dir=./results/train_on_${train_dataset}_test_on_${test_dataset}_3adapters_batch8_v1`

### 4.3 使用方法

1. **根据需求修改脚本头部配置**：

   - `train_dataset`、`test_dataset`、`k_shots`
   - 如有新的 checkpoint 路径，更新 `model_map[...]`
   - 如需调整 Box 检测阈值，修改 `box_thresh`（默认 0.16）

2. **运行测试**：

```bash
cd /home/root123/GYS/fine_tuning

bash scripts/test_custom_dataset_v1.sh
```

脚本会打印：

- 数据集路径
- 模型 checkpoint 绝对路径
- 使用的测试脚本版本
- 当前 seed、k_shots

完成后输出：

- `测试完成！结果保存在: ./results/train_on_...`

---

## 5. 测试脚本 v2：`scripts/test_custom_dataset_v2.sh`

`test_custom_dataset_v2.sh` 是 **v2 版本**测试脚本，使用 **AUPRO 评价指标**，固定使用 `test_custom_dataset_v2.py` 进行测试。

### 5.1 主要配置

- 数据集路径映射 `dataset_map`、模型映射 `model_map` 与 v1 基本一致
- **数据准备**：脚本会根据 `test_dataset` 的值自动选择数据准备脚本（与 v1 相同）
- 关键变量：
  - `device=0`
  - `train_dataset=hxzy_part1_v2`
  - `test_dataset=hxzy_part2`
  - `k_shots=1`
  - `test_script=test_custom_dataset_v2.py`：**固定使用 v2 测试脚本**（不可修改）
  - `n_ctx=12`
  - `vl_reduction=4`
  - `pq_mid_dim=128`
  - **注意**：v2 测试脚本**不支持** `--box_thresh` 参数（v1 脚本支持）

保存目录：

- `save_dir=./results/train_on_${train_dataset}_test_on_${test_dataset}_3adapters_batch8_v2`

### 5.2 few-shot / zero-shot 设置

同 v1：

- `k_shots=0`：
  - `seeds="10"`，只跑一次
  - 不使用 `reference` 目录的图像
- `k_shots>0`：
  - `seeds="20"`（目前配置）
  - 使用 `reference` 目录作为 few-shot prompt memory

### 5.3 使用方法

1. **修改脚本头部参数**

   根据实际情况修改：

   - `train_dataset`、`test_dataset`、`k_shots`
   - 新的 checkpoint 路径填入 `model_map`
   - 如有需要，调整 `n_ctx`、`vl_reduction`、`pq_mid_dim` 等超参数
   - **注意**：v2 脚本不支持 `box_thresh` 参数（v1 脚本支持）

2. **运行测试**

```bash
cd /home/root123/GYS/fine_tuning

bash scripts/test_custom_dataset_v2.sh
```

脚本会为每个 `seed` 调用：

- `python test_custom_dataset_v2.py ... --visualize`

结果与可视化将保存在：

- `./results/train_on_${train_dataset}_test_on_${test_dataset}_3adapters_batch8_...`

---

## 6. 测试脚本内部（`test_custom_dataset_v1.py`）的一些要点（简要）

`test_custom_dataset_v1.py`（`test_custom_dataset_v2.py` 结构类似）的大致流程：

- 解析命令行参数：
  - `--dataset`、`--test_data_path`、`--seed`、`--k_shots`
  - `--checkpoint_path`、`--save_path`
  - `--features_list`、`--image_size`、`--batch_size`
  - `--n_ctx`、`--vl_reduction`、`--pq_mid_dim`
  - `--visual_learner --textual_learner --pq_learner --pq_context --visualize`
- 加载：
  - 自定义 `Dataset` / `PromptDataset`
  - 训练好的 `PQAdapter` / `TextualAdapter` / `VisualAdapter`
  - `meta.json` 中的图像与 mask 信息
- 前向推理并计算指标：
  - 像素级 ROC-AUC / AP
  - 基于 box 的 precision / recall / F1 / mAP 等
  - 支持固定阈值/PR 曲线等
- 保存：
  - log 文本（例如保存在 `results/..._log.txt`）
  - 可选的可视化结果（热力图、检测框等）

通常不需要直接修改 Python 测试脚本；大多数实验配置通过 **shell 脚本的参数** 即可完成。

---

## 7. 常见操作示例

- **1）首次在 HXZY 训练集上训练 adapter（从头训练）**

```bash
# 在 scripts/train_custom_dataset.sh 中设置：
#   - data_root=/data1/GYS/HXZY_traindata_v3
#   - dataset_name=hxzy_part1
#   - checkpoint_path=""  # 保持为空，从头训练

# 运行训练（脚本会自动调用 dataset/custom_dataset.py 生成 meta.json）
bash scripts/train_custom_dataset.sh
```

- **1b）从 MVTec 预训练模型继续在 HXZY 上训练**

```bash
# 在 scripts/train_custom_dataset.sh 中设置：
#   - checkpoint_path="adapter_checkpoints/train_on_mvtec_3adapters_batch8/epoch_15.pth"

# 运行训练
bash scripts/train_custom_dataset.sh
```

- **2）使用训好的 hxzy_part1_v2 模型，在 HXZY 测试集上做 few-shot 测试（v1 脚本）**

在 `scripts/test_custom_dataset_v1.sh` 中设置：

- `train_dataset=hxzy_part1_v2`
- `test_dataset=hxzy_part2`
- `k_shots=1`

然后执行：

```bash
bash scripts/test_custom_dataset_v1.sh
```

- **3）使用同一模型，切换为 zero-shot 测试（不使用 reference 样本）**

在测试脚本里修改：

- `k_shots=0`

重新执行：

```bash
bash scripts/test_custom_dataset_v1.sh
```

- **4）使用 v2 测试脚本（不同后处理或指标计算）**

修改/确认 `scripts/test_custom_dataset_v2.sh` 里的参数，然后执行：

```bash
bash scripts/test_custom_dataset_v2.sh
```

**注意**：v1 和 v2 测试脚本的区别：
- **v1 脚本**：固定使用 `test_custom_dataset_v1.py`，使用 **BOX 评价指标（mAP@0.2）**，支持 `--box_thresh` 参数
- **v2 脚本**：固定使用 `test_custom_dataset_v2.py`，使用 **AUPRO 评价指标**，不支持 `--box_thresh` 参数

- **5）在 MVTec 或 VisA 数据集上训练/测试**

脚本会自动根据 `dataset_name` 或 `test_dataset` 的值选择对应的数据准备脚本：
- `dataset_name="mvtec"` → 自动使用 `dataset/mvtec.py`
- `dataset_name="visa"` → 自动使用 `dataset/visa.py`
- 其他 → 自动使用 `dataset/custom_dataset.py`

无需手动调用数据准备脚本，训练/测试脚本会自动处理。

---

## 8. 小结（当前脚本版本说明）

- **数据集准备（自动选择）**：
  - 训练和测试脚本会根据 `dataset_name` / `test_dataset` 的值**自动选择**对应的数据准备脚本：
    - `visa` → `dataset/visa.py`
    - `mvtec` → `dataset/mvtec.py`
    - 其他（如 `hxzy_part1`、`hxzy_part2`）→ `dataset/custom_dataset.py`
  - 无需手动调用数据准备脚本，训练/测试脚本会自动处理
- **训练**：
  - 通过 `train_custom_dataset.sh` 指定 `data_root` 与 `dataset_name`
  - `checkpoint_path` 可为空字符串（从头训练）或填写路径（从 checkpoint 继续训练）
  - 训练好的 adapter 权重保存在 `adapter_checkpoints/...`
- **测试**：通过 `test_custom_dataset_v1.sh` / `test_custom_dataset_v2.sh` 配置 `train_dataset`、`test_dataset`、`k_shots` 等参数，结果与可视化保存在 `results/...`

当前脚本版本的一些注意点：

- **数据准备脚本自动选择**：所有训练/测试脚本都会根据数据集名称自动选择对应的数据准备脚本（`visa.py` / `mvtec.py` / `custom_dataset.py`），无需手动调用
- **checkpoint_path 可选**：训练脚本中的 `checkpoint_path` 可以为空字符串，代码会从头开始训练；如果填写路径但文件不存在，会给出警告并从头训练
- **测试脚本固定对应**：
  - `test_custom_dataset_v1.sh` **固定使用** `test_custom_dataset_v1.py`，使用 **BOX 评价指标（mAP@0.2）**，支持 `--box_thresh` 参数
  - `test_custom_dataset_v2.sh` **固定使用** `test_custom_dataset_v2.py`，使用 **AUPRO 评价指标**，不支持 `--box_thresh` 参数
  - 两个脚本互不干扰，可根据需要选择使用
- **断点续训**：`train_custom_dataset.py` 支持从 `--checkpoint_path` 自动恢复 adapter 权重和 `epoch` 信息，便于断点续训



