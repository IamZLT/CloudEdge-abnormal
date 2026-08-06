# CloudEdge-abnormal

首选技术栈：

**Anomalib + OpenVINO/ONNX Runtime + KubeEdge（可选 Sedna 联合推理范式）+ MLflow + CVAT**

当前阶段目标：用 Anomalib 在本地数据上训练边侧/云端模型，导出边缘运行时，并给出 B0/B1/S 指标。

## Web 控制台

可视化总览指标、协同演示、云端 LLM 案例：

```bash
conda activate base
cd /data2/zlt/code/CloudEdge-abnormal
CUDA_VISIBLE_DEVICES=0 WEB_VLM_DEVICE=cuda:0 \
  python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

浏览器打开：`http://<host>:7860`

- **总览 / 指标**：15 类 B1 / 零样本 / LoRA 对比  
- **协同演示**：边侧分数 → 是否上云 → LLM JSON（可实时 LoRA）  
- **LLM 案例**：浏览 `hybrid_lora_8b` 难例复核结果  

## 环境

```bash
conda activate dinov3
# 已安装 anomalib / openvino / onnxruntime / mlflow
# Web / Qwen-VL：conda activate base
```
## 一键流程

```bash
# 1) 训练边侧 PaDiM(ResNet18) + 云端 PatchCore(WRN50)
CUDA_VISIBLE_DEVICES=0 python scripts/train_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category bottle

# 2) 评测 B0(全上云) / B1(仅边侧) / S(难例上云)
CUDA_VISIBLE_DEVICES=0 python scripts/bench_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category bottle
```

输出：

- `outputs/anomalib/bottle/...` 训练与导出产物
- `outputs/reports/bottle/metrics.md` 指标报告
- `outputs/mlruns/` MLflow 记录（若启用）

## Qwen-VL 云边协同（可选）

边侧小模型 **Qwen3-VL-4B**，云端大模型 **Qwen3-VL-8B**；难例（低置信）上云复核。

```bash
conda activate base   # 需要 transformers>=4.57（含 Qwen3VL）

# 单图：边侧
CUDA_VISIBLE_DEVICES=0 python edge/vlm_infer.py \
  --config configs/qwen_vl.yaml \
  --image datasets/mvtec/bottle/test/broken_large/000.png

# 单图：云端
CUDA_VISIBLE_DEVICES=1 python cloud/vlm_review.py \
  --config configs/qwen_vl.yaml \
  --image datasets/mvtec/bottle/test/broken_large/000.png

# 小样本 B0/B1/S 对比（默认 bottle 最多 20 张）
CUDA_VISIBLE_DEVICES=0,1 python scripts/bench_qwen_vl.py \
  --config configs/qwen_vl.yaml --category bottle --max-images 8
```

权重默认路径见 `configs/qwen_vl.yaml`（本机 `anomaly_detection_llm/model_card/`）。

## 混合：边侧 Anomalib + 云端大模型（推荐叙事）

边侧 PaDiM 快检给分数；难例上云由 **Qwen3-VL-8B** 直接推理并输出 LLM JSON（decision/reason）。

```bash
# 1) 导出边侧分数（dinov3）
conda activate dinov3
CUDA_VISIBLE_DEVICES=0 python scripts/export_edge_scores.py --config configs/hybrid.yaml --category bottle

# 2) 云端大模型复核难例（base）
conda activate base
CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid.py --config configs/hybrid.yaml --category bottle
```

输出：`outputs/hybrid/<category>/bench.md`、`llm_outputs.md`（含云端 LLM 原文）。

## Qwen-VL LoRA 微调（OK/NG JSON）

在 holdout 划分上微调边/云 VLM，对比零样本：

```bash
conda activate base

# 1) 构建 SFT 数据（train/good + 部分 test；另留 holdout）
python scripts/build_vlm_sft_data.py --config configs/qwen_vl_lora.yaml

# 2) LoRA 微调（默认 Qwen3-VL-4B）
CUDA_VISIBLE_DEVICES=3 python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml

# 3) holdout 上对比 zero-shot vs LoRA
CUDA_VISIBLE_DEVICES=3 python scripts/eval_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml --max-images 60
```

产物：`outputs/qwen_vl_lora/adapter/`、`eval_holdout.md`。

### 接入混合协同（边侧 Anomalib + 云端 LoRA VLM）

```bash
conda activate base
# 4B LoRA
CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid_multi.py \
  --config configs/hybrid_lora.yaml \
  --categories screw,cable,pill,capsule,zipper \
  --max-cloud-reviews 16 --device cuda:0

# 8B LoRA（云端大模型）
python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora_8b.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid_multi.py \
  --config configs/hybrid_lora_8b.yaml \
  --categories screw,cable,pill,capsule,zipper \
  --max-cloud-reviews 16 --device cuda:0
```

## 目录

```text
configs/                 # default.yaml + qwen_vl.yaml
scripts/train_anomalib.py
scripts/bench_anomalib.py
scripts/bench_qwen_vl.py # Qwen-VL 协同评测
src/vlm/                 # Qwen-VL 客户端 / 解析 / 路由
edge/vlm_infer.py        # 边侧 VLM
cloud/vlm_review.py      # 云端 VLM
deploy/kubeedge/         # KubeEdge/Sedna 骨架说明
docs/                    # 方案与任务分配
datasets/                # MVTec 等数据软链
```

## 说明

- KubeEdge 真集群部署不阻塞本阶段指标；先用 Sedna 风格难例路由验证协同收益。
- 旧的自研 `src/models.py` PatchCore-lite 仅作兜底，**视觉主路径以 Anomalib 为准**。
- Qwen-VL 路径用于对齐赛题「云端大模型 / 边侧轻量大模型」叙事，与 Anomalib 可并行。
