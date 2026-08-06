# CloudEdge-abnormal

云边协同工业缺陷检测（挑战杯 / XH-202606）。

**主路径**：边侧 Anomalib（PaDiM / ResNet18）快检 → 难例上云 → Qwen3-VL(+LoRA) JSON 复核。  
**辅路径**：边侧特征 gallery 对比（CLIP / DINOv3 / Qwen3.5-0.8B 视觉塔）与资源指标评测。

技术栈：Anomalib + OpenVINO/ONNX +（可选）KubeEdge/Sedna + MLflow + FastAPI Web。

## 当前结论（MVTec-15）

### 边侧方法对比（16-shot gallery / PaDiM 全量 good）

| Method | Image-AUROC | F1 | GFLOPs | Peak Mem | ≤1.5GB? |
|--------|-------------|----|--------|----------|---------|
| **PaDiM ResNet18（全量 good）** | **0.9145** | **0.9289** | **1.8** | **0.19 GB** | YES |
| DINOv3 ViT-L/16 gallery | 0.9013 | 0.9141 | 60.9 | 1.15 GB | YES |
| CLIP ViT-L/14 gallery | 0.8834 | 0.9108 | 77.8 | 1.16 GB | YES |
| Qwen3.5-0.8B vision gallery | 0.8714 | 0.9039 | ~17.5@224 | 0.30 GB | YES |
| PaDiM 16-shot（公平对比） | 0.8164 | 0.8969 | 1.8 | — | YES |

- 协议：特征法 gallery 仅用 `train/good`（16-shot，seed=42）；测试仅用 `test/*`，无泄漏。  
- 赛题 **≤1.5GB** = 单次推理**峰值内存**（非 FLOPs）。  
- 详细表：`outputs/reports/edge_methods/edge_methods_16shot_all15.md`、`edge_methods_memory.md`、`edge_methods_padim_full_vs_16shot.md`。

### 混合协同（边侧 PaDiM + 云端 VLM LoRA）

| Scheme | Mean F1 (15 类) |
|--------|-----------------|
| B1 边侧-only | 0.9285 |
| Zero-shot Qwen3-VL-8B | 0.8898 |
| LoRA 4B | 0.9255 |
| LoRA 8B | 0.9261 |

报告：`outputs/hybrid_lora_8b/all_categories.md`、`outputs/reports/mvtec_mean.md`。

---

## 环境

```bash
# Anomalib / PaDiM / DINOv3 AD
conda activate dinov3

# Web / Qwen3-VL / Qwen3.5 vision / CLIP
conda activate base   # transformers≥4.57（Qwen3VL）
```

权重默认在 `/data2/zlt/anomaly_detection_llm/model_card/`：

- `clip-vit-large-patch14`
- `dinov3-vitl16-pretrain-lvd1689m`
- `Qwen3.5-0.8B`（仅用视觉塔）
- `Qwen3-VL-4B-Instruct` / `Qwen3-VL-8B-Instruct`

数据：`datasets/mvtec` → MVTec-AD。

---

## Web 控制台

```bash
conda activate base
cd /data2/zlt/code/CloudEdge-abnormal
CUDA_VISIBLE_DEVICES=0 WEB_VLM_DEVICE=cuda:0 \
  python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

浏览器：`http://<host>:7860`

- 总览 / 指标：15 类 B1 / 零样本 / LoRA  
- 协同演示：边侧分数 → 路由 → LLM JSON；Anomalib 热力图（PaDiM / PatchCore，非 VLM）  
- LLM Cases：`hybrid_lora_8b` 难例复核  

Demo 请选带 `[LLM]` 标记的样本；标记按 `缺陷类型/文件名` 精确匹配缓存。

---

## 边侧 Anomalib（主方法）

```bash
conda activate dinov3

# 训练边侧 PaDiM + 云端 PatchCore
CUDA_VISIBLE_DEVICES=0 python scripts/train_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category all --no-export --skip-existing

# B0 / B1 / S
CUDA_VISIBLE_DEVICES=0 python scripts/bench_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category all
```

产物：`outputs/anomalib/<cat>/`、`outputs/reports/mvtec_mean.md`。

### PaDiM 16-shot（与特征法公平对比）

```bash
conda activate dinov3
CUDA_VISIBLE_DEVICES=0 python scripts/bench_padim_kshot.py \
  --shots 16 --seed 42 --categories all --device cuda:0 --tag padim16shot
```

---

## 边侧特征 gallery 对比（CLIP / DINOv3 / Qwen3.5）

统一协议：`train/good` → gallery；`test/*` → 评测；可选 `--max-gallery 16`。

```bash
# CLIP / Qwen（base）
conda activate base
CUDA_VISIBLE_DEVICES=0 python scripts/bench_edge_methods.py \
  --methods clip --categories all --max-gallery 16 --device cuda:0 --tag clip16all

CUDA_VISIBLE_DEVICES=0 python scripts/bench_edge_methods.py \
  --methods qwen35 --categories all --max-gallery 16 --device cuda:0 --tag qwen16all

# DINOv3 / 现成 PaDiM ckpt（dinov3）
conda activate dinov3
CUDA_VISIBLE_DEVICES=0 python scripts/bench_edge_methods.py \
  --methods dinov3 --categories all --max-gallery 16 --device cuda:0 --tag dino16all

CUDA_VISIBLE_DEVICES=0 python scripts/bench_edge_methods.py \
  --methods padim --categories all --device cuda:0 --tag padim16all
```

核心代码：`edge/methods/`（`gallery_ad.py`、`encoders.py`、`padim_ad.py`）。

---

## 混合：边侧 Anomalib + 云端 Qwen-VL

```bash
# 1) 导出边侧分数
conda activate dinov3
CUDA_VISIBLE_DEVICES=0 python scripts/export_edge_scores.py \
  --config configs/hybrid.yaml --category bottle

# 2) 零样本 / LoRA 云端复核
conda activate base
CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid.py \
  --config configs/hybrid.yaml --category bottle

CUDA_VISIBLE_DEVICES=0 python scripts/bench_hybrid_multi.py \
  --config configs/hybrid_lora_8b.yaml \
  --categories bottle,screw,cable --max-cloud-reviews 16 --device cuda:0
```

### LoRA 微调（OK/NG JSON）

```bash
conda activate base
python scripts/build_vlm_sft_data.py --config configs/qwen_vl_lora.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen_vl_lora.py --config configs/qwen_vl_lora.yaml --max-images 60

# 8B
python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora_8b.yaml
```

产物：`outputs/qwen_vl_lora*/adapter/`、`outputs/hybrid_lora*/`。

可选双 VLM（边 4B / 云 8B）：`configs/qwen_vl.yaml` + `scripts/bench_qwen_vl.py`（**非**当前 hybrid 主路径）。

---

## 目录

```text
configs/                 # default / hybrid / hybrid_lora* / qwen_vl*
scripts/
  train_anomalib.py      # 边/云 Anomalib 训练
  bench_anomalib.py      # B0/B1/S
  bench_edge_methods.py  # CLIP/DINOv3/Qwen/PaDiM 对比
  bench_padim_kshot.py   # PaDiM k-shot
  export_edge_scores.py / bench_hybrid*.py
  train_qwen_vl_lora.py / eval_qwen_vl_lora.py
edge/
  methods/               # gallery AD + encoders
  infer.py / vlm_infer.py
cloud/                   # 云端复核
web/                     # FastAPI 控制台
src/vlm/                 # Qwen-VL 客户端 / 路由
models/                  # open_clip / dino 工具（辅助）
docs/                    # 方案与指标清单
outputs/reports/         # 汇总指标
deploy/kubeedge/         # 部署骨架
datasets/mvtec           # 数据软链
```

---

## 说明

- 边侧主推 **PaDiM + 全量正常样本**；特征 gallery 适合少样本叙事，但同 16-shot 下仍弱于全量 PaDiM。  
- VLM **不出热力图**；Web 上 Cloud PatchCore 条带仅为传统 AD 对比。  
- 内存 / 时延以目标边缘硬件（OpenVINO 等）复测为准；桌面 GPU 数字仅作开发参考。  
- KubeEdge 真集群不阻塞本阶段指标；难例路由先验证协同收益。
