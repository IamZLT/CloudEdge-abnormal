# CloudEdge-abnormal


**主路径**：边侧 Anomalib（PaDiM / ResNet18）快检 → 难例上云 → Qwen3-VL(+LoRA) JSON 复核。  
**辅路径**：边侧多层 patch gallery（CLIP / DINOv3 / Qwen3.5-0.8B 视觉塔）与像素级对比。

技术栈：Anomalib + OpenVINO/ONNX +（可选）KubeEdge/Sedna + MLflow + FastAPI Web。

## 当前结论（MVTec-15）

### 边侧方法对比（16-shot + 多层 patch + 像素级）

特征法：**多层 mid→late patch-token NN → 距离图 → softmax 加权融合**（图像分 = map max）；像素指标在 256×256。

| Method | Image-AUROC | Pixel-AUROC | Pixel-F1 | Peak MB |
|--------|-------------|-------------|----------|---------|
| **DINOv3-ML** `[12,16,20,24]` | **0.9756** | **0.9716** | **0.5449** | 2352 |
| CLIP-ML `[12,16,20,24]` | 0.9681 | 0.9670 | 0.5350 | 1211 |
| PaDiM ResNet18（全量 good） | 0.9145 | 0.9666 | 0.5003 | 193 |
| Qwen-ML `[6,8,10,12]` pre-merge | 0.9378 | 0.9450 | 0.4339 | 217 |
| PaDiM 16-shot | 0.8164 | 0.9391 | 0.4389 | 193 |



**选型**：精度上限 → DINOv3/CLIP-ML；边缘部署 / ≤1.5GB → **PaDiM-full** 或 Qwen-ML；PaDiM-16shot 仅作公平对照。

#### 可视化样例

列顺序：Image | GT | CLIP-ML | DINOv3-ML | Qwen-ML | PaDiM-16 | PaDiM-full（更多见 `asserts/edge_mlpatch/`）。

**bottle**（物体类）

![bottle multi-layer compare](asserts/edge_mlpatch/bottle_compare.jpg)

**leather**（纹理类）

![leather multi-layer compare](asserts/edge_mlpatch/leather_compare.jpg)

**screw**（难例）

![screw multi-layer compare](asserts/edge_mlpatch/screw_compare.jpg)

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

# Web / Qwen3-VL / Qwen3.5 vision / CLIP（本机常用 clip env）
conda activate clip   # 或 base；需 transformers≥4.57 + safetensors
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

## 边侧多层 patch gallery（CLIP / DINOv3 / Qwen3.5）

统一协议：`train/good` → gallery；`test/*` → 评测；默认 `--max-gallery 16`。  
默认层：CLIP/DINO `[12,16,20,24]`，Qwen `[6,8,10,12]`；融合温度 `--fusion-temp 0.5`。

```bash
# 像素指标 + 逐类对比图（推荐）
# dinov3 env: CLIP + DINOv3；clip env: Qwen
CUDA_VISIBLE_DEVICES=1 python scripts/bench_edge_pixel_viz.py \
  --methods clip dinov3 --categories all --max-gallery 16 \
  --fusion-temp 0.5 --tag mlpatch16_all15 --shard clip_dino --skip-viz

CUDA_VISIBLE_DEVICES=2 python scripts/bench_edge_pixel_viz.py \
  --methods qwen35 --categories all --max-gallery 16 \
  --fusion-temp 0.5 --tag mlpatch16_all15 --shard qwen --skip-viz

# PaDiM（已有 ckpt 时）可并入同一 tag，或复用 maps/metrics 后：
python scripts/bench_edge_pixel_viz.py --viz-only --categories all --tag mlpatch16_all15
```

可选：旧版图像级 CLS-gallery 脚本 `scripts/bench_edge_methods.py`。  
核心代码：`edge/methods/encoders.py`、`patch_gallery_ad.py`、`pixel_metrics.py`、`viz_compare.py`。

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
  bench_edge_pixel_viz.py # 多层 patch + 像素指标 + 对比可视化
  bench_edge_methods.py  # 旧版 CLS-gallery 对比
  bench_padim_kshot.py   # PaDiM k-shot
  export_edge_scores.py / bench_hybrid*.py
  train_qwen_vl_lora.py / eval_qwen_vl_lora.py
edge/
  methods/               # encoders / patch_gallery_ad / pixel_metrics / viz
  infer.py / vlm_infer.py
cloud/                   # 云端复核
web/                     # FastAPI 控制台
src/vlm/                 # Qwen-VL 客户端 / 路由
models/                  # open_clip / dino 工具（辅助）
docs/                    # 方案与指标清单
asserts/edge_mlpatch/    # README 用可视化样例（jpg）
outputs/reports/         # 汇总指标
deploy/kubeedge/         # 部署骨架
datasets/mvtec           # 数据软链
```

---

## 说明

- 少样本图像/像素精度：多层 patch gallery（DINOv3/CLIP）高于全量 PaDiM；边缘部署仍主推 **PaDiM-full**（~0.19GB）。  
- Qwen 视觉塔与云端 Qwen-VL 同族，适合边云特征对齐，但 ViT-L 类显存更高。  
- VLM **不出热力图**；Web 上 Cloud PatchCore 条带仅为传统 AD 对比。  
- 内存 / 时延以目标边缘硬件（OpenVINO 等）复测为准；桌面 GPU 数字仅作开发参考。  
- KubeEdge 真集群不阻塞本阶段指标；难例路由先验证协同收益。
