# CloudEdge-abnormal


**主路径**：**多边缘节点**（默认 3，可配）各自跑 Qwen3.5-0.8B 视觉塔多层 patch gallery → **RouteAgent（默认 GGUF Q4）** 决定是否上云 → 共享云端 Qwen3-VL(+LoRA) JSON 复核。  
**辅路径**：CLIP / DINOv3 gallery、Anomalib PaDiM（可选 OpenVINO 导出）与像素级对比。

技术栈：Anomalib + OpenVINO/ONNX + llama-cpp-python（RouteAgent Q4）+（可选）KubeEdge/Sedna + MLflow + FastAPI Web。

边缘舰队 + 物理网络环境（`configs/default.yaml`）：

```yaml
collab:
  num_edge_nodes: 3
  edge_fleet:
    num_nodes: 3
    default_categories: [bottle, cable, capsule]
    default_cities: [Suzhou, Shenzhen, Chengdu]   # 距云距离不同 → 传播时延不同
  network_env:                 # src/network_env.py
    enabled: true
    cloud: {city: Shanghai}
    route_stretch: 1.5         # 光纤绕路系数
    fiber_km_per_ms: 200.0     # 光纤中约 c/1.5
    time_scale: 8.0            # 演示加速 OU/昼夜波动
```

链路模型：Haversine 距离 → 光纤路径 → 传播 RTT，再叠加接入网、OU 拥堵、昼夜负载、突发/偶发断网（随时间变化）。

```bash
python scripts/smoke_edge_fleet.py --config configs/default.yaml
python scripts/smoke_network_env.py --seconds 6
# 覆盖边缘数量：--num-nodes 5
```

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

**选型（默认边侧 = Qwen-ML）**：同族易与云端 Qwen-VL 对齐、峰值约 0.2GB；精度上限可换 DINOv3/CLIP-ML；极致轻量 / OpenVINO 可回退 PaDiM-full；PaDiM-16shot 仅作公平对照。

#### 可视化样例

列顺序：Image | GT | CLIP-ML | DINOv3-ML | Qwen-ML | PaDiM-16 | PaDiM-full

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

### 量化前后对比（HF vs GGUF）

量化包 = mmproj-F16（视觉）+ Q4_K_M（LLM）。

#### 1）边侧视觉 AD（15 类均值，16-shot，layers `[6,8,10,12]`）

AD **只加载视觉塔**；mmproj 为 F16，与 HF 视觉权重近 bit-exact → **精度几乎不变**。

| Metric | HF（未压缩视觉） | mmproj GGUF | Δ |
|--------|------------------|-------------|---|
| Image-AUROC | 0.9518 | 0.9517 | −0.0001 |
| Pixel-AUROC | 0.9425 | 0.9425 | 0.0000 |
| Pixel-F1 | 0.4896 | 0.4897 | +0.0001 |
| Image-F1 | 0.9430 | 0.9431 | +0.0001 |
| Latency ms | 981 | 1025 | +44 |
| Peak MB | 392 | 391 | −1 |
| FLOPs G | 22.87 | 22.87 | 0 |
| Vision disk MB | 192 | 196 | — |
| Full package MB | 1666 | **703** | **−58%** |

#### 2）RouteAgent 全模（bottle，4 样本；HF bf16 vs Q4 LLM + mmproj）

此处才加载 **Q4 decoder**，显存/延迟才拉开。Peak = 加载增量 VRAM。

| Metric | HF bf16 | GGUF Q4+mmproj | Δ |
|--------|---------|----------------|---|
| Load s | 5.75 | 0.91 | −4.84 |
| Latency ms | 3014 | **432** | **−2583** |
| Peak mem MB | 1798 | **1064** | **−734** |
| Parse OK rate | 1.00 | 1.00 | 0 |
| Package MB | 1666 | **703** | **−963** |

结论：边侧 AD 指标可视为无损；默认上云路由用 Q4 约 **省 0.7GB VRAM、延迟降约 7×**（桌面 GPU 开发参考）。

---



权重角色：

| 权重 | 用途 |
|------|------|
| Qwen3.5-0.8B（HF） | 边侧 vision AD；可选 RouteAgent 全量 |
| Qwen3.5 GGUF（mmproj-F16 + Q4） | **默认 RouteAgent** |
| CLIP / DINOv3 | 可选边侧 gallery |
| Qwen3-VL-4B/8B | 云端 VLM |

数据：MVTec-AD。

---

## Web 控制台

```bash
conda activate clip          # 必须；base 没有 llama_cpp
CUDA_VISIBLE_DEVICES=0 WEB_VLM_DEVICE=cuda:0 WEB_ROUTE_DEVICE=cuda:0 \
  python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

| 环境变量 | 含义 |
|----------|------|
| `WEB_ROUTE_DEVICE` | RouteAgent 设备 |
| `WEB_ROUTE_BACKEND` | `gguf`（默认）或 `hf` |
| `WEB_ROUTE_GGUF_DIR` | 量化包目录（可选覆盖） |
| `WEB_VLM_DEVICE` | 云端 LoRA 设备 |

- 页面：**Overview** · **Topology**（云边远近 + 链路实时 RTT/带宽/丢包）· **Node Demo**（单节点案例）  
- Topology 按地理距离排布边缘；Demo 绑定所选节点的物理链路  
- 建议先点 **Preload RouteAgent (Q4)**  

默认配置：`collab.route_agent.backend: gguf`，`vision_mode: text`；`network_env.enabled: true`。

---

## 边侧默认：Qwen3.5-0.8B（多层 patch）

边侧 AD **只用视觉塔**；上云决策走 RouteAgent（LLM，默认 Q4）。  
AD 已出 `score/thr/network` 后，Route 默认 **只吃文本 CONTEXT**（`vision_mode: text`），避免 GGUF/HF 再编一次图；需要看图时设 `vision_mode: full`。

```bash
conda activate clip
CUDA_VISIBLE_DEVICES=0 python -m edge.infer \
  --config configs/edge_qwen35.yaml \
  --image datasets/mvtec/bottle/test/broken_large/000.png \
  --category bottle
# 可选：--method clip|dinov3|padim|qwen35_q
# --no-route-agent 关掉路由；--network-profile outage 测断网硬门控
```

统一协议：`train/good` → gallery（默认 16-shot）；`test/*` → 评测。  
默认层：Qwen `[6,8,10,12]`（merger 前）；CLIP/DINO `[12,16,20,24]`；`--fusion-temp 0.5`。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_edge_pixel_viz.py \
  --methods qwen35 --categories all --max-gallery 16 \
  --fusion-temp 0.5 --tag mlpatch16_all15 --shard qwen --skip-viz
```

---

## 量化复现

| 组件 | 精度 | 谁用 |
|------|------|------|
| mmproj | F16 | 边侧 AD（可选）+ RouteAgent |
| LLM | Q4 | **仅 RouteAgent**（默认） |

指标见上文「量化前后对比」。复现：

```bash
conda activate clip

# 视觉 AD：HF vs mmproj
CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen_quant_compare.py \
  --categories all --max-gallery 16 --tag quant_cmp_all15 --skip-viz

# RouteAgent：HF bf16 vs GGUF Q4
CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_quant_compare.py \
  --category bottle --n-samples 4 --tag route_q4

# RouteAgent：full（二次视觉）vs text（复用 AD CONTEXT）
CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_vision_reuse.py \
  --category bottle --n-samples 8 --tag route_reuse
# bottle×8 / GGUF：~473ms → ~269ms（约 1.76×），text 与 CONTEXT 规则一致率 100%
```

`llama-cpp-python` 需带 CUDA（预编译 wheel 不兼容时可源码编译）：

```bash
conda activate clip
export CUDA_HOME=/usr/local/cuda-12.1
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70;75;80;86"
export FORCE_CMAKE=1
pip install 'llama-cpp-python==0.3.34' --no-cache-dir --force-reinstall --no-binary llama-cpp-python
# 必要时：pip install 'numpy==1.24.3'
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

---

## 边侧可选：Anomalib PaDiM

```bash
conda activate dinov3

CUDA_VISIBLE_DEVICES=0 python scripts/train_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category all --no-export --skip-existing

CUDA_VISIBLE_DEVICES=0 python scripts/bench_anomalib.py \
  --config configs/default.yaml --device cuda:0 --category all

CUDA_VISIBLE_DEVICES=0 python scripts/bench_network_profiles.py \
  --config configs/default.yaml --device cuda:0 --category bottle

CUDA_VISIBLE_DEVICES=0 python scripts/bench_padim_kshot.py \
  --shots 16 --seed 42 --categories all --device cuda:0 --tag padim16shot
```

网络剖面：`good|fair|weak|outage|custom`。弱网时难例上云失败则边侧本地决策。

---

## 混合：边侧快检 + 云端 Qwen-VL

```bash
conda activate clip
CUDA_VISIBLE_DEVICES=0 python -m edge.infer \
  --image datasets/mvtec/bottle/test/broken_large/000.png --category bottle
CUDA_VISIBLE_DEVICES=0 python -m edge.infer \
  --image datasets/mvtec/bottle/test/broken_large/000.png --category bottle \
  --network-profile outage

CUDA_VISIBLE_DEVICES=0 python scripts/bench_route_agent.py \
  --config configs/default.yaml --category bottle --max-samples 24

conda activate dinov3
CUDA_VISIBLE_DEVICES=0 python scripts/export_edge_scores.py \
  --config configs/hybrid.yaml --category bottle

conda activate clip
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
python scripts/train_qwen_vl_lora.py --config configs/qwen_vl_lora_8b.yaml
```

---

