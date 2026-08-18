# CloudEdge-abnormal

工业缺陷检测的云边协同方案：边缘端轻量快检，边界/高不确定样本按需上云复核。

## 架构

```
边缘节点（多节点，可配）
  └─ Qwen3.5-0.8B 视觉塔 AD（多层 patch kNN，image-level 判定）
        │
        ▼ 不确定 / 网络允许
CRR 手写路由（cost-risk）
  = 不确定度 + 多节点冲突 + 实时网络代价 + 云负载
        │ 上云
        ▼
共享云端检测器
  └─ DINOv3（像素 kNN）+ Qwen3.5（语义）保守融合
       （"只增强、不压制"）
```

- 检测标签由 **边缘 AD / 云端融合检测器** 给出；LLM/DINO 只做检测，**不负责路由**。
- 上云决策默认只由 **CRR 手写算法** 给出（`collab.route_agent.enabled: false`）；`RouteAgent`（LLM）是可选实验基线。
- 网络链路为物理仿真：Haversine 距离 → 光纤传播 RTT + 接入/拥塞/昼夜负载/突发断网。

## 关键文件

| 路径 | 说明 |
|------|------|
| `src/collab_routing/` | CRR 手写路由 + 多节点冲突/仲裁 + 云接纳 |
| `src/cloud_reviewer.py` | 统一云端入口：DINOv3 + Qwen3.5 融合 |
| `src/cloud_load.py` | 云端离散事件队列仿真（负载感知） |
| `edge/infer.py` | 边缘推理入口（Qwen3.5-0.8B patch-gallery） |
| `third_part/Cloud-abnormal-cx/` | 云端融合检测器（DINOv3 + Qwen 保守融合） |
| `third_part/cloud-edge-xj/` | 参考的多节点协同方案 |

## 权重角色

| 权重 | 用途 |
|------|------|
| Qwen3.5-0.8B（HF） | 边侧 vision AD |
| Qwen3.5 GGUF（Q4） | 可选 RouteAgent（实验基线） |
| DINOv3 ViT-L | 云端像素 kNN |
| Qwen3.5-2B / 9B | 云端语义复核（与 DINOv3 保守融合） |

数据：MVTec-AD（含 `mvtec_anomaly_llm` 分割）。

## 快速开始

```bash
conda activate clip

# 边缘单图推理（CRR 路由；--network-profile outage 测断网硬门控）
CUDA_VISIBLE_DEVICES=0 python -m edge.infer \
  --config configs/edge_qwen35.yaml \
  --image datasets/mvtec/bottle/test/broken_large/000.png \
  --category bottle

# 云端单图复核（DINO+Qwen 融合）
python cloud/vlm_review.py --image <img> --category bottle --threshold 0.67

# 云边协同端到端（真实模型 + 真实延迟）
CUDA_VISIBLE_DEVICES=0 python scripts/bench_e2e_collab_live.py \
  --categories bottle,cable --profiles fair,weak --limit 12

# 三方案对比（edge / cloud / collab_sft，可并行分卡）
CUDA_VISIBLE_DEVICES=0 python scripts/bench_cloud_edge_compare.py --scheme edge
CUDA_VISIBLE_DEVICES=6 python scripts/bench_cloud_edge_compare.py --scheme cloud
CUDA_VISIBLE_DEVICES=1 python scripts/bench_cloud_edge_compare.py --scheme collab_sft
```

## Web 控制台

```bash
CUDA_VISIBLE_DEVICES=0 python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

页面：Overview · Topology（云边远近 + 实时 RTT/带宽/丢包）· Node Demo。
