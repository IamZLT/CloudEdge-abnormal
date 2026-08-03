# CloudEdge-abnormal

首选技术栈：

**Anomalib + OpenVINO/ONNX Runtime + KubeEdge（可选 Sedna 联合推理范式）+ MLflow + CVAT**

当前阶段目标：用 Anomalib 在本地数据上训练边侧/云端模型，导出边缘运行时，并给出 B0/B1/S 指标。

## 环境

```bash
conda activate dinov3
# 已安装 anomalib / openvino / onnxruntime / mlflow
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

## 目录

```text
configs/                 # 默认配置
scripts/train_anomalib.py
scripts/bench_anomalib.py
edge/                    # 边侧推理入口（后续接 OpenVINO）
cloud/                   # 云端复核入口
deploy/kubeedge/         # KubeEdge/Sedna 骨架说明
docs/                    # 方案与任务分配
datasets/                # MVTec 等数据软链
```

## 说明

- KubeEdge 真集群部署不阻塞本阶段指标；先用 Sedna 风格难例路由验证协同收益。
- 旧的自研 `src/models.py` PatchCore-lite 仅作兜底，**主路径以 Anomalib 为准**。
