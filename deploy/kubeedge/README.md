# KubeEdge / Sedna 部署说明（本阶段为骨架）

首选方案中的编排层：

- **KubeEdge**：边侧推理 Pod 下发、断网自治
- **Sedna JointInference 范式**：难例上云（本仓库用 `scripts/bench_anomalib.py` 先验证算法收益）
- **MLflow**：训练指标与模型元数据（`outputs/mlruns`）
- **CVAT**：难例回流标注（后续接入）

## 建议落地顺序

1. 本机跑通 Anomalib 训练/评测与 OpenVINO/ONNX 导出
2. 边侧容器化推理服务（OpenVINO Runtime）
3. 云端复核服务（PatchCore / 更大模型）
4. KubeEdge 1 云 + N 边（与 `collab.edge_fleet.num_nodes` 对齐，默认 3）+ 模型镜像下发
5. 可选接入 Sedna `JointInferenceService` CRD

逻辑多边已在应用层落地（`src/edge_fleet.py` + Web `/api/edge_nodes`）；K8s YAML 可按节点环境再补。
