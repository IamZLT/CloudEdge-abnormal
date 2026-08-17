# 云边协同工业缺陷检测系统

## 目录结构

- `edge/`：边缘节点读取、预处理、轻量推理
- `control/`：协同控制层，决定是否上传云端
- `cloud/`：边缘侧 HTTP 客户端、云端网关以及网关内部的大模型 API 调用
- `common/`：通用数据结构、配置、可视化与评测指标

## 运行方式

1. 创建目录结构
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
3. 在 `config.yaml` 中统一配置设备、边缘模型、云端网关、云端模型、质量路由策略和数据集根目录。
4. 在云端主机启动网关（必须从项目根目录使用模块方式启动）：

   ```bash
   /data2/dxj/envs/torch/bin/python -m cloud.gateway_server --config config.yaml
   ```

5. 在边缘主机运行数据集任务：

   ```bash
   /data2/dxj/envs/torch/bin/python main.py
   ```

调用链为 `边缘 main.py → HTTP 云端网关 → 云端本机视觉大模型 API`。边缘主机读取
`cloud.gateway_url`，该值必须是边缘可达的云端内网地址，例如
`http://192.168.1.20:7790`；云端的 `gateway_host: 0.0.0.0` 允许网关监听所有网卡。
`cloud.model_api_base_url: http://127.0.0.1:7788/v1` 只由云端网关读取，要求模型
API 与网关运行在同一台云端主机。跨主机部署时还需在防火墙/安全组放通 TCP 7790。

## 说明

- 边缘层当前使用可复现的确定性占位推理输出 `DetectionResult`；未配置真实模型时，稳定性与一致性结果属于仿真代理指标
- 协同控制层默认根据 SAEC 风格图像质量/复杂度分数决定是否上传云端；高复杂度图像上云复核，简单图像留在边缘
- 边缘到网关使用 `multipart/form-data` 上传原始文件字节和版本化 metadata；默认 metadata
  只含 run/request/device 标识及可选 context，不上传边缘检测结果
- 网关将原始图像和用户提示发送给视觉大模型，不把边缘结果写入 prompt；这一内部调用
  使用 Base64 data URL，但不计入边缘到云端的数据上传量
- `edge.target_size: null` 表示边缘预处理保留原始分辨率，边缘上传原始图片字节
- MVTec 自动读取各类别的 `test/` 图像，并排除 `ground_truth/`
- VisA 自动读取各类别的 `Data/Images/Normal` 和 `Data/Images/Anomaly`，并排除 `Masks/`
- `categories: []` 表示处理全部类别；可填写类别名称进行筛选
- `max_images: null` 表示全量处理；联调时可改为一个正整数
- 检测结果逐条写入 `outputs/results.jsonl`；成功与失败记录均会保存
- `output_append: false` 表示每次运行覆盖结果文件，改为 `true` 可追加写入

## 图像质量路由

边缘端会对原图的轻量评分副本计算五项无监督指标：灰度熵、边缘密度、Laplacian 方差、Sobel 梯度均值和 JPEG 压缩残差，并按 `policy.quality_weights` 加权得到 `image_quality.score`。

当 `image_quality.score >= policy.quality_score_threshold` 时，样本被认为复杂并上传云端。若配置了 `policy.quality_cloud_ratio`，程序会在每次运行开始时根据本次输入数据集自动校准阈值，例如 `0.3` 表示约 30% 高复杂度图像上云；设为 `null` 则使用固定阈值。`policy.quality_calibration_max_images` 控制阈值校准的均匀抽样数量，默认 1000，避免全量数据集启动阶段长时间无输出。

## 可视化输出

- 正常图像按原始字节复制到 `outputs/visualizations/normal/<数据集名称>/<数据集相对路径>`
- 异常图像在原始分辨率上绘制红色 bbox，并保存到对应的 `anomaly/` 目录
- `results.jsonl` 中的 `visualization_path` 指向生成文件；可视化失败单独记录，不会把已完成的检测误算为推理失败

## 云边评测指标

评测口径来自项目 PDF 第 4—7 页，运行 `main.py` 后生成：

- `outputs/metrics/events.jsonl`：每张图像的端到端时延、上传信息、节点结果和冲突解决信息
- `outputs/metrics/summary.json`：汇总指标与 PDF 目标的达标情况
- `outputs/metrics/report.md`：便于直接审阅的指标报告

端到端时延从读取单张图像前开始，到最终 `DetectionResult` 就绪结束，不包含可视化与文件写入。资源与通信效率采用三个主指标：边缘到网关的图像检测 HTTP 请求体上传量、完整评测窗口平均上行吞吐相对配置链路容量的带宽占用率，以及云端网关完整评测窗口内的 GPU 利用率。`evaluation.link_capacity_mbps` 是人工配置的链路容量，不是网络设备实测值；上传量不含 HTTP 头与 TCP/IP 开销，也不含 start/finish 控制请求。GPU 来自 `nvidia-smi` 的整卡主机级指标，同卡其他进程会影响结果；只有当 `cloud.gpu_index` 对云端网关主机可见时才能采集。

启用 `evaluation.multi_edge.enabled` 后，同一原图会由多个模拟边缘节点并行处理。各节点使用保持原分辨率的数据增强模拟节点差异；标签不完全一致即计为冲突，冲突图像由云端原图仲裁，失败时回退到边缘多数决策。冲突解决成功率表示云端仲裁成功返回有效结果的比例；MVTec 的 `good/其他缺陷目录` 和 VisA 的 `Normal/Anomaly` 路径用于另行计算最终决策的真值正确率。

如需先验证评测链路而不运行全部数据，可执行：

```bash
/data2/dxj/envs/torch/bin/python evaluate.py --samples-per-dataset 100 --output-dir outputs/metrics/smoke
```

正常模式的 `evaluate.py` 同样要求云端网关已经启动；`--simulate-outage` 模式不会访问
网关。抽样报告会在内部标记为 smoke，不应替代 `main.py` 对完整数据集生成的正式报告。

## 断网降级与业务保持率

边缘端采用本地优先的容错策略：本地推理结果始终先存在；启动注册或云复核第一次失败
后打开断路器，后续需要云复核的任务不再访问网络，而是立即返回边缘结果。冷却时间结束
后只允许一个半开请求探测恢复；如果网关已重启，客户端会自动重新注册 `run_id`。

高复杂度或多节点冲突在断网时默认执行 fail-safe：输出 `anomaly`，并在
`metadata.resilience` 中标记 `provisional: true`、`requires_cloud_review: true` 和
`business_action: divert_for_review`，把工件送入隔离/人工复检，而不是不确定地放行。
该行为可通过 `resilience.fail_safe_uncertain_as_anomaly` 配置。

断网限时业务保持率定义为：

```text
在业务时限内产生合法 normal/anomaly 决策，且随后同步写入本地事件成功的任务数
÷ 故障窗口内总任务数 × 100%
```

分母包含失败和迟到任务；本地写入必须成功，但写入耗时本身不计入 200 ms 决策时限。
默认业务时限为 200 ms、目标为 90%。运行确定性的 100% 断网
压力评测时会强制每个样本请求云复核，避免因原本云请求比例过低而虚高：

```bash
/data2/dxj/envs/torch/bin/python evaluate.py \
  --samples-per-dataset 100 \
  --simulate-outage \
  --output-dir outputs/metrics/offline_smoke
```

报告同时给出整体保持率、需云任务子集保持率、本地降级次数、断路器快速跳过次数和断网
决策真值正确率。业务保持率衡量软件链路连续性，不等于检测准确率；当前占位边缘模型不能
用于证明工业检测质量。正式产线还应增加 SQLite/WAL 持久补传队列、图像 spool、PLC 动作
确认和网络恢复后的限速重放，当前版本尚未将这些外部系统确认纳入保持率。

## 跨主机部署检查

- 云端模型 API：在云端确认 `curl http://127.0.0.1:7788/v1/models` 可访问
- 云端网关：启动后确认 `curl http://127.0.0.1:7790/health` 返回 `status: ok`
- 边缘连通性：在边缘确认 `curl http://<云端内网IP>:7790/health` 可访问
- 数据集：`datasets.*.root` 是边缘主机本地路径；云端不需要挂载数据集，因为图像经 HTTP 上传
- 依赖：边缘和云端均需项目代码及 Python 依赖；云端额外需要能访问本机模型服务和 `nvidia-smi`
