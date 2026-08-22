# CloudEdge-abnormal

面向工业异常检测的云边协同实验平台。项目以边端快速感知、动态上云和云端多模态复核为主线，同时提供统一的数据集、评价指标和实验接口。

## 核心流程

```text
输入图像
   ├─ 边端异常检测（Qwen 视觉塔 / DINOv3 / CLIP / PaDiM）
   ├─ RouteAgent + 网络状态：判断是否上云
   └─ 云端检测 Agent
        ├─ WR50 PatchCore 异常专家
        ├─ 不确定样本路由（默认 10%）
        ├─ Qwen3.5-9B 证据复核
        └─ 保守融合 / 冲突验证
```

云端 Agent 为 Qwen 提供原图、PatchCore 热力图、候选异常区域和最近正常参考图。未路由的样本直接使用 WR50 结果，以限制大模型调用开销。

## 当前实验结果

统一协议：224×224 输入，正常训练样本建立 PatchCore memory bank，留出正常样本校准阈值，测试标签不参与阈值或融合调参。表中为类别宏平均。

| Dataset | Method | AUROC | AUPRC | F1 | Mean latency |
|---|---|---:|---:|---:|---:|
| MVTec | WR50 PatchCore | 0.9774 | 0.9921 | 0.9479 | 4.90 ms |
| MVTec | WR50 + Qwen Agent | 0.9633 | 0.9806 | 0.9333 | 630.11 ms |
| ViSA | WR50 PatchCore | 0.9804 | 0.8622 | 0.6525 | 11.24 ms |
| ViSA | WR50 + Qwen Agent | 0.9790 | 0.8692 | 0.7838 | 654.73 ms |
| Real-IAD | WR50 PatchCore | 0.8808 | 0.8460 | 0.6704 | 15.98 ms |
| Real-IAD | WR50 + Qwen Agent | 0.8783 | 0.8403 | 0.7031 | 635.80 ms |

当前 Agent 能提高 ViSA 和 Real-IAD 的固定阈值 F1，但尚未稳定提高 AUROC。因此 WR50 仍是默认主检测器，VLM 主要用于困难样本复核、证据解释和后续风险门控研究。

## 目录结构

```text
CloudEdge-abnormal/
├─ edge/                 # 边端感知与推理入口
├─ cloud/                # 云端 VLM 复核
├─ src/
│  ├─ detection_agent/  # 证据、路由、融合和冲突验证
│  ├─ collab_routing/   # 云边任务路由策略
│  └─ vlm/              # 统一 VLM 客户端
├─ scripts/              # 训练、评测、分析和冒烟测试
├─ configs/              # 数据集、模型与实验配置
├─ web/                  # FastAPI 演示与云边拓扑界面
├─ tests/                # 数据、评价、Agent 与 VLM 接口测试
└─ docs/                 # 详细方案与实验说明
```

## 快速开始

### 1. 安装

```bash
conda create -n cloudedge python=3.11 -y
conda activate cloudedge
pip install -r requirements.txt
```

### 2. 准备数据和权重

数据集与模型权重不纳入 Git。将 MVTec、ViSA 和 Real-IAD 放入 `datasets/`，将所需模型放入 `model_card/`，或修改 `configs/datasets.yaml` 及对应模型配置。

```text
datasets/{mvtec,visa,realiad}
model_card/{Qwen3.5-9B,MiniCPM-V-4_5-int4,...}
```

```bash
python scripts/check_datasets.py
```

### 3. WR50 基线

```bash
python scripts/bench_detection_branches.py \
  --config configs/mvtec_patchcore_wr50.yaml \
  --datasets mvtec --models patchcore_wr50 \
  --device cuda:0 --out outputs/mvtec_wr50
```

### 4. 云端 Agent

```bash
python scripts/bench_detection_agent.py \
  --config configs/agent_patchcore_wr50_qwen9b.yaml \
  --dataset visa --categories all --phase all \
  --device cuda:0 --out outputs/agent_visa --resume
```

`--dataset` 支持 `mvtec`、`visa` 和 `realiad`。实验按类别保存 manifest、VLM review 和最终结果，可中断续跑。

### 5. 测试

```bash
pytest -q
```

## Web 演示

```bash
python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

Web 界面展示边端节点、动态网络状态、上云路由和检测结果。模型路径和设备可通过 `configs/default.yaml` 或 `WEB_*` 环境变量配置。

## 说明

- 仓库不包含数据集、模型权重、运行日志和全量输出。
- 大模型延迟与显存数据来自 RTX 3090，仅作开发环境参考。
- 当前为研究原型，自动覆盖强专家结果前，应在独立验证集上固化门控策略。
