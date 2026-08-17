# 云端工业缺陷检测（Qwen3.5 + DINOv3）

本目录实现一个训练无关的云端工业缺陷检测器：DINOv3 ViT-L/16 从正常训练图构建 patch 级记忆库，输出精细异常图；冻结的 Qwen3.5 使用正常参考图对测试图做多模态语义复核并给出缺陷区域。两路结果采用保守融合，Qwen 只增强可疑区域，不会压低 DINOv3 对微小缺陷的响应。

## 关键保证

- 默认模型：`/data/cx/models/Qwen/Qwen3.5-2B`。
- `--use-large` 切换到 `/data/cx/models/Qwen/Qwen3.5-9B`。
- Qwen 和 DINOv3 均执行 `eval()`、`requires_grad_(False)` 和 `torch.inference_mode()`；代码没有优化器和反向传播，多模态大模型参数不会被修改。
- DINOv3 支持 Hugging Face 本地目录，也支持目录内官方 `.pth` 权重；后者直接调用当前目录中的 `dinov3` 源码。
- 输出 Image-level 与 Pixel-level 的 AUROC、AP、F1-max，以及每个数据集和类别的单样本平均处理时间。

## 环境

建议 Python 3.10+、Linux、CUDA GPU。依赖：

```bash
pip install -r requirements.txt
```

默认路径均在 `configs/default.yaml` 中，可复制配置后修改。DINOv3 权重目录若是 Hugging Face 格式，应包含 `config.json` 和模型权重；若是官方 PyTorch 格式，目录内应有对应的 `.pth` 文件。

## 运行

先为每个类别使用正常训练图构建记忆库：

```bash
python -m cloud_abnormal --config configs/default.yaml fit --dataset mvtec
python -m cloud_abnormal --config configs/default.yaml fit --dataset visa
```

再运行默认 2B 融合评估：

```bash
python -m cloud_abnormal --config configs/default.yaml evaluate --dataset mvtec
python -m cloud_abnormal --config configs/default.yaml evaluate --dataset visa
```

9B 和 DINO-only 消融：

```bash
python -m cloud_abnormal --config configs/default.yaml evaluate --dataset mvtec --use-large
python -m cloud_abnormal --config configs/default.yaml evaluate --dataset mvtec --disable-qwen
```

结果写入 `outputs/*_metrics.json`。其中 `overall` 同时给出全样本汇总结果与 `macro_average` 类别宏平均，`overall.mean_time_seconds` 是该数据集单样本平均处理时间；Qwen 推理缓存位于 `outputs/qwen_cache`，重复评估会命中缓存。正式报告时应注明首次推理时延与缓存重跑时延不同，公平对比使用首次推理结果。

## 方法说明与调参

1. DINOv3 从第 6/12/18/24 层提取多层 patch 特征，拼接归一化后用正常样本记忆库做 kNN 距离。
2. 留出部分正常训练图估计距离中位数与高分位尺度，把不同类别的分数校准到 `[0,1]`。
3. Qwen 同时观察最多 3 张正常参考图和测试图，输出异常概率、缺陷类型、原因及 0-1000 坐标区域。
4. Pixel 分数以 DINO 为主，仅在 Qwen 建议区域增加语义证据；Image 分数对 DINO/Qwen 做默认 75%/25% 融合，但取不低于原始 DINO 分数的结果，因此语义分支不会压低强视觉异常。

若显存不足，优先把 `model.image_size` 改为 336、减少 `qwen.normal_references`，或使用 2B。若 Qwen 在某类纹理上误报，降低 `fusion.qwen_image_weight` 与 `fusion.qwen_pixel_weight`。最终权重应只在独立验证集上选择，不要使用测试标签调参。

## 数据布局

MVTec 使用标准 `category/train/good`、`category/test/*`、`category/ground_truth/*`。VisA 同时支持官方转换后的 `visa_pytorch/1cls` 布局和 `split_csv/1cls.csv` 布局；发现异常测试图缺少 mask 时会直接报错，防止静默产生错误 Pixel-level 指标。
