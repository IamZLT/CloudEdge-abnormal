import numpy as np
from common.schemas import DetectionResult


class InferenceEngine:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        return None

    def predict(self, image: np.ndarray) -> DetectionResult:
        # 当前尚未接入真实边缘模型。使用可复现、对图像增强敏感的统计量
        # 生成模拟结果，确保多节点一致性实验可以复核，而不是由全局随机数决定。
        # 轻量模拟器在原分辨率输入上做稀疏特征提取，不创建缩放图像。
        sampled = image[::8, ::8]
        value_scale = 255.0 if np.issubdtype(sampled.dtype, np.integer) else 1.0
        mean_intensity = float(np.mean(sampled)) / value_scale
        contrast = float(np.std(sampled)) / value_scale
        anomaly_score = 0.7 * mean_intensity + 0.3 * contrast
        threshold = 0.5
        label = "anomaly" if anomaly_score < threshold else "normal"
        confidence = min(0.99, 0.55 + abs(anomaly_score - threshold) * 10.0)
        return DetectionResult(
            label=label,
            confidence=confidence,
            defect_category=None if label == "normal" else "unknown",
            bbox=(
                None
                if label == "normal"
                else {
                    "x1": 0,
                    "y1": 0,
                    "x2": image.shape[1] - 1,
                    "y2": image.shape[0] - 1,
                }
            ),
            timestamp=None,
            source="edge",
            metadata={
                "model": "deterministic-placeholder",
                "simulation": True,
                "anomaly_score": anomaly_score,
            },
        )
