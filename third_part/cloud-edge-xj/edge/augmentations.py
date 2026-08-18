from typing import Optional

import cv2
import numpy as np


def apply_augmentation(
    image: np.ndarray,
    augmentation: str,
    parameter: Optional[float] = None,
) -> np.ndarray:
    """模拟不同边缘节点的成像差异，始终保持原始宽高。"""
    name = augmentation.strip().lower()
    if name in {"none", "identity"}:
        return image
    if name == "horizontal_flip":
        return np.ascontiguousarray(image[:, ::-1])
    if name == "brightness":
        factor = 1.05 if parameter is None else parameter
        if np.issubdtype(image.dtype, np.integer):
            return cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return np.clip(image * factor, 0.0, 1.0)
    if name == "contrast":
        factor = 1.05 if parameter is None else parameter
        if np.issubdtype(image.dtype, np.integer):
            return cv2.convertScaleAbs(
                image,
                alpha=factor,
                beta=128.0 * (1.0 - factor),
            )
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0.0, 1.0)
    if name == "gaussian_blur":
        kernel_size = 3 if parameter is None else max(1, round(parameter))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    raise ValueError(f"不支持的边缘节点数据增强：{augmentation!r}")
