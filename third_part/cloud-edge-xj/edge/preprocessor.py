import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    # 当前占位边缘模型直接接收原始 uint8 像素，避免全分辨率浮点副本。
    # 接入真实模型后，可在其专用推理实现中增加所需的归一化。
    return image
