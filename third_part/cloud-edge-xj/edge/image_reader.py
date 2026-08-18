from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def read_image(path: str) -> Optional[np.ndarray]:
    if cv2 is None:
        raise ImportError("opencv-python is required for image reading")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from path: {path}")
    return image
