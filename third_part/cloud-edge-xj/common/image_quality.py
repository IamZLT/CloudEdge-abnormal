from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "entropy": 0.30,
    "edge_density": 0.25,
    "laplacian": 0.20,
    "sobel": 0.15,
    "jpeg_residual": 0.10,
}


@dataclass(frozen=True)
class ImageQualityScore:
    score: float
    entropy: float
    edge_density: float
    laplacian: float
    sobel: float
    jpeg_residual: float
    score_size: int
    jpeg_quality: int
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for image quality routing")


def _resize_for_scoring(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == size and width == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _read_image(path: str) -> np.ndarray:
    _require_cv2()
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from path: {path}")
    return image


def calculate_image_quality_score(
    image: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    score_size: int = 192,
    jpeg_quality: int = 30,
    weights: Optional[Dict[str, float]] = None,
) -> ImageQualityScore:
    """SAEC-style scene complexity score used as an image-quality router.

    The source image is never modified. Only a small temporary copy is resized
    for lightweight scoring, following the SAEC complexity-score recipe.
    """

    _require_cv2()
    if image is None:
        if image_path is None:
            raise ValueError("image or image_path is required")
        image = _read_image(image_path)
    if image is None:
        raise ValueError("image is required")

    score_size = int(score_size)
    if score_size <= 0:
        raise ValueError("score_size must be positive")
    jpeg_quality = max(1, min(100, int(jpeg_quality)))
    score_weights = dict(DEFAULT_QUALITY_WEIGHTS)
    if weights:
        score_weights.update({key: float(value) for key, value in weights.items()})

    resized = _resize_for_scoring(image, score_size)
    if resized.dtype != np.uint8:
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    entropy = float(
        -np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(256)
    )

    edges = cv2.Canny(gray, 64, 128)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_norm = float(np.log1p(np.var(laplacian)) / 8.0)

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_norm = float(np.mean(np.hypot(gradient_x, gradient_y)) / 16.0)

    encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        jpeg_residual = 0.0
    else:
        decoded_gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        jpeg_residual = float(
            np.mean(cv2.absdiff(gray, decoded_gray).astype(np.float32)) / 255.0
        )

    score = (
        score_weights["entropy"] * entropy
        + score_weights["edge_density"] * edge_density
        + score_weights["laplacian"] * laplacian_norm
        + score_weights["sobel"] * sobel_norm
        + score_weights["jpeg_residual"] * jpeg_residual
    )

    return ImageQualityScore(
        score=float(score),
        entropy=entropy,
        edge_density=edge_density,
        laplacian=laplacian_norm,
        sobel=sobel_norm,
        jpeg_residual=jpeg_residual,
        score_size=score_size,
        jpeg_quality=jpeg_quality,
        weights=score_weights,
    )
