"""Edge-side anomaly detection method zoo.

Default production path: Qwen3.5-0.8B multi-layer pre-merger patch gallery
(`configs/edge_qwen35.yaml`, `python -m edge.infer`).
"""

from .gallery_ad import FeatureGalleryAD, MethodResult, best_f1_threshold
from .patch_gallery_ad import PatchGalleryAD

__all__ = ["FeatureGalleryAD", "PatchGalleryAD", "MethodResult", "best_f1_threshold"]
