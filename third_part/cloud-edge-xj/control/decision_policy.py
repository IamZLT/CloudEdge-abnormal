from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional

import numpy as np

from common.config import Config
from common.image_quality import calculate_image_quality_score
from common.schemas import DetectionResult, UploadDecision


class DecisionPolicy:
    def __init__(self, config: Config):
        self.config = config
        self.quality_score_threshold = config.quality_score_threshold
        self._quality_scores_by_path: Dict[str, Dict[str, object]] = {}
        self.last_calibration_errors = []

    def calibrate_quality_threshold(self, image_paths: Iterable[str]) -> Optional[float]:
        if self.config.quality_cloud_ratio is None:
            return None
        ratio = float(self.config.quality_cloud_ratio)
        if ratio <= 0:
            self.quality_score_threshold = float("inf")
            return self.quality_score_threshold
        if ratio >= 1:
            self.quality_score_threshold = float("-inf")
            return self.quality_score_threshold

        paths = list(image_paths)
        paths = self._sample_calibration_paths(paths)
        scores = []
        self.last_calibration_errors = []
        workers = max(1, int(self.config.quality_calibration_workers))
        if workers == 1:
            scored_items = [self._score_calibration_path(path) for path in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                scored_items = list(executor.map(self._score_calibration_path, paths))
        for image_path, quality_dict, error in scored_items:
            if error is not None:
                self.last_calibration_errors.append(
                    {"image_path": image_path, "error": error}
                )
                continue
            self._quality_scores_by_path[str(image_path)] = quality_dict
            scores.append(float(quality_dict["score"]))
        if not scores:
            return None
        self.quality_score_threshold = float(np.quantile(scores, 1.0 - ratio))
        return self.quality_score_threshold

    def _sample_calibration_paths(self, paths):
        max_images = self.config.quality_calibration_max_images
        if max_images is None or max_images <= 0 or max_images >= len(paths):
            return paths
        if max_images == 1:
            return [paths[len(paths) // 2]]
        return [
            paths[round(index * (len(paths) - 1) / (max_images - 1))]
            for index in range(max_images)
        ]

    def _score_calibration_path(self, image_path):
        try:
            quality = calculate_image_quality_score(
                image_path=image_path,
                score_size=self.config.quality_score_size,
                jpeg_quality=self.config.quality_jpeg_quality,
                weights=self.config.quality_weights,
            )
        except Exception as exc:
            return image_path, {}, str(exc)
        return image_path, quality.to_dict(), None

    def calibration_sample_count(self, image_paths: Iterable[str]) -> int:
        return len(self._sample_calibration_paths(list(image_paths)))

    def calibration_cache_count(self) -> int:
        return len(self._quality_scores_by_path)

    def attach_quality_score(
        self,
        result: DetectionResult,
        image,
        image_path: Optional[str] = None,
    ) -> DetectionResult:
        quality_dict = (
            self._quality_scores_by_path.get(str(image_path)) if image_path else None
        )
        if quality_dict is None:
            quality = calculate_image_quality_score(
                image=image,
                image_path=image_path,
                score_size=self.config.quality_score_size,
                jpeg_quality=self.config.quality_jpeg_quality,
                weights=self.config.quality_weights,
            )
            quality_dict = quality.to_dict()
        metadata = dict(result.metadata or {})
        metadata["image_quality"] = quality_dict
        result.metadata = metadata
        return result

    def should_upload(self, result: DetectionResult) -> UploadDecision:
        metadata = result.metadata or {}
        quality = metadata.get("image_quality") or {}
        score = quality.get("score")
        if score is None:
            raise ValueError(
                "image_quality score is required before image-quality routing"
            )
        score = float(score)
        should_upload = score >= self.quality_score_threshold
        return UploadDecision(
            should_upload=should_upload,
            reason="high_image_complexity" if should_upload else "low_image_complexity",
            threshold=self.quality_score_threshold,
            score=score,
            policy="image_quality",
        )
