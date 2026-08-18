import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2

from common.schemas import DetectionResult


class ResultVisualizer:
    def __init__(self, output_dir: str):
        path = Path(output_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        self.output_dir = path

    def _output_path(
        self,
        dataset_name: str,
        dataset_root: str,
        image_path: str,
        label: str,
    ) -> Path:
        source = Path(image_path).resolve()
        root = Path(dataset_root).resolve()
        try:
            relative_path = source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"图像不在数据集根目录内：{source}，数据集根目录：{root}") from exc

        destination = self.output_dir / label / dataset_name / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def _bbox_coordinates(bbox: Dict[str, int], width: int, height: int) -> Tuple[int, int, int, int]:
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (round(float(bbox[key])) for key in ("x1", "y1", "x2", "y2"))
        elif all(key in bbox for key in ("x", "y", "w", "h")):
            x1 = round(float(bbox["x"]))
            y1 = round(float(bbox["y"]))
            x2 = x1 + round(float(bbox["w"]))
            y2 = y1 + round(float(bbox["h"]))
        else:
            raise ValueError("anomaly 结果的 bbox 必须包含 x1/y1/x2/y2")

        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"无效的 anomaly bbox：{bbox}")
        return x1, y1, x2, y2

    def save(
        self,
        dataset_name: str,
        dataset_root: str,
        image_path: str,
        result: DetectionResult,
    ) -> str:
        label = result.label.strip().lower()
        if label not in {"normal", "anomaly"}:
            raise ValueError(f"不支持可视化 label：{result.label!r}")

        destination = self._output_path(dataset_name, dataset_root, image_path, label)
        if label == "normal":
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                suffix=destination.suffix,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copyfile(image_path, temporary_path)
                os.chmod(temporary_path, 0o644)
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            return str(destination)

        if result.bbox is None:
            raise ValueError("anomaly 结果缺少 bbox，无法绘制异常区域")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取待可视化图像：{image_path}")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._bbox_coordinates(result.bbox, width, height)
        thickness = max(2, round(min(width, height) / 300))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=destination.suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            if not cv2.imwrite(str(temporary_path), image):
                raise OSError(f"可视化图像保存失败：{destination}")
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(destination)
