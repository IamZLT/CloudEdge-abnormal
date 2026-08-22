from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .schemas import ExpertEvidence, RegionEvidence


@dataclass
class EvidenceBoard:
    image: Image.Image
    regions: list[RegionEvidence]
    reference_path: str


def patch_concentration(scores: np.ndarray, fraction: float = 0.10) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values - float(values.min(initial=0.0))
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    keep = max(1, int(round(len(values) * fraction)))
    return float(np.sort(values)[-keep:].sum() / total)


def top_regions(
    scores: np.ndarray,
    image_size: tuple[int, int],
    top_k: int = 2,
    context_cells: float = 2.5,
) -> list[RegionEvidence]:
    grid = np.asarray(scores, dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"patch scores must be HxW, got {grid.shape}")
    gh, gw = grid.shape
    width, height = image_size
    selected: list[tuple[int, int]] = []
    for flat in np.argsort(grid.reshape(-1))[::-1]:
        row, col = divmod(int(flat), gw)
        if any(abs(row - old_r) <= 1 and abs(col - old_c) <= 1 for old_r, old_c in selected):
            continue
        selected.append((row, col))
        if len(selected) >= top_k:
            break
    regions = []
    cell_w, cell_h = width / gw, height / gh
    crop_w, crop_h = cell_w * context_cells, cell_h * context_cells
    for row, col in selected:
        cx, cy = (col + 0.5) * cell_w, (row + 0.5) * cell_h
        left = max(0, int(round(cx - crop_w / 2)))
        top = max(0, int(round(cy - crop_h / 2)))
        right = min(width, int(round(cx + crop_w / 2)))
        bottom = min(height, int(round(cy + crop_h / 2)))
        regions.append(
            RegionEvidence(
                bbox_xyxy=(left, top, right, bottom),
                score=float(grid[row, col]),
                grid_rc=(row, col),
            )
        )
    return regions


def _heatmap(scores: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(scores, dtype=float)
    low, high = float(values.min()), float(values.max())
    normalized = np.zeros_like(values) if high <= low else (values - low) / (high - low)
    red = (255 * normalized).astype(np.uint8)
    green = (180 * np.sqrt(normalized)).astype(np.uint8)
    blue = (35 * (1.0 - normalized)).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _panel(image: Image.Image, label: str, size: int) -> Image.Image:
    image = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill="black")
    return canvas


def build_evidence_board(
    evidence: ExpertEvidence,
    panel_size: int = 336,
) -> EvidenceBoard:
    with Image.open(evidence.image_path) as source:
        original = source.convert("RGB")
    regions = evidence.regions or top_regions(evidence.patch_scores, original.size)
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    for index, region in enumerate(regions, start=1):
        draw.rectangle(region.bbox_xyxy, outline=(255, 30, 30), width=max(2, original.width // 180))
        draw.text((region.bbox_xyxy[0] + 3, region.bbox_xyxy[1] + 3), str(index), fill=(255, 30, 30))

    heat = _heatmap(evidence.patch_scores, original.size)
    overlay = Image.blend(original, heat, alpha=0.45)
    crop = original.crop(regions[0].bbox_xyxy) if regions else original
    reference_path = evidence.reference_path
    if reference_path and Path(reference_path).is_file():
        with Image.open(reference_path) as source:
            reference = source.convert("RGB")
    else:
        reference = Image.new("RGB", original.size, (220, 220, 220))
        ImageDraw.Draw(reference).text((10, 10), "normal reference unavailable", fill="black")

    panels = [
        _panel(annotated, "A. Query + PatchCore region", panel_size),
        _panel(overlay, "B. PatchCore anomaly heatmap", panel_size),
        _panel(crop, "C. Highest-score local crop", panel_size),
        _panel(reference, "D. Nearest normal reference", panel_size),
    ]
    board = Image.new("RGB", (panel_size * 2, (panel_size + 28) * 2), "white")
    board.paste(panels[0], (0, 0))
    board.paste(panels[1], (panel_size, 0))
    board.paste(panels[2], (0, panel_size + 28))
    board.paste(panels[3], (panel_size, panel_size + 28))
    return EvidenceBoard(board, regions, reference_path)
