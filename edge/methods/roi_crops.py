"""Anomaly-map → pixel ROI crops (new helper; does not alter PatchGalleryAD)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class RoiCrop:
    """One ROI in original image coordinates (xyxy, inclusive-exclusive)."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    score: float
    patch_ij: tuple[int, int]  # (row, col) on anomaly grid
    crop: Image.Image


def amap_to_rois(
    image: Image.Image,
    amap: np.ndarray,
    *,
    top_k: int = 2,
    pad_ratio: float = 0.35,
    min_side: int = 64,
    score_floor: float | None = None,
) -> list[RoiCrop]:
    """Pick top-k anomalous grid cells and crop dilated boxes from ``image``.

    ``amap`` is [H,W] aligned with the encoder patch grid (not upsampled).
    Boxes are mapped linearly onto the original PIL size.
    """
    if amap.ndim != 2:
        raise ValueError(f"amap must be HxW, got {amap.shape}")
    gh, gw = amap.shape
    w, h = image.size
    flat = amap.reshape(-1)
    order = np.argsort(-flat)
    rois: list[RoiCrop] = []
    used = np.zeros_like(amap, dtype=bool)

    for idx in order:
        if len(rois) >= top_k:
            break
        score = float(flat[idx])
        if score_floor is not None and score < score_floor:
            break
        i, j = divmod(int(idx), gw)
        if used[i, j]:
            continue
        # suppress immediate neighbors so top-k are spatially diverse
        i0, i1 = max(0, i - 1), min(gh, i + 2)
        j0, j1 = max(0, j - 1), min(gw, j + 2)
        used[i0:i1, j0:j1] = True

        # cell → pixel, then dilate
        x0 = int(j * w / gw)
        x1 = int((j + 1) * w / gw)
        y0 = int(i * h / gh)
        y1 = int((i + 1) * h / gh)
        cw, ch = max(1, x1 - x0), max(1, y1 - y0)
        pad_x = int(cw * pad_ratio) + max(0, (min_side - cw) // 2)
        pad_y = int(ch * pad_ratio) + max(0, (min_side - ch) // 2)
        bx0 = max(0, x0 - pad_x)
        by0 = max(0, y0 - pad_y)
        bx1 = min(w, x1 + pad_x)
        by1 = min(h, y1 + pad_y)
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        crop = image.crop((bx0, by0, bx1, by1))
        rois.append(
            RoiCrop(
                box=(bx0, by0, bx1, by1),
                score=score,
                patch_ij=(i, j),
                crop=crop,
            )
        )
    return rois


def make_roi_collage(rois: list[RoiCrop], *, gap: int = 4, bg=(20, 20, 20)) -> Image.Image | None:
    """Horizontal collage of ROI crops (for single-image VLM input)."""
    if not rois:
        return None
    imgs = [r.crop.convert("RGB") for r in rois]
    # normalize height
    th = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != th:
            nw = max(1, int(im.width * th / im.height))
            im = im.resize((nw, th), Image.Resampling.BICUBIC)
        resized.append(im)
    tw = sum(im.width for im in resized) + gap * (len(resized) - 1)
    out = Image.new("RGB", (tw, th), bg)
    x = 0
    for im in resized:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def make_multiscale_collage(
    image: Image.Image,
    rois: list[RoiCrop],
    *,
    full_height: int = 224,
    gap: int = 4,
    bg=(20, 20, 20),
) -> Image.Image | None:
    """Full downscaled image (LEFT) + top-k ROI crops (RIGHT), horizontally tiled.

    The left panel preserves global appearance (color, texture, layout) that a
    single local crop would discard; the right panels magnify fine defects.
    """
    if not rois:
        return None
    full = image.convert("RGB").copy()
    fw, fh = full.size
    full = full.resize((max(1, int(fw * full_height / fh)), full_height), Image.Resampling.BICUBIC)

    panels = [full]
    for r in rois:
        p = r.crop.convert("RGB")
        if p.height != full_height:
            nw = max(1, int(p.width * full_height / p.height))
            p = p.resize((nw, full_height), Image.Resampling.BICUBIC)
        panels.append(p)

    tw = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (tw, full_height), bg)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def draw_roi_boxes(image: Image.Image, rois: list[RoiCrop], color=(255, 64, 64)) -> Image.Image:
    """Debug overlay of ROI boxes on a copy of the image."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    for r in rois:
        draw.rectangle(r.box, outline=color, width=3)
    return vis
