"""Side-by-side anomaly-map visualization across edge methods."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_heatmap(amap: np.ndarray, rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay jet-like heatmap on RGB image. Both HxWx3 uint8 out."""
    a = amap.astype(np.float32)
    a = a - a.min()
    denom = a.max() + 1e-8
    a = a / denom
    # simple jet-ish colormap without matplotlib dependency
    r = np.clip(1.5 - np.abs(4 * a - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * a - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * a - 1), 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    heat_u = (heat * 255).astype(np.float32)
    base = rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * heat_u
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_rgb(img: Image.Image, hw: tuple[int, int]) -> np.ndarray:
    return np.array(img.resize((hw[1], hw[0]), Image.BICUBIC).convert("RGB"))


def _fit_map(amap: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if amap.shape == hw:
        return amap
    from .pixel_metrics import upsample_amap

    return upsample_amap(amap, hw)


def render_comparison_row(
    image_path: Path,
    gt_mask: np.ndarray,
    method_maps: dict[str, np.ndarray],
    method_order: list[str],
    cell: int = 256,
    titles: dict[str, str] | None = None,
) -> Image.Image:
    """One row: Image | GT | method1 | method2 | ..."""
    titles = titles or {}
    img = Image.open(image_path).convert("RGB")
    rgb = _resize_rgb(img, (cell, cell))
    gt = np.array(Image.fromarray((gt_mask * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST))
    gt_rgb = np.stack([gt, np.zeros_like(gt), np.zeros_like(gt)], axis=-1)

    panels = [("Image", rgb), ("GT", gt_rgb)]
    for m in method_order:
        amap = method_maps.get(m)
        if amap is None:
            blank = np.zeros_like(rgb)
            panels.append((titles.get(m, m), blank))
            continue
        amap_r = _fit_map(amap, (img.size[1], img.size[0]))
        amap_r = np.array(
            Image.fromarray((amap_r * 255 / (amap_r.max() + 1e-8)).astype(np.uint8)).resize(
                (cell, cell), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
        overlay = _to_heatmap(amap_r, rgb)
        panels.append((titles.get(m, m), overlay))

    header_h = 28
    canvas = Image.new("RGB", (cell * len(panels), cell + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, (title, arr) in enumerate(panels):
        canvas.paste(Image.fromarray(arr), (i * cell, header_h))
        draw.text((i * cell + 6, 6), title, fill=(0, 0, 0), font=font)
    return canvas


def render_category_grid(
    category: str,
    samples: list[Path],
    gt_by_path: dict[str, np.ndarray],
    maps_by_method: dict[str, dict[str, np.ndarray]],
    method_order: list[str],
    out_path: Path,
    cell: int = 256,
    titles: dict[str, str] | None = None,
) -> Path:
    rows = []
    for p in samples:
        key = str(p.resolve())
        # also try non-resolved / alternate roots
        keys = [key, str(p), str(Path(p).resolve())]
        gt = None
        for k in keys:
            if k in gt_by_path:
                gt = gt_by_path[k]
                break
        if gt is None:
            from .pixel_metrics import mvtec_gt_mask

            gt = mvtec_gt_mask(p)
        method_maps = {}
        for m in method_order:
            mm = maps_by_method.get(m, {})
            hit = None
            for k in keys:
                if k in mm:
                    hit = mm[k]
                    break
            # fuzzy: match by parent/name
            if hit is None:
                for mk, mv in mm.items():
                    if Path(mk).name == p.name and Path(mk).parent.name == p.parent.name:
                        hit = mv
                        break
            method_maps[m] = hit
        rows.append(render_comparison_row(p, gt, method_maps, method_order, cell=cell, titles=titles))

    if not rows:
        raise ValueError(f"no viz rows for {category}")
    w = rows[0].width
    h = sum(r.height for r in rows) + 36
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 8), f"{category}", fill=(0, 0, 0), font=font)
    y = 36
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
