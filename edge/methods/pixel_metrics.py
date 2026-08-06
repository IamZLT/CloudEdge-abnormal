"""Pixel-level AD metrics and MVTec GT helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def mvtec_gt_mask(image_path: Path, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load binary GT mask for an MVTec test image. Good samples → all zeros."""
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        w0, h0 = im.size

    defect = image_path.parent.name
    if defect == "good":
        mask = np.zeros((h0, w0), dtype=np.uint8)
    else:
        gt = image_path.parents[2] / "ground_truth" / defect / f"{image_path.stem}_mask.png"
        if not gt.exists():
            alt = image_path.parents[2] / "ground_truth" / defect / image_path.name
            gt = alt if alt.exists() else gt
        if not gt.exists():
            raise FileNotFoundError(f"GT mask missing for {image_path}: tried {gt}")
        mask = np.array(Image.open(gt).convert("L"))
        mask = (mask > 0).astype(np.uint8)

    if target_hw is not None and mask.shape != target_hw:
        mask = np.array(
            Image.fromarray(mask * 255).resize((target_hw[1], target_hw[0]), Image.NEAREST)
        )
        mask = (mask > 0).astype(np.uint8)
    return mask


def upsample_amap(amap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample HxW anomaly map to (H, W)."""
    if amap.shape == target_hw:
        return amap.astype(np.float32)
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(amap, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def pixel_auroc(
    gt_masks: list[np.ndarray],
    amaps: list[np.ndarray],
    eval_hw: tuple[int, int] = (256, 256),
) -> float:
    ys, ss = [], []
    for g, a in zip(gt_masks, amaps):
        g2 = g
        if g2.shape != eval_hw:
            g2 = np.array(
                Image.fromarray((g2 * 255).astype(np.uint8)).resize(
                    (eval_hw[1], eval_hw[0]), Image.NEAREST
                )
            )
            g2 = (g2 > 0).astype(np.uint8)
        a2 = a if a.shape == eval_hw else upsample_amap(a, eval_hw)
        ys.append(g2.reshape(-1).astype(np.uint8))
        ss.append(a2.reshape(-1).astype(np.float32))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def best_pixel_f1(
    gt_masks: list[np.ndarray],
    amaps: list[np.ndarray],
    n_thr: int = 49,
    eval_hw: tuple[int, int] = (256, 256),
) -> tuple[float, float, float, float]:
    """Optimistic best F1 over thresholds on the same split."""
    ys, ss = [], []
    for g, a in zip(gt_masks, amaps):
        g2 = g
        if g2.shape != eval_hw:
            g2 = np.array(
                Image.fromarray((g2 * 255).astype(np.uint8)).resize(
                    (eval_hw[1], eval_hw[0]), Image.NEAREST
                )
            )
            g2 = (g2 > 0).astype(np.uint8)
        a2 = a if a.shape == eval_hw else upsample_amap(a, eval_hw)
        ys.append(g2.reshape(-1).astype(np.uint8))
        ss.append(a2.reshape(-1).astype(np.float32))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.min() == y.max():
        return float("nan"), 0.0, 0.0, 0.0
    best = (-1.0, 0.0, 0.0, float(np.median(s)))
    # subsample for threshold search speed
    if y.size > 2_000_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(y.size, size=2_000_000, replace=False)
        y_s, s_s = y[idx], s[idx]
    else:
        y_s, s_s = y, s
    for t in np.quantile(s_s, np.linspace(0.02, 0.98, n_thr)):
        pred = (s_s >= t).astype(np.uint8)
        f1 = float(f1_score(y_s, pred, zero_division=0))
        if f1 > best[0]:
            best = (
                f1,
                float(precision_score(y_s, pred, zero_division=0)),
                float(recall_score(y_s, pred, zero_division=0)),
                float(t),
            )
    return best


def pick_viz_samples(
    data_root: Path,
    category: str,
    n: int = 3,
    seed: int = 42,
) -> list[Path]:
    """Pick up to n defect images, diversifying defect subtypes; prefer larger masks."""
    from collections import defaultdict

    from .gallery_ad import mvtec_test_split

    rng = np.random.default_rng(seed)
    by_type: dict[str, list[Path]] = defaultdict(list)
    for path, y in mvtec_test_split(data_root, category):
        if y == 1:
            by_type[path.parent.name].append(path)

    ranked: dict[str, list[Path]] = {}
    for t, paths in by_type.items():
        areas = []
        for p in paths:
            try:
                m = mvtec_gt_mask(p)
                areas.append((float(m.mean()), p))
            except Exception:
                areas.append((0.0, p))
        areas.sort(key=lambda x: -x[0])
        ranked[t] = [p for _, p in areas]

    types = sorted(ranked.keys())
    rng.shuffle(types)
    picked: list[Path] = []
    i = 0
    while len(picked) < n and types:
        progressed = False
        for t in types:
            if i < len(ranked[t]):
                picked.append(ranked[t][i])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        i += 1
    return picked
