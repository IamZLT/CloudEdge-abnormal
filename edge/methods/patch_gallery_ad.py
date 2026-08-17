"""Multi-layer patch-token feature gallery with softmax-weighted fusion."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from .encoders import EncodePatchesFn, PatchTokens
from .gallery_ad import MethodResult, best_f1_threshold
from .pixel_metrics import best_pixel_f1, mvtec_gt_mask, pixel_auroc, upsample_amap


def _l2_normalize(tok: torch.Tensor) -> torch.Tensor:
    return tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)


def _nn_distance(tok: torch.Tensor, gallery: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """tok [N,D], gallery [G,D] → distance [N] = 1 - max cosine."""
    max_sim = None
    for i in range(0, gallery.shape[0], chunk):
        sims = tok @ gallery[i : i + chunk].T
        m = sims.max(dim=1).values
        max_sim = m if max_sim is None else torch.maximum(max_sim, m)
    return 1.0 - max_sim


def softmax_fuse_maps(maps: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """maps: [L,H,W] → fused [H,W] with softmax weights over layers.

    Larger per-layer distance → larger weight (emphasize layers that fire).
    """
    # stabilize
    x = maps / max(float(temperature), 1e-6)
    x = x - x.max(axis=0, keepdims=True)
    w = np.exp(x)
    w = w / (w.sum(axis=0, keepdims=True) + 1e-8)
    return (w * maps).sum(axis=0).astype(np.float32)


class PatchGalleryAD:
    """Per-layer NN distance maps → softmax-weighted anomaly map."""

    def __init__(
        self,
        encode_patches: EncodePatchesFn,
        device: str = "cuda:0",
        name: str = "patch_gallery",
        max_gallery_patches: int | None = 50000,
        fusion_temperature: float = 0.5,
    ):
        self.encode_patches = encode_patches
        self.device = device
        self.name = name
        self.max_gallery_patches = max_gallery_patches
        self.fusion_temperature = fusion_temperature
        # one gallery bank per layer
        self.galleries: list[torch.Tensor] | None = None
        self.layer_ids: list[int] = []

    @torch.inference_mode()
    def build_gallery(self, paths: list[Path], seed: int = 42) -> float:
        t0 = time.perf_counter()
        pts = [self.encode_patches(Image.open(p).convert("RGB")) for p in paths]
        self.build_gallery_from_tokens(pts, seed=seed)
        return time.perf_counter() - t0

    @torch.inference_mode()
    def build_gallery_from_tokens(self, pts: list[PatchTokens], seed: int = 42) -> None:
        """Build the gallery bank from pre-encoded patch tokens.

        Enables leave-one-out scoring (e.g. test/good support set) without
        re-encoding the same images per query.
        """
        if not pts:
            self.galleries = None
            return
        per_layer: list[list[torch.Tensor]] = [[] for _ in pts[0].layer_tokens]
        self.layer_ids = list(pts[0].layer_ids)
        for pt in pts:
            for li, tok in enumerate(pt.layer_tokens):
                # Keep banks on the inference device for fast NN at score time.
                t = _l2_normalize(tok.float().to(self.device, non_blocking=True))
                per_layer[li].append(t)
        galleries: list[torch.Tensor] = []
        rng = np.random.default_rng(seed)
        for feats in per_layer:
            g = torch.cat(feats, dim=0) if feats else torch.empty(0, device=self.device)
            if self.max_gallery_patches and g.shape[0] > self.max_gallery_patches:
                idx = rng.choice(g.shape[0], size=self.max_gallery_patches, replace=False)
                g = g[torch.from_numpy(idx).to(g.device)]
            galleries.append(g.contiguous())
        self.galleries = galleries

    @torch.inference_mode()
    def score_patches(self, pt: PatchTokens) -> tuple[float, np.ndarray]:
        assert self.galleries is not None and len(self.galleries) > 0
        h, w = pt.grid_hw
        maps = []
        for tok, gal in zip(pt.layer_tokens, self.galleries):
            t = _l2_normalize(tok.float().to(gal.device, non_blocking=True))
            dist = _nn_distance(t, gal).detach().float().cpu().numpy().reshape(h, w)
            maps.append(dist.astype(np.float32))
        stack = np.stack(maps, axis=0)  # [L,H,W]
        if stack.shape[0] == 1:
            amap = stack[0]
        else:
            amap = softmax_fuse_maps(stack, temperature=self.fusion_temperature)
        return float(amap.max()), amap

    @torch.inference_mode()
    def score_image(self, image: Image.Image) -> tuple[float, np.ndarray]:
        return self.score_patches(self.encode_patches(image))

    def evaluate(
        self,
        category: str,
        train_paths: list[Path],
        test_items: list[tuple[Path, int]],
        *,
        flops_g: float | None = None,
        params_m: float | None = None,
        warmup: int = 3,
        notes: str = "",
        extra: dict | None = None,
        map_cache: dict[str, np.ndarray] | None = None,
        cache_paths: set[str] | None = None,
        seed: int = 42,
    ) -> MethodResult:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        gallery_s = self.build_gallery(train_paths, seed=seed)

        for i in range(min(warmup, len(test_items))):
            _ = self.score_image(Image.open(test_items[i][0]).convert("RGB"))

        labels, scores, lats = [], [], []
        gt_masks, amaps = [], []
        for path, y in test_items:
            img = Image.open(path).convert("RGB")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s, amap = self.score_image(img)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
            labels.append(y)
            scores.append(s)
            gt = mvtec_gt_mask(path, target_hw=(256, 256))
            amap_u = upsample_amap(amap, (256, 256))
            gt_masks.append(gt)
            amaps.append(amap_u)
            if map_cache is not None:
                key = str(path.resolve())
                if cache_paths is None or key in cache_paths or str(path) in cache_paths:
                    map_cache[key] = amap_u

        labels_a = np.asarray(labels, dtype=int)
        scores_a = np.asarray(scores, dtype=float)
        auroc = float(roc_auc_score(labels_a, scores_a)) if len(np.unique(labels_a)) > 1 else float("nan")
        f1, prec, rec, thr = best_f1_threshold(labels_a, scores_a)
        p_auroc = pixel_auroc(gt_masks, amaps)
        p_f1, p_prec, p_rec, p_thr = best_pixel_f1(gt_masks, amaps)

        peak = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            peak = float(torch.cuda.max_memory_allocated() / (1024**2))

        n_gal_patches = int(self.galleries[0].shape[0]) if self.galleries else 0
        return MethodResult(
            method=self.name,
            category=category,
            n_gallery=len(train_paths),
            n_test=len(test_items),
            image_auroc=auroc,
            f1=f1,
            precision=prec,
            recall=rec,
            threshold=thr,
            gallery_build_s=float(gallery_s),
            infer_latency_ms_mean=float(np.mean(lats)),
            infer_latency_ms_std=float(np.std(lats)),
            flops_g=flops_g,
            params_m=params_m,
            peak_mem_mb=peak,
            notes=notes,
            extra={
                **(extra or {}),
                "n_gallery_patches_per_layer": n_gal_patches,
                "layers": list(self.layer_ids),
                "fusion": "softmax_distance",
                "fusion_temperature": self.fusion_temperature,
            },
            pixel_auroc=p_auroc,
            pixel_f1=p_f1,
            pixel_precision=p_prec,
            pixel_recall=p_rec,
            pixel_threshold=p_thr,
        )
