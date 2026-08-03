"""Edge inference entry (thin wrapper)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_transform
from src.models import PatchCoreConfig, PatchCoreLite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--backbone", default="resnet18")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    model = PatchCoreLite(
        PatchCoreConfig(name="edge", backbone=args.backbone, device=device)
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_bank(state)

    tfm = build_transform(args.image_size)
    x = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    score = float(model.predict_score(x).item())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000
    decision = "NG" if score >= model.threshold else "OK"
    hard = abs(score - model.threshold) < 0.05
    print(
        json.dumps(
            {
                "decision": decision,
                "score": score,
                "threshold": model.threshold,
                "hard_flag": hard,
                "latency_ms": latency_ms,
                "path": "LOCAL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
