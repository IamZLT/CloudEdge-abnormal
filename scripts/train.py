#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import MVTecCategory, summarize_split
from src.models import PatchCoreConfig, PatchCoreLite


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(role: str, cfg: dict, out_dir: Path):
    device = cfg.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        cfg["device"] = device

    role_cfg = cfg[role]
    ds_train = MVTecCategory(cfg["data_root"], cfg["category"], "train", cfg["image_size"])
    ds_test = MVTecCategory(cfg["data_root"], cfg["category"], "test", cfg["image_size"])
    print(f"[{role}] train={summarize_split(ds_train)} test={summarize_split(ds_test)}")

    train_loader = DataLoader(ds_train, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = PatchCoreLite(
        PatchCoreConfig(
            name=role_cfg["name"],
            backbone=role_cfg["backbone"],
            layers=role_cfg.get("layers", ["layer2", "layer3"]),
            coreset_ratio=role_cfg.get("coreset_ratio", 0.1),
            max_memory_bank=role_cfg.get("max_memory_bank", 10000),
            device=device,
        )
    )
    print(f"[{role}] fitting memory bank with {role_cfg['backbone']} ...")
    model.fit(tqdm(train_loader, desc=f"fit-{role}"))

    # calibrate threshold on test (demo/report; for strict protocol use holdout)
    scores, labels = [], []
    with torch.no_grad():
        for images, y, _ in tqdm(test_loader, desc=f"score-{role}"):
            s = model.predict_score(images)
            scores.extend(s.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    thr, f1 = model.calibrate_threshold(scores, labels)
    from src.metrics import binary_detection_metrics

    det = binary_detection_metrics(labels, scores, thr)
    print(f"[{role}] threshold={thr:.4f} f1={f1:.4f} auroc={det['image_auroc']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{role_cfg['name']}.pt"
    torch.save(model.state_dict_bank(), ckpt)
    meta = {
        "role": role,
        "category": cfg["category"],
        "data_root": cfg["data_root"],
        "image_size": cfg["image_size"],
        "device": device,
        "checkpoint": str(ckpt),
        "metrics": det,
        "bank_size": int(model.memory_bank.shape[0]) if model.memory_bank is not None else 0,
    }
    with open(out_dir / f"{role_cfg['name']}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(ROOT / "outputs/checkpoints"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.category:
        cfg["category"] = args.category
    if args.device:
        cfg["device"] = args.device

    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out) / cfg["category"]
    edge_meta = train_one("edge", cfg, out_dir)
    cloud_meta = train_one("cloud", cfg, out_dir)
    summary = {"edge": edge_meta, "cloud": cloud_meta}
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
