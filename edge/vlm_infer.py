"""Edge Qwen-VL inference (small model)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import create_vlm_client


def main():
    p = argparse.ArgumentParser(description="Edge Qwen-VL OK/NG infer")
    p.add_argument("--config", default=str(ROOT / "configs/qwen_vl.yaml"))
    p.add_argument("--image", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--backend", default=None, help="auto|qwen3_vl|transformers|internvl|minicpm")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    edge_cfg = cfg["edge"]
    client = create_vlm_client(
        model_path=args.model_path or edge_cfg["model_path"],
        backend=args.backend or edge_cfg.get("backend", "auto"),
        device=args.device or edge_cfg.get("device", "cuda:0"),
        dtype=edge_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(edge_cfg.get("max_new_tokens", 128)),
        role="edge",
        prompt=cfg.get("prompt"),
    )
    res = client.infer(args.image)
    hard = client.is_hard(
        res,
        float(edge_cfg.get("conf_low", 0.55)),
        float(edge_cfg.get("conf_high", 0.85)),
        use_band=bool(cfg.get("collab", {}).get("uncertain_band", True)),
    )
    out = res.to_dict()
    out["hard_flag"] = hard
    out["path"] = "CLOUD_REVIEW" if hard else "LOCAL"
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
