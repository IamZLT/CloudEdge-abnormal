"""Cloud Qwen-VL review (large model)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import QwenVLClient


def main():
    p = argparse.ArgumentParser(description="Cloud Qwen-VL OK/NG review")
    p.add_argument("--config", default=str(ROOT / "configs/qwen_vl.yaml"))
    p.add_argument("--image", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cloud_cfg = cfg["cloud"]
    client = QwenVLClient(
        model_path=args.model_path or cloud_cfg["model_path"],
        device=args.device or cloud_cfg.get("device", "cuda:0"),
        dtype=cloud_cfg.get("dtype", "bfloat16"),
        max_new_tokens=int(cloud_cfg.get("max_new_tokens", 160)),
        role="cloud",
        prompt=cfg.get("prompt"),
    )
    res = client.infer(args.image)
    out = res.to_dict()
    out["path"] = "CLOUD"
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
