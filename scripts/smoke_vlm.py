#!/usr/bin/env python3
"""Load one local VLM checkpoint and run the unified detector interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm import create_vlm_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    client = create_vlm_client(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        role="smoke",
    )
    print(json.dumps(client.infer(args.image).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
