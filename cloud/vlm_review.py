"""Cloud review CLI: DINOv3 (pixel kNN) + Qwen3.5 (semantic) fusion.

Thin wrapper over ``src.cloud_reviewer.CloudReviewer`` — the unified cloud-side
detector. Outputs a continuous anomaly score (0..1) plus a fixed-threshold OK/NG
decision.

Usage:
  python cloud/vlm_review.py --image <img> --category <cat> \
      --config third_part/Cloud-abnormal-cx/configs/default_224.yaml \
      --memory-dir outputs/cloud_abnormal_cx_224/memory

Notes:
  - Memory banks are per-category and must already exist (run the `fit` step).
  - --threshold defaults to 0.5; on the 224-resolution MVTec-LLM split the F1-max
    operating point is ~0.67, so pass --threshold 0.67 for the balanced point.
  - --disable-qwen runs the DINO branch alone (A/B of the fusion gain).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloud_reviewer import CloudReviewer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cloud DINOv3+Qwen3.5 fusion review (OK/NG + anomaly score)"
    )
    p.add_argument("--image", required=True, help="Single image to review")
    p.add_argument("--category", required=True, help="Product category (selects the memory bank)")
    p.add_argument("--config", default=None)
    p.add_argument("--memory-dir", default=None)
    p.add_argument("--dataset", default="mvtec_llm")
    p.add_argument("--threshold", type=float, default=0.5, help="OK/NG score threshold")
    p.add_argument("--use-large", action="store_true", help="Use Qwen3.5-9B instead of 2B")
    p.add_argument("--disable-qwen", action="store_true", help="DINO branch only (no LLM fusion)")
    p.add_argument("--device", default=None, help="Override cfg.model.device")
    args = p.parse_args()

    reviewer = CloudReviewer(
        config_path=args.config,
        memory_dir=args.memory_dir,
        dataset=args.dataset,
        use_large=args.use_large,
        disable_qwen=args.disable_qwen,
        device=args.device,
        threshold=args.threshold,
    )
    result = reviewer.review(args.image, args.category)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
