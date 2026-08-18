from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import evaluate_dataset, fit_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Frozen Qwen3.5 + DINOv3 cloud anomaly detection")
    result.add_argument("--config", default="configs/default.yaml")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--dataset", choices=("mvtec", "mvtec_llm", "visa"), required=True)
        item.add_argument("--root", help="Override dataset root")
        item.add_argument("--memory-dir", default="outputs/memory")
        item.add_argument("--categories", help="Comma-separated categories; default=all")
        if name == "evaluate":
            item.add_argument("--use-large", action="store_true")
            item.add_argument("--disable-qwen", action="store_true")
    return result


def _split_categories(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config)
    default_roots = {
        "mvtec": cfg.data.mvtec_root,
        "mvtec_llm": cfg.data.mvtec_llm_root,
        "visa": cfg.data.visa_root,
    }
    root = args.root or default_roots[args.dataset]
    categories = _split_categories(args.categories)
    if args.command == "fit":
        fit_dataset(cfg, args.dataset, root, args.memory_dir, categories=categories)
    else:
        metrics = evaluate_dataset(
            cfg,
            args.dataset,
            root,
            args.memory_dir,
            use_large=args.use_large,
            disable_qwen=args.disable_qwen,
            categories=categories,
        )
        print(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

