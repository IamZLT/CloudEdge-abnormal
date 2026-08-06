#!/usr/bin/env python3
"""Train edge/cloud models with Anomalib (preferred stack)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_datamodule(cfg: dict):
    from anomalib.data import MVTecAD

    return MVTecAD(
        root=cfg["data_root"],
        category=cfg["category"],
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )


def resolve_role_cfg(cfg: dict, role: str) -> dict:
    """Anomalib role config. Edge default is Qwen gallery; PaDiM lives under edge.alternatives."""
    role_cfg = dict(cfg.get(role) or {})
    if role == "edge" and not role_cfg.get("model"):
        alt = (role_cfg.get("alternatives") or {}).get("padim") or {}
        role_cfg = {**role_cfg, **alt}
    if not role_cfg.get("model"):
        # fallback for older configs
        if role == "edge":
            role_cfg.setdefault("model", "Padim")
            role_cfg.setdefault("backbone", "resnet18")
        elif role == "cloud":
            role_cfg.setdefault("model", "Patchcore")
            role_cfg.setdefault("backbone", "wide_resnet50_2")
    return role_cfg


def build_model(role_cfg: dict):
    from src.offline_timm import enable as enable_offline_timm

    name = role_cfg["model"].lower()
    backbone = role_cfg.get("backbone", "resnet18")
    enable_offline_timm(backbone)
    if name == "padim":
        from anomalib.models import Padim

        return Padim(backbone=backbone)
    if name == "patchcore":
        from anomalib.models import Patchcore

        return Patchcore(backbone=backbone)
    if name in {"efficientad", "efficient_ad"}:
        from anomalib.models import EfficientAd

        return EfficientAd()
    raise ValueError(f"Unsupported Anomalib model: {role_cfg['model']}")


def train_role(role: str, cfg: dict, *, do_export: bool = True) -> dict:
    from anomalib.engine import Engine

    role_cfg = resolve_role_cfg(cfg, role)
    # keep anomalib outputs under outputs/anomalib even if results_dir is generic
    results_root = Path(cfg.get("anomalib_results_dir") or cfg.get("results_dir") or "outputs/anomalib")
    if results_root.name != "anomalib" and role in {"edge", "cloud"}:
        # default.yaml now uses results_dir=outputs; park Anomalib under outputs/anomalib
        results_root = Path("outputs/anomalib")
    out_dir = results_root / cfg["category"] / role
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(cfg)
    model = build_model(role_cfg)
    use_gpu = str(cfg.get("device", "")).startswith("cuda")
    engine = Engine(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        default_root_dir=str(out_dir),
        max_epochs=1,
        # Padim/Patchcore are embedding methods; one "epoch" is enough
        enable_checkpointing=True,
    )

    print(f"[{cfg['category']}/{role}] training {role_cfg['model']} / {role_cfg.get('backbone')} ...")
    engine.fit(model=model, datamodule=datamodule)
    test_out = engine.test(model=model, datamodule=datamodule)

    # export for edge runtime (optional; skip on full-dataset sweeps)
    export_fmt = role_cfg.get("export", "torch")
    export_path = None
    if do_export:
        try:
            from anomalib.deploy import ExportType

            fmt_map = {
                "torch": ExportType.TORCH,
                "onnx": ExportType.ONNX,
                "openvino": ExportType.OPENVINO,
            }
            if export_fmt in fmt_map:
                export_path = engine.export(
                    model=model,
                    export_type=fmt_map[export_fmt],
                    export_root=str(out_dir / "export"),
                )
                print(f"[{cfg['category']}/{role}] exported ({export_fmt}): {export_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{cfg['category']}/{role}] export skipped/failed: {exc}")

    metrics = test_out[0] if isinstance(test_out, list) and test_out else test_out
    # flatten lightning metric keys if needed
    flat = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                flat[str(k)] = float(v)
            except Exception:  # noqa: BLE001
                flat[str(k)] = v

    ckpt = None
    # best effort locate lightning ckpt
    for p in out_dir.rglob("*.ckpt"):
        ckpt = str(p)
        break

    meta = {
        "role": role,
        "model": role_cfg["model"],
        "backbone": role_cfg.get("backbone"),
        "category": cfg["category"],
        "metrics": flat,
        "checkpoint": ckpt,
        "export_path": str(export_path) if export_path else None,
        "export_format": export_fmt,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # optional mlflow
    if cfg.get("mlflow", {}).get("enabled", False):
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
            with mlflow.start_run(run_name=f"{cfg['category']}-{role}"):
                mlflow.log_params(
                    {
                        "role": role,
                        "model": role_cfg["model"],
                        "backbone": role_cfg.get("backbone"),
                        "category": cfg["category"],
                    }
                )
                for k, v in flat.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k.replace("/", "_"), float(v))
                mlflow.log_artifact(str(out_dir / "train_meta.json"))
        except Exception as exc:  # noqa: BLE001
            print(f"[{role}] mlflow logging skipped: {exc}")

    print(f"[{cfg['category']}/{role}] done. metrics={flat}")
    return meta


def list_mvtec_categories(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    cats = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "train" / "good").exists() and (p / "test").exists()
    )
    return cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None, help="single category, or 'all'")
    parser.add_argument("--categories", default=None, help="comma-separated list; overrides --category")
    parser.add_argument("--device", default=None)
    parser.add_argument("--roles", default="edge,cloud", help="edge,cloud or either")
    parser.add_argument("--no-export", action="store_true", help="skip OpenVINO/ONNX export")
    parser.add_argument("--skip-existing", action="store_true", help="skip category if train_meta exists")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.category == "all":
        categories = list_mvtec_categories(cfg["data_root"])
    elif args.category:
        categories = [args.category]
    else:
        categories = [cfg["category"]]

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    all_summary = {}
    for cat in categories:
        cfg["category"] = cat
        base = Path(cfg["results_dir"]) / cat
        if args.skip_existing and all((base / r / "train_meta.json").exists() for r in roles):
            print(f"[{cat}] skip existing train_meta for roles={roles}")
            continue
        summary = {}
        for role in roles:
            if args.skip_existing and (base / role / "train_meta.json").exists():
                print(f"[{cat}/{role}] skip existing")
                summary[role] = json.loads((base / role / "train_meta.json").read_text(encoding="utf-8"))
                continue
            summary[role] = train_role(role, cfg, do_export=not args.no_export)
        out = base / "train_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        all_summary[cat] = {r: summary[r].get("metrics", {}) for r in summary}
        print(f"Wrote {out}")

    if len(categories) > 1:
        agg_path = Path(cfg["results_dir"]) / "train_all_summary.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
