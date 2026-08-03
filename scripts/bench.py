#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collab import CollabConfig, run_baselines
from src.data import MVTecCategory
from src.models import PatchCoreConfig, PatchCoreLite


def load_model(role_cfg: dict, ckpt_path: Path, device: str) -> PatchCoreLite:
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
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_bank(state)
    return model


def to_markdown(report: dict, category: str) -> str:
    m = report["contest_mapped"]
    d = report["detection"]
    lat = report["latency"]
    comm = report["communication"]
    lines = [
        f"# Cloud-Edge Anomaly Bench — `{category}`",
        "",
        "## Detection",
        "",
        "| Scheme | Image-AUROC | F1 | Precision | Recall | FN rate | FP rate |",
        "|--------|-------------|----|-----------|--------|---------|---------|",
    ]
    for name, key in [
        ("B0 cloud-only", "B0_cloud_only"),
        ("B1 edge-only", "B1_edge_only"),
        ("S collab", "S_collab"),
        ("edge model", "edge"),
        ("cloud model", "cloud"),
    ]:
        x = d[key]
        lines.append(
            f"| {name} | {x['image_auroc']:.4f} | {x['f1']:.4f} | {x['precision']:.4f} | "
            f"{x['recall']:.4f} | {x['fn_rate']:.4f} | {x['fp_rate']:.4f} |"
        )

    lines += [
        "",
        "## Latency (ms)",
        "",
        "| Scheme | mean | p50 | p95 |",
        "|--------|------|-----|-----|",
        f"| B0 | {lat['B0']['mean_ms']:.2f} | {lat['B0']['p50_ms']:.2f} | {lat['B0']['p95_ms']:.2f} |",
        f"| B1 | {lat['B1']['mean_ms']:.2f} | {lat['B1']['p50_ms']:.2f} | {lat['B1']['p95_ms']:.2f} |",
        f"| S all | {lat['S_all']['mean_ms']:.2f} | {lat['S_all']['p50_ms']:.2f} | {lat['S_all']['p95_ms']:.2f} |",
        f"| S local-path | {lat['S_local_path']['mean_ms']:.2f} | {lat['S_local_path']['p50_ms']:.2f} | {lat['S_local_path']['p95_ms']:.2f} |",
        "",
        "## Communication",
        "",
        f"- Hard upload ratio: **{comm['hard_upload_ratio']:.2%}**",
        f"- Upload reduce vs B0: **{comm['upload_reduce_vs_B0']:.2%}**",
        f"- B0 bytes: {comm['B0_upload_bytes']:,} | S bytes: {comm['S_upload_bytes']:,}",
        "",
        "## Contest-mapped metrics",
        "",
        "| ID | Value | Target / note |",
        "|----|-------|---------------|",
        f"| M1 AUROC retention | {m['M1_capability_retention_auroc']:.2%} | 80%–90% |",
        f"| M1 F1 retention | {m['M1_capability_retention_f1']:.2%} | 80%–90% |",
        f"| M2 first-response reduce | {m['M2_first_response_reduce_vs_cloud']:.2%} | ≥75% |",
        f"| M3 edge peak mem | {m['M3_edge_peak_mem_mb']:.1f} MB | ≤1536 MB (pass={m['M3_pass_leq_1536mb']}) |",
        f"| M4 weak-net keep rate | {m['M4_weak_net_service_keep_rate']:.2%} | ≥90% |",
        f"| M5 local e2e mean | {m['M5_mean_e2e_local_ms']:.2f} ms | ≤200 ms (pass={m['M5_pass_local_leq_200ms']}) |",
        f"| M5 all-path mean | {m['M5_mean_e2e_all_ms']:.2f} ms | report P50/P95 separately |",
        f"| M6 conflict ratio | {m['M6_conflict_ratio']:.2%} | ≤5% |",
        f"| M7 resolve rate | {m['M7_conflict_resolve_rate']:.2%} | ≥90% |",
        f"| C4 latency reduce vs B0 | {m['C4_latency_reduce_vs_B0']:.2%} | higher better |",
        f"| C5 F1 Δ vs B1 | {m['C5_f1_delta_vs_B1']:+.4f} | ≥0 preferred |",
        f"| C5 AUROC Δ vs B1 | {m['C5_auroc_delta_vs_B1']:+.4f} | ≥0 preferred |",
        "",
        f"Hard mining band: [{report['hard_mining']['band_low']:.4f}, {report['hard_mining']['band_high']:.4f}], "
        f"n_hard={report['hard_mining']['n_hard']}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--out", default=str(ROOT / "outputs/reports"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.category:
        cfg["category"] = args.category
    device = args.device or cfg.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    category = cfg["category"]
    ckpt_dir = Path(args.ckpt_dir or ROOT / "outputs/checkpoints" / category)
    edge = load_model(cfg["edge"], ckpt_dir / f"{cfg['edge']['name']}.pt", device)
    cloud = load_model(cfg["cloud"], ckpt_dir / f"{cfg['cloud']['name']}.pt", device)

    ds = MVTecCategory(cfg["data_root"], category, "test", cfg["image_size"])
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=device.startswith("cuda"))

    collab = CollabConfig(**cfg.get("collab", {}))
    report = run_baselines(edge, cloud, loader, collab, torch.device(device))
    report["category"] = category
    report["device"] = device

    out_dir = Path(args.out) / category
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    md_path = out_dir / "metrics.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md = to_markdown(report, category)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
