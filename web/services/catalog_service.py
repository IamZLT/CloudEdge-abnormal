"""Dataset listing, bench lookups, and Anomalib viz helpers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from web.config import DATA_ROOT, IMG_EXT, ROOT, read_json


@lru_cache(maxsize=1)
def load_summary() -> dict:
    all_cats = read_json(ROOT / "outputs" / "hybrid_lora_8b" / "all_categories.json") or []
    mean_md = ROOT / "outputs" / "reports" / "mvtec_mean.md"
    h8 = read_json(ROOT / "outputs" / "hybrid_lora_8b" / "hybrid_mean.json") or {}
    h4 = read_json(ROOT / "outputs" / "hybrid_lora" / "hybrid_mean.json") or {}
    if all_cats:
        b1 = float(np.mean([r["B1_f1"] for r in all_cats]))
        zs = float(np.mean([r["ZS8B_f1"] for r in all_cats if r.get("ZS8B_f1") is not None]))
        l4 = float(np.mean([r["Lora4B_f1"] for r in all_cats if r.get("Lora4B_f1") is not None]))
        l8 = float(np.mean([r["Lora8B_f1"] for r in all_cats if r.get("Lora8B_f1") is not None]))
    else:
        b1 = zs = l4 = l8 = float("nan")
    return {
        "categories": all_cats,
        "means": {
            "B1_f1": b1,
            "ZS8B_f1": zs,
            "Lora4B_f1": l4,
            "Lora8B_f1": l8,
        },
        "anomalib_report": str(mean_md) if mean_md.exists() else None,
        "hybrid_lora_8b": h8.get("mean_S_f1"),
        "hybrid_lora_4b": h4.get("mean_S_f1"),
        "stack": {
            "edge": "Qwen3.5-0.8B vision multi-layer patch gallery (PaDiM optional)",
            "cloud": "Qwen3-VL-8B + LoRA (also 4B LoRA available)",
            "collab": "Multi-edge fleet + Qwen3.5 RouteAgent (GGUF Q4) + per-node network sim",
        },
    }


def list_categories() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        p.name for p in DATA_ROOT.iterdir() if p.is_dir() and (p / "test").exists()
    )


def sample_key(path: str | Path) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def cloud_reviewed_keys(category: str) -> set[str]:
    keys: set[str] = set()
    for source in ("hybrid_lora_8b", "hybrid_lora", "hybrid"):
        bench = read_json(ROOT / "outputs" / source / category / "bench.json")
        if not bench:
            continue
        for row in bench.get("rows", []):
            if row.get("cloud") and row.get("path"):
                keys.add(sample_key(row["path"]))
    return keys


def list_test_images(category: str, limit: int = 40) -> list[dict]:
    test = DATA_ROOT / category / "test"
    if not test.exists():
        return []
    reviewed = cloud_reviewed_keys(category)
    items = []
    for sub in sorted(test.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                key = sample_key(p)
                items.append(
                    {
                        "path": str(p.resolve()),
                        "rel": f"{category}/test/{sub.name}/{p.name}",
                        "name": p.name,
                        "defect_type": "none" if label == 0 else sub.name,
                        "label": label,
                        "gt": "OK" if label == 0 else "NG",
                        "has_llm": key in reviewed,
                        "sample_key": key,
                    }
                )
    with_llm = [x for x in items if x["has_llm"]]
    without = [x for x in items if not x["has_llm"]]
    picked: list[dict] = []
    for pool in (with_llm, without):
        oks = [x for x in pool if x["label"] == 0]
        ngs = [x for x in pool if x["label"] == 1]
        while (oks or ngs) and len(picked) < limit:
            if ngs:
                picked.append(ngs.pop(0))
            if len(picked) >= limit:
                break
            if oks:
                picked.append(oks.pop(0))
    return picked[:limit]


def edge_lookup(category: str, image_path: str) -> dict | None:
    pack = read_json(ROOT / "outputs" / "hybrid_lora_8b" / category / "edge_scores.json")
    if not pack:
        pack = read_json(ROOT / "outputs" / "hybrid" / category / "edge_scores.json")
    if not pack:
        return None
    target = str(Path(image_path).resolve())
    name = Path(image_path).name
    parent = Path(image_path).parent.name
    for it in pack.get("items", []):
        p = it.get("path") or ""
        if str(Path(p).resolve()) == target or (
            Path(p).name == name and Path(p).parent.name == parent
        ):
            return {
                "edge_score": it["edge_score"],
                "edge_pred": "NG" if it.get("edge_pred") else "OK",
                "hard": bool(it.get("hard")),
                "threshold": pack.get("threshold"),
                "band_low": pack.get("band_low"),
                "band_high": pack.get("band_high"),
            }
    return None


def case_lookup(category: str, image_path: str, source: str = "hybrid_lora_8b") -> dict | None:
    bench = read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        return None
    key = sample_key(image_path)
    for row in bench.get("rows", []):
        p = row.get("path") or ""
        if p and sample_key(p) == key:
            return row
    return None


def case_lookup_with_cloud(category: str, image_path: str) -> dict | None:
    best: dict | None = None
    for source in ("hybrid_lora_8b", "hybrid_lora", "hybrid"):
        row = case_lookup(category, image_path, source)
        if not row:
            continue
        if row.get("cloud"):
            return row
        if best is None:
            best = row
    return best


def url_for_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return f"/api/image?path={path.resolve()}"


def find_anomalib_viz(category: str, image_path: str, role: str) -> Path | None:
    parent = Path(image_path).parent.name
    name = Path(image_path).name
    root = ROOT / "outputs" / "anomalib" / category / role
    if not root.exists():
        return None
    hits = sorted(root.glob(f"**/images/{parent}/{name}"))
    return hits[0] if hits else None


def find_gt_mask(category: str, image_path: str) -> Path | None:
    parent = Path(image_path).parent.name
    if parent == "good":
        return None
    stem = Path(image_path).stem
    mask = DATA_ROOT / category / "ground_truth" / parent / f"{stem}_mask.png"
    return mask if mask.exists() else None


def build_viz_payload(
    category: str,
    image_path: str,
    *,
    include_cloud: bool = True,
) -> dict[str, Any]:
    edge_viz = find_anomalib_viz(category, image_path, "edge")
    cloud_viz = find_anomalib_viz(category, image_path, "cloud") if include_cloud else None
    gt_mask = find_gt_mask(category, image_path)
    return {
        "edge_strip": url_for_file(edge_viz),
        "cloud_strip": url_for_file(cloud_viz) if include_cloud else None,
        "gt_mask": url_for_file(gt_mask),
        "include_cloud": bool(include_cloud),
        "legend": "Anomalib only (PaDiM/PatchCore). VLM outputs JSON, not heatmaps.",
    }


def list_cases(category: str, source: str = "hybrid_lora_8b", limit: int = 24) -> dict[str, Any]:
    bench = read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        raise FileNotFoundError(f"no bench for {category} under {source}")
    rows: list[dict[str, Any]] = []
    for row in bench.get("rows", []):
        if not row.get("cloud") and source.startswith("hybrid"):
            continue
        viz = build_viz_payload(category, row["path"])
        rows.append(
            {
                "path": row["path"],
                "name": Path(row["path"]).name,
                "gt": "NG" if row.get("label") == 1 else "OK",
                "edge_pred": row.get("edge_pred"),
                "edge_score": row.get("edge_score"),
                "final": row.get("final_decision"),
                "path_type": row.get("path_type"),
                "viz": viz,
                "cloud": {
                    "decision": (row.get("cloud") or {}).get("decision"),
                    "confidence": (row.get("cloud") or {}).get("confidence"),
                    "defect_type": (row.get("cloud") or {}).get("defect_type"),
                    "reason": (row.get("cloud") or {}).get("reason"),
                    "raw": (row.get("cloud") or {}).get("raw"),
                    "latency_ms": (row.get("cloud") or {}).get("latency_ms"),
                }
                if row.get("cloud")
                else None,
            }
        )
        if len(rows) >= limit:
            break
    if len(rows) < min(8, limit):
        for row in bench.get("rows", []):
            if any(r["path"] == row["path"] for r in rows):
                continue
            rows.append(
                {
                    "path": row["path"],
                    "name": Path(row["path"]).name,
                    "gt": "NG" if row.get("label") == 1 else "OK",
                    "edge_pred": row.get("edge_pred"),
                    "edge_score": row.get("edge_score"),
                    "final": row.get("final_decision"),
                    "path_type": row.get("path_type"),
                    "cloud": None,
                }
            )
            if len(rows) >= limit:
                break
    return {
        "category": category,
        "source": source,
        "detection": bench.get("detection"),
        "cloud_model": bench.get("cloud"),
        "cases": rows,
    }
