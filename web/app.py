#!/usr/bin/env python3
"""CloudEdge-abnormal Web console.

Env: conda activate base
Run:
  CUDA_VISIBLE_DEVICES=0 uvicorn web.app:app --host 0.0.0.0 --port 7860
  # or
  python -m web.app
"""
from __future__ import annotations

import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
DATA_ROOT = ROOT / "datasets" / "mvtec"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = FastAPI(title="CloudEdge Defect Console", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cloud_client = None
_cloud_lock = threading.Lock()
_cloud_error: str | None = None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_summary() -> dict:
    all_cats = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / "all_categories.json") or []
    anomalib = None
    mean_md = ROOT / "outputs" / "reports" / "mvtec_mean.md"
    # parse light stats from hybrid means
    h8 = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / "hybrid_mean.json") or {}
    h4 = _read_json(ROOT / "outputs" / "hybrid_lora" / "hybrid_mean.json") or {}
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
            "edge": "Anomalib PaDiM / resnet18",
            "cloud": "Qwen3-VL-8B + LoRA (also 4B LoRA available)",
            "collab": "Hard-example upload (uncertain score band)",
        },
    }


def list_categories() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and (p / "test").exists()
    )


def _sample_key(path: str | Path) -> str:
    """Stable key: defect_folder/filename (avoids 000.png collisions across classes)."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def _cloud_reviewed_keys(category: str) -> set[str]:
    """Keys of images that already have cloud LLM outputs in benches."""
    keys: set[str] = set()
    for source in ("hybrid_lora_8b", "hybrid_lora", "hybrid"):
        bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
        if not bench:
            continue
        for row in bench.get("rows", []):
            if row.get("cloud") and row.get("path"):
                keys.add(_sample_key(row["path"]))
    return keys


def list_test_images(category: str, limit: int = 40) -> list[dict]:
    test = DATA_ROOT / category / "test"
    if not test.exists():
        return []
    reviewed = _cloud_reviewed_keys(category)
    items = []
    for sub in sorted(test.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                key = _sample_key(p)
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
    # Prefer samples with cached cloud LLM output, then balance OK/NG
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
    pack = _read_json(ROOT / "outputs" / "hybrid_lora_8b" / category / "edge_scores.json")
    if not pack:
        pack = _read_json(ROOT / "outputs" / "hybrid" / category / "edge_scores.json")
    if not pack:
        return None
    target = str(Path(image_path).resolve())
    name = Path(image_path).name
    parent = Path(image_path).parent.name
    for it in pack.get("items", []):
        p = it.get("path") or ""
        if str(Path(p).resolve()) == target or (Path(p).name == name and Path(p).parent.name == parent):
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
    bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        return None
    key = _sample_key(image_path)
    for row in bench.get("rows", []):
        p = row.get("path") or ""
        if p and _sample_key(p) == key:
            return row
    return None


def case_lookup_with_cloud(category: str, image_path: str) -> dict | None:
    """Prefer a bench row that includes cloud LLM output across hybrid sources."""
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


def _url_for_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return f"/api/image?path={path.resolve()}"


def find_anomalib_viz(category: str, image_path: str, role: str) -> Path | None:
    """Locate Anomalib strip viz: Image | GT | Heatmap | Pred mask."""
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


def build_viz_payload(category: str, image_path: str) -> dict[str, Any]:
    edge_viz = find_anomalib_viz(category, image_path, "edge")
    cloud_viz = find_anomalib_viz(category, image_path, "cloud")
    gt_mask = find_gt_mask(category, image_path)
    return {
        "edge_strip": _url_for_file(edge_viz),
        # heavy Anomalib cloud (PatchCore) — NOT Qwen-VL; VLM has no heatmap
        "cloud_strip": _url_for_file(cloud_viz),
        "gt_mask": _url_for_file(gt_mask),
        "legend": "Anomalib only (PaDiM/PatchCore). VLM outputs JSON, not heatmaps.",
    }


def get_cloud_client():
    global _cloud_client, _cloud_error
    if _cloud_client is not None:
        return _cloud_client
    with _cloud_lock:
        if _cloud_client is not None:
            return _cloud_client
        try:
            from src.vlm import QwenVLClient

            cfg_path = ROOT / "configs" / "hybrid_lora.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cloud = cfg["cloud"]
            device = os.environ.get("WEB_VLM_DEVICE", cloud.get("device", "cuda:0"))
            _cloud_client = QwenVLClient(
                model_path=cloud["model_path"],
                adapter_path=cloud.get("adapter_path"),
                device=device,
                dtype=cloud.get("dtype", "bfloat16"),
                max_new_tokens=int(cloud.get("max_new_tokens", 128)),
                role="cloud",
                prompt=cfg.get("prompt"),
            )
            _cloud_error = None
            return _cloud_client
        except Exception as exc:  # noqa: BLE001
            _cloud_error = str(exc)
            raise


@app.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "cloud_loaded": _cloud_client is not None,
        "cloud_error": _cloud_error,
        "data_root": str(DATA_ROOT),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@app.get("/api/summary")
def api_summary():
    return load_summary()


@app.get("/api/categories")
def api_categories():
    return {"categories": list_categories()}


@app.get("/api/images")
def api_images(category: str = Query(...), limit: int = Query(36, ge=1, le=120)):
    if category not in list_categories():
        raise HTTPException(404, f"unknown category: {category}")
    return {"category": category, "images": list_test_images(category, limit=limit)}


@app.get("/api/image")
def api_image(path: str = Query(...)):
    p = Path(path).resolve()
    # allow dataset + outputs only
    allowed = [DATA_ROOT.resolve(), (ROOT / "outputs").resolve()]
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(403, "path not allowed")
    if not p.exists() or p.suffix.lower() not in IMG_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(p)


@app.get("/api/cases")
def api_cases(category: str = Query(...), source: str = Query("hybrid_lora_8b"), limit: int = 24):
    bench = _read_json(ROOT / "outputs" / source / category / "bench.json")
    if not bench:
        raise HTTPException(404, f"no bench for {category} under {source}")
    rows = []
    for row in bench.get("rows", []):
        if not row.get("cloud") and source.startswith("hybrid"):
            # still include some local-only for context, but prefer reviewed
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
    # if few cloud rows, fill with first rows
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


@app.post("/api/demo")
async def api_demo(
    category: str = Form(...),
    image_path: str | None = Form(None),
    live_cloud: str = Form("false"),
    file: UploadFile | None = File(None),
):
    """Demo one image: edge score (precomputed) + optional live LoRA cloud / cached case."""
    live = str(live_cloud).lower() in {"1", "true", "yes", "on"}
    tmp_path = None
    if file is not None and file.filename:
        tmp_dir = ROOT / "outputs" / "web_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / Path(file.filename).name
        tmp_path.write_bytes(await file.read())
        image_path = str(tmp_path.resolve())

    if not image_path:
        raise HTTPException(400, "image_path or file required")
    p = Path(image_path)
    if not p.exists():
        raise HTTPException(404, f"image not found: {image_path}")

    edge = edge_lookup(category, image_path)
    cached = case_lookup_with_cloud(category, image_path)

    result: dict[str, Any] = {
        "category": category,
        "image_path": str(p.resolve()),
        "image_url": f"/api/image?path={p.resolve()}",
        "edge": edge,
        "viz": build_viz_payload(category, image_path),
        "cached_case": None,
        "cloud_live": None,
        "route": None,
        "final_decision": None,
    }

    if cached:
        result["cached_case"] = {
            "gt": "NG" if cached.get("label") == 1 else "OK",
            "edge_pred": cached.get("edge_pred"),
            "edge_score": cached.get("edge_score"),
            "final": cached.get("final_decision"),
            "path_type": cached.get("path_type"),
            "cloud": cached.get("cloud"),
        }

    # routing preview from edge
    hard = bool(edge.get("hard")) if edge else True
    if not live:
        if cached and cached.get("cloud"):
            result["route"] = cached.get("path_type") or ("CLOUD_REVIEW" if hard else "LOCAL")
            result["final_decision"] = cached.get("final_decision")
        elif edge:
            result["route"] = "CLOUD_REVIEW" if hard else "LOCAL"
            result["final_decision"] = edge.get("edge_pred")
        return result

    # live cloud VLM
    try:
        client = get_cloud_client()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"cloud model unavailable: {exc}") from exc

    # only call cloud if hard or no edge info
    if edge and not hard:
        result["route"] = "LOCAL"
        result["final_decision"] = edge.get("edge_pred")
        result["cloud_live"] = {"skipped": True, "reason": "edge confident / not hard"}
        return result

    vlm = client.infer(p)
    result["cloud_live"] = vlm.to_dict()
    result["route"] = "CLOUD_REVIEW"
    result["final_decision"] = vlm.decision
    return result


@app.post("/api/cloud/load")
def api_cloud_load():
    try:
        get_cloud_client()
        return {"ok": True, "loaded": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


def main():
    import uvicorn

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "7860"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
