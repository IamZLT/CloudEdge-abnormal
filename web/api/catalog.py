"""Categories / images / cases / static image file serving."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from web.config import DATA_ROOT, IMG_EXT, ROOT
from web.services import catalog_service

router = APIRouter(tags=["catalog"])


@router.get("/api/summary")
def api_summary():
    return catalog_service.load_summary()


@router.get("/api/categories")
def api_categories():
    return {"categories": catalog_service.list_categories()}


@router.get("/api/images")
def api_images(category: str = Query(...), limit: int = Query(36, ge=1, le=120)):
    if category not in catalog_service.list_categories():
        raise HTTPException(404, f"unknown category: {category}")
    return {
        "category": category,
        "images": catalog_service.list_test_images(category, limit=limit),
    }


@router.get("/api/image")
def api_image(path: str = Query(...)):
    p = Path(path).resolve()
    allowed = [DATA_ROOT.resolve(), (ROOT / "outputs").resolve()]
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(403, "path not allowed")
    if not p.exists() or p.suffix.lower() not in IMG_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(p)


@router.get("/api/cases")
def api_cases(
    category: str = Query(...),
    source: str = Query("hybrid_lora_8b"),
    limit: int = 24,
):
    try:
        return catalog_service.list_cases(category, source=source, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
