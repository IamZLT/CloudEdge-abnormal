"""Single-node demo endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from web.services import demo_service

router = APIRouter(tags=["demo"])


@router.post("/api/demo")
async def api_demo(
    category: str = Form(...),
    image_path: str | None = Form(None),
    live_cloud: str = Form("false"),
    use_route_agent: str = Form("true"),
    edge_node_id: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    live = str(live_cloud).lower() in {"1", "true", "yes", "on"}
    use_agent = str(use_route_agent).lower() in {"1", "true", "yes", "on"}

    if file is not None and file.filename:
        try:
            path = demo_service.save_upload_bytes(file.filename, await file.read())
            image_path = str(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    if not image_path:
        raise HTTPException(400, "image_path or file required")
    if not Path(image_path).exists():
        raise HTTPException(404, f"image not found: {image_path}")

    try:
        return demo_service.run_demo(
            category=category,
            image_path=image_path,
            live_cloud=live,
            use_route_agent=use_agent,
            edge_node_id=edge_node_id,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        if "cloud model unavailable" in str(exc).lower() or "unavailable" in str(exc).lower():
            raise HTTPException(503, f"cloud model unavailable: {exc}") from exc
        raise
