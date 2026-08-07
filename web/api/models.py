"""Model preload endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from web.services import model_service

router = APIRouter(tags=["models"])


@router.post("/api/route_agent/load")
def api_route_agent_load():
    try:
        model_service.get_route_agent()
        info = model_service.route_agent_info()
        return {"ok": True, "loaded": True, **info}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": str(exc), **model_service.route_agent_info()},
            status_code=503,
        )


@router.post("/api/cloud/load")
def api_cloud_load():
    try:
        model_service.get_cloud_client()
        return {"ok": True, "loaded": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
