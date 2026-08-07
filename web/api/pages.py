"""HTML shell."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from web.config import STATIC_DIR

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )
