"""HTTP routers for the CloudEdge web console."""
from __future__ import annotations

from fastapi import FastAPI

from web.api import catalog, demo, fleet, health, models, network, pages


def register_routers(app: FastAPI) -> None:
    app.include_router(pages.router)
    app.include_router(health.router)
    app.include_router(fleet.router)
    app.include_router(network.router)
    app.include_router(catalog.router)
    app.include_router(demo.router)
    app.include_router(models.router)
