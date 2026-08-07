#!/usr/bin/env python3
"""CloudEdge-abnormal Web console (app factory).

Package layout:
  web/
    app.py              # create_app() — uvicorn entry: web.app:app
    config.py           # paths + yaml
    api/                # FastAPI routers
    services/           # business logic (fleet, models, catalog, demo)
    static/             # frontend assets

Run:
  CUDA_VISIBLE_DEVICES=0 uvicorn web.app:app --host 0.0.0.0 --port 7860
  # or
  python -m web
"""
from __future__ import annotations

import os
import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.api import register_routers
from web.config import STATIC_DIR, load_default_collab
from web.services import fleet_service, model_service


def create_app() -> FastAPI:
    app = FastAPI(title="CloudEdge Defect Console", version="0.2.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_routers(app)

    @app.on_event("startup")
    def _on_startup() -> None:
        from src.collab_routing import configure_routing, list_policies

        collab = load_default_collab()
        router = configure_routing(collab)
        print(
            f"[web] collab route_policy={router.name} policies={list_policies()}",
            flush=True,
        )
        fleet_service.init_network_state()
        fleet_service.start_net_sampler()
        preload = os.environ.get("WEB_PRELOAD", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if preload:
            threading.Thread(
                target=model_service.preload_models, name="model-preload", daemon=True
            ).start()
        else:
            print("[web] model preload skipped (WEB_PRELOAD=0)", flush=True)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "7860"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
