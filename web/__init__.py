"""CloudEdge web console package.

Prefer importing the ASGI app via ``web.app:app``.
"""
from web.app import app, create_app

__all__ = ["app", "create_app"]
