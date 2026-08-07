"""Web package paths and YAML helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC_DIR = WEB_DIR / "static"
DATA_ROOT = ROOT / "datasets" / "mvtec"
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_default_yaml() -> dict[str, Any]:
    if DEFAULT_CFG.exists():
        return yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8")) or {}
    return {}


def load_default_collab() -> dict[str, Any]:
    return dict(load_default_yaml().get("collab") or {})


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))
