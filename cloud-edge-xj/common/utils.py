import json
from datetime import datetime, timezone
from typing import Any, Dict


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_info(message: str, data: Dict[str, Any] = None) -> None:
    if data is None:
        data = {}
    print(f"[INFO] {current_timestamp()} - {message} - {json.dumps(data, ensure_ascii=False)}")


def serialize_result(result: Any) -> str:
    try:
        return json.dumps(result, default=lambda o: o.__dict__, ensure_ascii=False)
    except TypeError:
        return str(result)
