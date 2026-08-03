from __future__ import annotations

import json
import re
from typing import Any


def parse_vlm_json(text: str) -> dict[str, Any]:
    """Extract decision JSON from VLM free-form text."""
    raw = (text or "").strip()
    if not raw:
        return {
            "decision": "OK",
            "confidence": 0.0,
            "defect_type": "none",
            "reason": "empty_response",
            "parse_ok": False,
        }

    # strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if fence:
        raw = fence.group(1)

    # first {...} block
    m = re.search(r"\{.*\}", raw, flags=re.S)
    payload: dict[str, Any] = {}
    parse_ok = False
    if m:
        try:
            payload = json.loads(m.group(0))
            parse_ok = True
        except json.JSONDecodeError:
            payload = {}

    decision = str(payload.get("decision", "")).strip().upper()
    if decision not in {"OK", "NG"}:
        # heuristic fallback
        low = raw.lower()
        if any(k in low for k in ("ng", "defect", "anomaly", "broken", "crack", "contamination")):
            decision = "NG"
        elif "ok" in low or "normal" in low or "good" in low:
            decision = "OK"
        else:
            decision = "OK"

    conf = payload.get("confidence", None)
    try:
        confidence = float(conf) if conf is not None else (0.6 if parse_ok else 0.3)
    except (TypeError, ValueError):
        confidence = 0.3
    confidence = max(0.0, min(1.0, confidence))

    defect_type = str(payload.get("defect_type", "none") or "none")
    reason = str(payload.get("reason", "") or "")

    return {
        "decision": decision,
        "confidence": confidence,
        "defect_type": defect_type,
        "reason": reason,
        "parse_ok": parse_ok,
        "raw": text,
    }
