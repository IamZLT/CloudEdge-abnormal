from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class DetectionResult:
    label: str
    confidence: float
    defect_category: Optional[str] = None
    bbox: Optional[Dict[str, int]] = None
    timestamp: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskMeta:
    device_id: str
    image_id: str
    priority: int = 0
    upload_reason: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class UploadDecision:
    should_upload: bool
    reason: str
    threshold: float
    score: Optional[float] = None
    policy: str = "image_quality"
