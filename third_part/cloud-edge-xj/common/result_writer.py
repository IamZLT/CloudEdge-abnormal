import json
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from common.schemas import DetectionResult
from common.utils import current_timestamp


class JsonlResultWriter:
    def __init__(self, output_path: str, append: bool = False):
        path = Path(output_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a" if append else "w", encoding="utf-8")

    def _write(self, record: Dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def write_success(
        self,
        dataset: str,
        image_path: str,
        result: DetectionResult,
        visualization_path: Optional[str] = None,
        visualization_error: Optional[str] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write(
            {
                "status": "success",
                "dataset": dataset,
                "image_path": image_path,
                "result": result.to_dict(),
                "visualization_path": visualization_path,
                "visualization_error": visualization_error,
                "evaluation": evaluation,
                "recorded_at": current_timestamp(),
            }
        )

    def write_error(self, dataset: str, image_path: Optional[str], error: str) -> None:
        self._write(
            {
                "status": "error",
                "dataset": dataset,
                "image_path": image_path,
                "error": error,
                "recorded_at": current_timestamp(),
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlResultWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
