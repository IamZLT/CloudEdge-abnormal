import base64
import json
import mimetypes
import re
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict

from common.schemas import DetectionResult
from common.utils import current_timestamp


SYSTEM_PROMPT = """
You are an expert in industrial visual defect inspection.

Inspect the provided image and determine whether it is normal or contains an anomaly.

Requirements:
- Set "label" to "normal" if no visible anomaly is found.
- Set "label" to "anomaly" if a visible anomaly is found.
- If the label is "anomaly", provide a concise and specific anomaly category
  in "defect category", such as "scratch", "crack", "dent", "stain",
  "hole", "missing part", "misalignment", or "foreign object".
- If the exact anomaly category cannot be identified, use "unknown".
- If the label is "normal", set "defect category" to null.
- "confidence" must be a number between 0 and 1.
- If the label is "anomaly", use "bbox" to locate the anomaly in the original
  image with pixel coordinates. Here, "x1" and "y1" are the top-left coordinates,
  while "x2" and "y2" are the bottom-right coordinates.
- If the label is "normal", set "bbox" to null.
- Keep "metadata.reason" concise and based on visible evidence.
- Do not output Markdown or any text outside the JSON object.
- Return exactly one JSON object.

The output should use this JSON schema:
{
  "label": "normal or anomaly",
  "defect category": "scratch, crack, dent, stain, hole, missing part, misalignment, foreign object, unknown, or null",
  "confidence": 0.91,
  "bbox": {
    "x1": 325,
    "y1": 184,
    "x2": 451,
    "y2": 232
  },
  "metadata": {
    "reason": "Concise visible evidence."
  }
}
"""


class LargeModelAPI:
    def __init__(self, cloud_client, model: str, max_tokens: int = 512, temperature: float = 0.1):
        self.cloud_client = cloud_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _image_bytes_data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return LargeModelAPI._image_bytes_data_url(path.read_bytes(), mime_type)

    def prepare_payload_from_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Inspect this industrial image and return the assessment "
                                "in the required JSON format."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_bytes_data_url(image_bytes, mime_type)
                            },
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def prepare_payload(self, result: DetectionResult, image_path: str) -> Dict[str, Any]:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return self.prepare_payload_from_image_bytes(path.read_bytes(), mime_type)

    @staticmethod
    def _repair_bbox_shorthand(text: str) -> str:
        """兼容模型偶尔返回的 {"x1": x1, y1, x2, y2} 非法 JSON。"""
        number = r"-?\d+(?:\.\d+)?"
        pattern = re.compile(
            rf'"bbox"\s*:\s*\{{\s*"x1"\s*:\s*({number})\s*,\s*'
            rf"({number})\s*,\s*({number})\s*,\s*({number})\s*\}}"
        )

        def replace(match: re.Match) -> str:
            x1, y1, x2, y2 = (float(value) for value in match.groups())
            bbox = {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
            return f'"bbox":{json.dumps(bbox, separators=(",", ":"))}'

        return pattern.sub(replace, text)

    @staticmethod
    def _extract_prediction(response: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("云端响应缺少 choices[0].message.content") from exc

        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))

        # 视觉模型偶尔会先复述示例 JSON，再输出最终结果。逐个解析对象，
        # 避免贪婪匹配把多个对象拼在一起导致解析失败。
        text = LargeModelAPI._repair_bbox_shorthand(str(content))
        decoder = json.JSONDecoder()
        predictions = []
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "label" in candidate:
                predictions.append(candidate)

        if not predictions:
            raise ValueError(f"云端模型未返回有效检测 JSON：{text[:300]}")
        return predictions[-1]

    def parse_response(self, response: Dict[str, Any]) -> DetectionResult:
        prediction = self._extract_prediction(response)
        label = str(prediction.get("label", "unknown")).strip().lower()
        label = {"ok": "normal", "defect": "anomaly"}.get(label, label)
        if label not in {"normal", "anomaly"}:
            raise ValueError(
                f"云端模型返回了不支持的 label：{label!r}，只允许 'normal' 或 'anomaly'"
            )
        defect_category = prediction.get("defect_category", prediction.get("defect category"))
        return DetectionResult(
            label=label,
            confidence=float(prediction.get("confidence", 0.0)),
            defect_category=None if defect_category is None else str(defect_category),
            bbox=prediction.get("bbox"),
            timestamp=current_timestamp(),
            source="cloud",
            metadata=prediction.get("metadata", {}),
        )

    def call_large_model_bytes(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> DetectionResult:
        call_started_ns = perf_counter_ns()
        payload_started_ns = perf_counter_ns()
        payload = self.prepare_payload_from_image_bytes(image_bytes, mime_type)
        payload_encode_ms = (perf_counter_ns() - payload_started_ns) / 1_000_000
        response = self.cloud_client.post(payload)
        cloud_result = self.parse_response(response)
        transport = dict(getattr(self.cloud_client, "last_call_metrics", {}) or {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        cloud_metrics = {
            "original_image_bytes": len(image_bytes),
            "payload_encode_ms": payload_encode_ms,
            "request_body_bytes": transport.get("request_body_bytes"),
            "response_body_bytes": transport.get("response_body_bytes"),
            "http_round_trip_ms": transport.get("http_round_trip_ms"),
            "http_status": transport.get("http_status"),
            "cloud_total_ms": (perf_counter_ns() - call_started_ns) / 1_000_000,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "gpu": transport.get("gpu"),
        }
        metadata = dict(cloud_result.metadata or {})
        metadata["cloud_metrics"] = cloud_metrics
        cloud_result.metadata = metadata
        return cloud_result

    def call_large_model(self, result: DetectionResult, image_path: str) -> DetectionResult:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return self.call_large_model_bytes(path.read_bytes(), mime_type)
