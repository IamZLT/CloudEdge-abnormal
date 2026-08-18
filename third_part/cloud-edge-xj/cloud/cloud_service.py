from common.schemas import DetectionResult
from cloud.model_api import LargeModelAPI


class CloudService:
    def __init__(self, model_api: LargeModelAPI):
        self.model_api = model_api

    def review_result(self, result: DetectionResult, image_path: str) -> DetectionResult:
        cloud_result = self.model_api.call_large_model(result, image_path)
        return cloud_result

    def inspect_image_bytes(self, image_bytes: bytes, mime_type: str) -> DetectionResult:
        return self.model_api.call_large_model_bytes(image_bytes, mime_type)
