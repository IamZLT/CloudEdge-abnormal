import json

import pytest

from src.vlm import detect_vlm_backend


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3_vl", "qwen3_vl"),
        ("qwen3_5", "transformers"),
        ("internvl_chat", "internvl"),
        ("minicpmv", "minicpm"),
    ],
)
def test_detect_vlm_backend(tmp_path, model_type, expected):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert detect_vlm_backend(tmp_path) == expected


def test_detect_vlm_backend_rejects_unknown(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown"}))
    with pytest.raises(ValueError, match="unsupported VLM"):
        detect_vlm_backend(tmp_path)
