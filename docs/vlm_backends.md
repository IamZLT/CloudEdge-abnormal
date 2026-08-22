# Unified VLM backends

All supported vision-language models expose the same entry point and return
`VLMResult` (`decision`, `confidence`, `defect_type`, `reason`, `latency_ms`,
`parse_ok`, `peak_mem_mb`). This keeps the detection metrics comparable even
though each upstream model has a different inference API.

```python
from src.vlm import create_vlm_client

client = create_vlm_client(
    model_path="model_card/Qwen3.5-4B",
    backend="auto",
    device="cuda:0",
    dtype="bfloat16",
    role="cloud",
)
result = client.infer("example.png")
print(result.to_dict())
```

`backend="auto"` reads the checkpoint's `config.json`. Supported backends are:

| Checkpoint family | Backend |
|---|---|
| Qwen3-VL | `qwen3_vl` |
| Qwen3.5 | `transformers` |
| InternVL3.5 | `internvl` |
| MiniCPM-V 4.5 | `minicpm` |

The first-priority checkpoint paths are recorded in `configs/vlm_models.yaml`.
The existing benchmark accepts model overrides without editing its config:

```bash
conda activate cloudedge
CUDA_VISIBLE_DEVICES=0,1 python scripts/bench_qwen_vl.py \
  --cloud-model model_card/Qwen3.5-4B \
  --cloud-backend auto --max-images 20
```

For a fair comparison, keep the prompt, dataset subset, decoding parameters,
and evaluation threshold fixed. Qwen3.5 and MiniCPM run with thinking disabled
and deterministic decoding by default so that the short JSON response fits the
benchmark budget.
