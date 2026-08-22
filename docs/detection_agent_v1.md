# PatchCore + Qwen3.5-9B Detection Agent v1

## Objective

Use PatchCore-DINOv3 as the stable anomaly expert and invoke Qwen3.5-9B only
for a fixed budget of uncertain samples.  The VLM receives localized and
retrieval-augmented evidence, while conservative guardrails prevent arbitrary
overrides of confident PatchCore decisions.

## Pipeline

1. Fit PatchCore only on normal training images.
2. Hold out 20% of normal training images for the 99th-percentile threshold.
3. Produce image score, patch anomaly map, top local regions and nearest normal reference.
4. Route the 30% most uncertain/diffuse samples without using test labels.
5. Build a four-panel evidence board: query, heatmap, local crop and normal reference.
6. Ask Qwen3.5-9B for structured OK/NG review and spatial agreement.
7. Apply asymmetric, bounded fusion only when parsing, confidence and region checks pass.
8. Fall back to PatchCore on any review failure.

## Code map

- `src/models.py`: PatchCore expert evidence and normal-reference retrieval.
- `src/detection_agent/schemas.py`: stable Agent contracts.
- `src/detection_agent/evidence_builder.py`: localization and evidence boards.
- `src/detection_agent/router.py`: label-free budget router.
- `src/detection_agent/vlm_reviewer.py`: Qwen review prompt and parsing.
- `src/detection_agent/fusion.py`: conservative fusion guardrails.
- `src/detection_agent/pipeline.py`: final decision pipeline.
- `scripts/analyze_agent_headroom.py`: offline complementarity and Oracle diagnostics.
- `scripts/bench_detection_agent.py`: resumable prepare/review/evaluate benchmark.
- `configs/agent_patchcore_qwen9b.yaml`: frozen v1 experiment configuration.

## Reproduction on server157

```bash
cd ~/projects/cloud_edge/code/CloudEdge-abnormal

# Diagnostic replay from existing results (no GPU)
PYTHONPATH=. ~/miniconda3/envs/cloudedge/bin/python \
  scripts/analyze_agent_headroom.py

# Bottle smoke test
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. ~/miniconda3/envs/cloudedge/bin/python \
  scripts/bench_detection_agent.py \
  --categories bottle --phase all --device cuda:0 \
  --out outputs/detection_agent_bottle --resume

# Full MVTec, resumable
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. ~/miniconda3/envs/cloudedge/bin/python \
  scripts/bench_detection_agent.py \
  --categories all --phase all --device cuda:0 \
  --out outputs/detection_agent_mvtec --resume
```

## MVTec result

All numbers are macro averages over 15 categories and 1,725 test images.

| Metric | PatchCore-DINOv3 | Agent v1 | Delta |
|---|---:|---:|---:|
| AUROC | 0.9466 | 0.9434 | -0.0032 |
| AUPRC | 0.9716 | 0.9701 | -0.0015 |
| F1 | 0.8570 | 0.8970 | +0.0400 |
| Recall | 0.8031 | 0.8877 | +0.0847 |

Agent diagnostics:

- Review budget: 30.0% (517/1,725 images).
- Reviews passing fusion guardrails: 497.
- Changed decisions: 166.
- Corrected PatchCore errors: 113.
- Harmful overrides: 53.
- Override precision: 68.1%.
- Mean end-to-end latency: 1,866.4 ms/image.
- P95 end-to-end latency: 6,715.4 ms/image.
- Peak VLM memory: 18,148.3 MiB.
- Inference failures: 0.

The v1 Agent materially improves the operating point (F1 and recall) while
slightly reducing ranking metrics.  It is therefore a successful first
prototype, but not yet a Pareto improvement over PatchCore.

## Validity constraints

- MVTec test labels are never used for threshold fitting, routing or fusion.
- Oracle numbers from `analyze_agent_headroom.py` are diagnostic only.
- Fusion hyperparameters are frozen in YAML before the live full run.
- Every VLM response and final per-image decision is persisted for auditing.

## Next iteration

1. Learn a risk gate on synthetic anomalies or ViSA, not MVTec test labels.
2. Suppress OK overrides when PatchCore heatmaps are sharply localized.
3. Add review self-consistency only for VLM/PatchCore conflicts.
4. Calibrate VLM confidence and evaluate 10%, 20% and 30% review budgets.
5. Reproduce WideResNet50-2 PatchCore under the same protocol as the stronger expert baseline.
