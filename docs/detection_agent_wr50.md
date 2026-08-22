# WR50 PatchCore reproduction and Detection Agent optimization

## Outcome

The project now uses the canonical PatchCore WideResNet50-2 expert under the
same unified dataset, threshold, metric, latency and memory interface as the
other cloud methods. On all 15 MVTec categories (1,725 test images), the
label-free reproduction reaches macro AUROC 0.9774 and macro F1 0.9479.

The Agent experiments show that the current VLMs are useful for evidence and
audit, but are not yet reliable enough to override the stronger WR50 expert.
The deployable default is therefore `advisory_only`: Qwen and MiniCPM generate
structured semantic evidence, while WR50 retains the final score and label.

## Unified protocol

- Backbone: ImageNet-pretrained `wide_resnet50_2`.
- Features: `layer2` and `layer3`.
- Coreset: Anomalib k-center greedy, ratio 0.1.
- Input: 224 x 224.
- Split: 80% of normal training images form the PatchCore gallery; 20% are
  held out only for calibration.
- Threshold: 99th percentile of held-out normal scores.
- Test labels are not used for thresholding, routing, prompting or fusion.
- Evaluation: macro average over the 15 MVTec categories.

The older Anomalib result (AUROC 0.9829, F1 0.9741) uses a different evaluation
protocol, including a test-derived adaptive F1 threshold, and should not be
mixed directly with this unified result. As a reproduction check, the unified
bottle result is AUROC 1.0000 and F1 0.9921, matching the historical bottle F1
of approximately 0.9920.

## Agent design

1. WR50 PatchCore produces the anomaly score, 28 x 28 patch evidence, candidate
   regions and nearest normal reference.
2. A label-free router selects the most uncertain 10% of test samples.
3. Qwen3.5-9B reviews a four-panel evidence board: query, heatmap, local crop and
   nearest normal reference.
4. Only Qwen/WR50 decision conflicts enter a second verifier.
5. Two verifier variants were measured: Qwen self-verification and heterogeneous
   MiniCPM-V-4.5-int4 verification.
6. The default advisory policy preserves WR50 on every conflict and retains all
   VLM outputs for audit or human review.

## Full MVTec results

| Version | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Mean latency (ms/image) | Peak memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WR50 PatchCore | 0.977449 | 0.992112 | 0.947930 | 0.964397 | 0.937341 | 0.925953 | 4.93 | 1,886.34 |
| Qwen v1, 10% review | 0.963312 | 0.980640 | 0.933306 | 0.922247 | 0.952042 | 0.903949 | 630.11 | 18,148.55 |
| Qwen self-verifier | 0.973289 | 0.988959 | 0.941382 | 0.941317 | 0.947683 | 0.916020 | 952.65 | 18,185.91 |
| Qwen + MiniCPM automatic verifier | 0.977362 | 0.992089 | 0.947663 | 0.963877 | 0.937341 | 0.925512 | 1,154.09 | 18,148.55 |
| **Qwen + MiniCPM advisory (default)** | **0.977449** | **0.992112** | **0.947930** | **0.964397** | **0.937341** | **0.925953** | **1,154.09** | **18,148.55** |

Agent override audit:

| Version | First reviews | Conflicts | Accepted overrides | Corrected | Harmful | Override precision |
|---|---:|---:|---:|---:|---:|---:|
| Qwen v1 | 173 | 89 | 89 | 30 | 59 | 33.71% |
| Qwen self-verifier | 173 | 89 | 49 | 18 | 31 | 36.73% |
| Qwen + MiniCPM automatic | 173 | 89 | 1 | 0 | 1 | 0.00% |
| Qwen + MiniCPM advisory | 173 | 89 | 0 | 0 | 0 | n/a |

The automatic Agent does **not** outperform WR50. This negative result is kept
explicitly rather than tuning a gate on MVTec test labels. The heterogeneous
verifier is effective at abstaining, but the one accepted override is still a
false alarm. Automatic overrides should remain disabled until a frozen policy
improves results on an external validation dataset such as ViSA.

## Code and commands

- `src/wr50_patchcore.py`: Anomalib-compatible WR50 expert and evidence API.
- `src/detection_agent/conflict_verifier.py`: second-pass verifier and advisory gate.
- `scripts/bench_detection_branches.py`: unified WR50 benchmark.
- `scripts/bench_detection_agent.py`: Qwen first-pass Agent.
- `scripts/bench_detection_agent_v2.py`: self/heterogeneous verification and advisory evaluation.
- `configs/mvtec_patchcore_wr50.yaml`: WR50 baseline.
- `configs/agent_patchcore_wr50_qwen9b.yaml`: 10% Qwen first pass.
- `configs/agent_patchcore_wr50_qwen9b_v2.yaml`: Qwen self-verifier.
- `configs/agent_patchcore_wr50_qwen9b_minicpm_v2.yaml`: automatic heterogeneous verifier.
- `configs/agent_patchcore_wr50_qwen9b_minicpm_advisory.yaml`: deployable default.

```bash
cd ~/projects/cloud_edge/code/CloudEdge-abnormal

CUDA_VISIBLE_DEVICES=6 ~/miniconda3/envs/cloudedge/bin/python \
  scripts/bench_detection_branches.py \
  --config configs/mvtec_patchcore_wr50.yaml \
  --datasets mvtec --models patchcore_wr50 \
  --device cuda:0 --out outputs/mvtec_patchcore_wr50_unified

CUDA_VISIBLE_DEVICES=6 ~/miniconda3/envs/cloudedge/bin/python \
  scripts/bench_detection_agent.py \
  --config configs/agent_patchcore_wr50_qwen9b.yaml \
  --categories all --phase all --device cuda:0 \
  --out outputs/detection_agent_wr50_b10_mvtec --resume

# Reuse completed verification JSON and evaluate the safe final policy.
CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/cloudedge/bin/python \
  scripts/bench_detection_agent_v2.py \
  --config configs/agent_patchcore_wr50_qwen9b_minicpm_advisory.yaml \
  --v1-root outputs/detection_agent_wr50_b10_mvtec \
  --categories all --phase evaluate --device cpu \
  --out outputs/detection_agent_wr50_b10_advisory_mvtec --resume
```

## Next valid optimization

Freeze a learnable risk gate using held-out normal images plus synthetic defects,
then validate it unchanged on ViSA before testing on MVTec again. Useful gate
features are WR50 margin, patch concentration, affected area, nearest-normal
similarity, first/second reviewer agreement and defect localization consistency.
This is the shortest defensible route to an Agent that may exceed WR50 without
test-set threshold tuning.
