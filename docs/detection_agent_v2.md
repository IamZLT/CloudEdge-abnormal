# Detection Agent v2: conflict-only verification

## Motivation

Agent v1 reviewed 30% of MVTec samples and changed 166 decisions. It corrected
113 PatchCore errors but introduced 53 errors. Of those changes, 159 were
`OK -> NG`, so the main failure mode was false-alarm amplification rather than
missed-defect suppression. First-review confidence did not separate good and bad
overrides (approximately 0.947 versus 0.950), so confidence thresholding alone is
not an adequate guard.

## Policy

1. PatchCore + DINOv3 remains the primary expert.
2. Agent v1 performs budgeted evidence review unchanged.
3. If v1 does not reverse PatchCore, v2 preserves PatchCore's score and decision.
4. If v1 reverses PatchCore, the same Qwen3.5-9B model receives a new skeptical
   verification prompt and the existing four-panel evidence board.
5. An override is accepted only when the verifier explicitly confirms it, agrees
   on the decision and region, parses successfully, and reports confidence >= 0.90.
6. Every failure or abstention falls back to PatchCore.

This produces a fail-safe Agent: VLM output can affect the final result only after
two role-separated passes agree on a concrete localized defect.

## Commands

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/bench_detection_agent_v2.py \
  --categories bottle --phase all --device cuda:0 \
  --out outputs/detection_agent_v2_bottle --resume

CUDA_VISIBLE_DEVICES=6 python scripts/bench_detection_agent_v2.py \
  --categories all --phase all --device cuda:0 \
  --out outputs/detection_agent_v2_mvtec --resume
```

The script reuses v1 evidence and writes one JSON file per verification, so it is
safe to resume. It never reloads or reruns PatchCore.

## Evaluation caveat

The v2 design followed retrospective inspection of the completed MVTec v1 run.
Consequently, the v2 MVTec result is an exploratory development result, not an
unbiased validation result. Freeze the policy and validate it unchanged on ViSA,
Real-IAD, or a predefined held-out category split before making a generalization
claim.
