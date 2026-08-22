#!/usr/bin/env bash
set -euo pipefail

MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
MODEL_ROOT="${MODEL_ROOT:-model_card}"

download() {
  local repo_id="$1"
  local directory="$2"
  echo "[$(date '+%F %T')] downloading ${repo_id} -> ${MODEL_ROOT}/${directory}"
  "${MODELSCOPE_BIN}" download "${repo_id}" \
    --local-dir "${MODEL_ROOT}/${directory}" \
    --max-workers 4
  echo "[$(date '+%F %T')] completed ${repo_id}"
}

mkdir -p "${MODEL_ROOT}"

# Default pilot models first, then the larger comparison models.
download Qwen/Qwen3.5-2B Qwen3.5-2B
download Qwen/Qwen3.5-4B Qwen3.5-4B
download OpenBMB/MiniCPM-V-4_5-int4 MiniCPM-V-4_5-int4
download Qwen/Qwen3.5-9B Qwen3.5-9B
download OpenGVLab/InternVL3_5-8B InternVL3_5-8B
