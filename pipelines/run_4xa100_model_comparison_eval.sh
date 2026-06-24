#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
CONFIG="${CONFIG:-configs/experiment/stage2_model_comparison_eval_8b_4xa100.yaml}"
PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

if [[ -f /workspace/secrets/hf.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/secrets/hf.env
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" scripts/run_model_comparison_eval.py \
  --config "${CONFIG}" \
  --phase "${PHASE}" \
  "$@"
