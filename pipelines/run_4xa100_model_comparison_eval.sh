#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
CONFIG="${CONFIG:-configs/experiment/stage2_model_comparison_eval_8b_4xa100.yaml}"
PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-python3}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

"${PYTHON}" scripts/run_model_comparison_eval.py \
  --config "${CONFIG}" \
  --phase "${PHASE}" \
  "$@"
