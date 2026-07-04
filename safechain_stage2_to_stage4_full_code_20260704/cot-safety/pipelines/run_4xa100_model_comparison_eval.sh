#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt250_3xidle.yaml}"
PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-python3}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_base_env.sh"

cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

"${PYTHON}" scripts/run_model_comparison_eval.py \
  --config "${CONFIG}" \
  --phase "${PHASE}" \
  "$@"
