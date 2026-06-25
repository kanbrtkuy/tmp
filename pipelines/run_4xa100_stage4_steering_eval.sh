#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
LEGACY_ROOT="${LEGACY_ROOT:-${ROOT}/legacy/PauseProbe}"
PYTHON="${PYTHON:-python}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

export DEVICES="${DEVICES:-0,1,2,3}"
export MODEL="${MODEL:-${COT_SAFETY_OUTPUT_ROOT}/deepseek_8b_intra_pause_cot4_trusted_cot_18k/final}"
export DELTA="${DELTA:-${LEGACY_ROOT}/runs/steering/intra_pause_learned_delta_8b/zero_l16_steps80/learned_delta.pt}"
export OUT_ROOT="${OUT_ROOT:-${COT_SAFETY_RUN_ROOT}/steering/intra_pause_full_steering_eval_8b_4xa100}"
export LAYER="${LAYER:-16}"

ROOT="${LEGACY_ROOT}" PYTHON="${PYTHON}" bash "${LEGACY_ROOT}/scripts/steering/run_intra_pause_full_steering_eval.sh"
