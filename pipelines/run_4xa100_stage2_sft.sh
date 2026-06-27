#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-configs/experiment/stage2_intra_pause_format_only_8b_cot4_save50_max250_4xa100.yaml}"
PYTHON="${PYTHON:-python}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

cd "${ROOT}"
mkdir -p "${COT_SAFETY_RUN_ROOT}" logs

EXTRA_ARGS=()
if [[ "${SKIP_DATA_PREP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_data_prep)
fi
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_train)
fi
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" scripts/run_stage2_sft.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
