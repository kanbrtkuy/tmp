#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
CONFIG="${CONFIG:-configs/experiment/stage1_positionscan_8b_4xa100.yaml}"
PYTHON="${PYTHON:-python}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

cd "${ROOT}"
mkdir -p "${COT_SAFETY_RUN_ROOT}" logs

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" scripts/smoke_test.py

EXTRA_ARGS=()
if [[ -n "${MAX_PER_SOURCE:-}" ]]; then
  EXTRA_ARGS+=(--max_per_source "${MAX_PER_SOURCE}")
fi
if [[ "${SKIP_DATA_PREP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_data_prep)
fi
if [[ "${SKIP_HIDDEN_EXTRACTION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_hidden_extraction)
fi
if [[ "${SKIP_SINGLE_SCAN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_single_scan)
fi
if [[ "${SKIP_MULTILAYER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_multilayer)
fi
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" scripts/run_stage1_positionscan.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
