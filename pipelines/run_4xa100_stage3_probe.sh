#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LEGACY_ROOT="${LEGACY_ROOT:-${ROOT}/legacy/PauseProbe}"
CONFIG="${CONFIG:-configs/experiment/stage3_intra_pause_probe_8b_4xa100.yaml}"
PYTHON="${PYTHON:-python}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_stage3_env.sh"

cd "${ROOT}"
mkdir -p "${COT_SAFETY_RUN_ROOT}" logs

EXTRA_ARGS=()
if [[ "${SKIP_BASE_DATA_PREP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_base_data_prep)
fi
if [[ "${SKIP_INTRA_DATA_PREP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_intra_data_prep)
fi
if [[ "${SKIP_HIDDEN_EXTRACTION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_hidden_extraction)
fi
if [[ "${SKIP_SINGLE_SCAN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_single_scan)
fi
if [[ "${SKIP_POOLED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_pooled)
fi
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" scripts/run_stage3_intra_pause_probe.py \
  --config "${CONFIG}" \
  --legacy-root "${LEGACY_ROOT}" \
  --python "${PYTHON}" \
  "${EXTRA_ARGS[@]}"
