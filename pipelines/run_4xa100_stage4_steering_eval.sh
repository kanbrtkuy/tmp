#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LEGACY_ROOT="${LEGACY_ROOT:-${ROOT}/legacy/PauseProbe}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/experiment/stage4_pause_steering_8b_4xa100.yaml}"
PHASE="${PHASE:-eval}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

export DEVICES="${DEVICES:-0,1,2,3}"
export COT_SAFETY_LEGACY_ROOT="${COT_SAFETY_LEGACY_ROOT:-${LEGACY_ROOT}}"

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" "${ROOT}/scripts/run_stage4_steering.py" \
  --config "${CONFIG}" \
  --legacy-root "${LEGACY_ROOT}" \
  --python "${PYTHON}" \
  --phase "${PHASE}"
