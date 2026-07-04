#!/usr/bin/env bash
# RunPod environment for Stage 1 PositionScan / hidden extraction / probes.
#
# Stage 1 uses base inference/probe dependencies and should not require the
# Stage 2 SFT stack such as TRL or bitsandbytes.

if [[ -n "${COT_SAFETY_STAGE1_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export COT_SAFETY_STAGE1_ENV_SOURCED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runpod_base_env.sh"

export COT_SAFETY_STAGE="stage1"
