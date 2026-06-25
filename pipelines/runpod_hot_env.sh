#!/usr/bin/env bash
# Source this file from RunPod pipeline wrappers to keep hot runtime I/O off
# /workspace network/FUSE volumes. Persistent storage still lives under
# COT_SAFETY_COLD_ROOT and can be synced back after a run.

if [[ -n "${COT_SAFETY_HOT_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export COT_SAFETY_HOT_ENV_SOURCED=1

COT_SAFETY_HF_ENV_FILE="${COT_SAFETY_HF_ENV_FILE:-/workspace/secrets/hf.env}"
if [[ -f "${COT_SAFETY_HF_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${COT_SAFETY_HF_ENV_FILE}"
fi

export COT_SAFETY_COLD_ROOT="${COT_SAFETY_COLD_ROOT:-/workspace}"
export COT_SAFETY_USE_HOT_STORAGE="${COT_SAFETY_USE_HOT_STORAGE:-1}"

if [[ "${COT_SAFETY_USE_HOT_STORAGE}" == "1" ]]; then
  if [[ -z "${COT_SAFETY_HOT_ROOT:-}" ]]; then
    if [[ -d /dev/shm && -w /dev/shm ]]; then
      export COT_SAFETY_HOT_ROOT="/dev/shm/cot-safety-hot"
    else
      export COT_SAFETY_HOT_ROOT="/tmp/cot-safety-hot"
    fi
  else
    export COT_SAFETY_HOT_ROOT
  fi

  mkdir -p \
    "${COT_SAFETY_HOT_ROOT}/models/judges" \
    "${COT_SAFETY_HOT_ROOT}/data" \
    "${COT_SAFETY_HOT_ROOT}/outputs" \
    "${COT_SAFETY_HOT_ROOT}/runs" \
    "${COT_SAFETY_HOT_ROOT}/hf_cache" \
    "${COT_SAFETY_HOT_ROOT}/tmp"

  export COT_SAFETY_MODEL_ROOT="${COT_SAFETY_MODEL_ROOT:-${COT_SAFETY_HOT_ROOT}/models}"
  export COT_SAFETY_JUDGE_ROOT="${COT_SAFETY_JUDGE_ROOT:-${COT_SAFETY_HOT_ROOT}/models/judges}"
  export COT_SAFETY_DATA_ROOT="${COT_SAFETY_DATA_ROOT:-${COT_SAFETY_HOT_ROOT}/data}"
  export COT_SAFETY_OUTPUT_ROOT="${COT_SAFETY_OUTPUT_ROOT:-${COT_SAFETY_HOT_ROOT}/outputs}"
  export COT_SAFETY_RUN_ROOT="${COT_SAFETY_RUN_ROOT:-${COT_SAFETY_HOT_ROOT}/runs}"
  export HF_HOME="${COT_SAFETY_HF_HOME:-${COT_SAFETY_HOT_ROOT}/hf_cache}"
  export TMPDIR="${TMPDIR:-${COT_SAFETY_HOT_ROOT}/tmp}"
else
  export COT_SAFETY_MODEL_ROOT="${COT_SAFETY_MODEL_ROOT:-${COT_SAFETY_COLD_ROOT}/models}"
  export COT_SAFETY_JUDGE_ROOT="${COT_SAFETY_JUDGE_ROOT:-${COT_SAFETY_COLD_ROOT}/models/judges}"
  export COT_SAFETY_DATA_ROOT="${COT_SAFETY_DATA_ROOT:-${COT_SAFETY_COLD_ROOT}/data}"
  export COT_SAFETY_OUTPUT_ROOT="${COT_SAFETY_OUTPUT_ROOT:-${COT_SAFETY_COLD_ROOT}/outputs}"
  export COT_SAFETY_RUN_ROOT="${COT_SAFETY_RUN_ROOT:-${COT_SAFETY_COLD_ROOT}/cot-safety/runs}"
  export HF_HOME="${HF_HOME:-${COT_SAFETY_COLD_ROOT}/hf_cache}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v findmnt >/dev/null 2>&1; then
  for path in "${COT_SAFETY_MODEL_ROOT}" "${COT_SAFETY_JUDGE_ROOT}" "${COT_SAFETY_OUTPUT_ROOT}" "${COT_SAFETY_RUN_ROOT}"; do
    mkdir -p "${path}"
    fstype="$(findmnt -T "${path}" -no FSTYPE 2>/dev/null || true)"
    if [[ "${fstype}" == fuse* ]]; then
      printf 'warning: %s is on %s; stage models/checkpoints to COT_SAFETY_HOT_ROOT before running GPU jobs\n' "${path}" "${fstype}" >&2
    fi
  done
fi
