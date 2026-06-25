#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_hot_env.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/runpod_stage_hot_storage.sh [options]

Options:
  --model NAME       Copy /workspace/models/NAME to $COT_SAFETY_MODEL_ROOT/NAME.
  --judge NAME       Copy /workspace/models/judges/NAME to $COT_SAFETY_JUDGE_ROOT/NAME.
  --output PATH      Copy /workspace/outputs/PATH to $COT_SAFETY_OUTPUT_ROOT/PATH.
  --data PATH        Copy /workspace/data/PATH to $COT_SAFETY_DATA_ROOT/PATH.
  --check-only       Print storage mapping and filesystem types only.
  --print-env        Print exports for the active hot runtime paths.

Example:
  bash pipelines/runpod_stage_hot_storage.sh \
    --model DeepSeek-R1-Distill-Llama-8B \
    --judge Llama-Guard-3-8B \
    --judge HarmBench-Llama-2-13b-cls \
    --output deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun/checkpoint-500 \
    --data model_comparison_eval/deepseek_8b_stage2
USAGE
}

fs_type() {
  local path="$1"
  if command -v findmnt >/dev/null 2>&1; then
    findmnt -T "${path}" -no FSTYPE 2>/dev/null || true
  fi
}

print_env() {
  cat <<EOF
export COT_SAFETY_COLD_ROOT="${COT_SAFETY_COLD_ROOT}"
export COT_SAFETY_HOT_ROOT="${COT_SAFETY_HOT_ROOT:-}"
export COT_SAFETY_MODEL_ROOT="${COT_SAFETY_MODEL_ROOT}"
export COT_SAFETY_JUDGE_ROOT="${COT_SAFETY_JUDGE_ROOT}"
export COT_SAFETY_DATA_ROOT="${COT_SAFETY_DATA_ROOT}"
export COT_SAFETY_OUTPUT_ROOT="${COT_SAFETY_OUTPUT_ROOT}"
export COT_SAFETY_RUN_ROOT="${COT_SAFETY_RUN_ROOT}"
export HF_HOME="${HF_HOME}"
EOF
}

print_check() {
  echo "Cold root: ${COT_SAFETY_COLD_ROOT} ($(fs_type "${COT_SAFETY_COLD_ROOT}"))"
  echo "Hot root:  ${COT_SAFETY_HOT_ROOT:-disabled} ($(fs_type "${COT_SAFETY_HOT_ROOT:-${COT_SAFETY_COLD_ROOT}}"))"
  echo "Model root: ${COT_SAFETY_MODEL_ROOT} ($(fs_type "${COT_SAFETY_MODEL_ROOT}"))"
  echo "Judge root: ${COT_SAFETY_JUDGE_ROOT} ($(fs_type "${COT_SAFETY_JUDGE_ROOT}"))"
  echo "Data root:  ${COT_SAFETY_DATA_ROOT} ($(fs_type "${COT_SAFETY_DATA_ROOT}"))"
  echo "Output root: ${COT_SAFETY_OUTPUT_ROOT} ($(fs_type "${COT_SAFETY_OUTPUT_ROOT}"))"
  echo "Run root:   ${COT_SAFETY_RUN_ROOT} ($(fs_type "${COT_SAFETY_RUN_ROOT}"))"
  df -h "${COT_SAFETY_HOT_ROOT:-${COT_SAFETY_COLD_ROOT}}" "${COT_SAFETY_COLD_ROOT}" 2>/dev/null || true
}

copy_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "missing source: ${src}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  local src_real dst_real
  src_real="$(readlink -f "${src}")"
  dst_real="$(readlink -m "${dst}")"
  if [[ "${src_real}" == "${dst_real}" ]]; then
    echo "already staged: ${src} -> ${dst}"
    return 0
  fi

  echo "staging: ${src} -> ${dst}"
  if command -v rsync >/dev/null 2>&1; then
    if [[ -d "${src}" ]]; then
      mkdir -p "${dst}"
      rsync -a --info=progress2 "${src}/" "${dst}/"
    else
      rsync -a --info=progress2 "${src}" "${dst}"
    fi
  else
    cp -a "${src}" "${dst}"
  fi
}

if [[ "$#" -eq 0 ]]; then
  usage
  print_check
  exit 0
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      shift
      copy_path "${COT_SAFETY_COLD_ROOT}/models/$1" "${COT_SAFETY_MODEL_ROOT}/$1"
      ;;
    --judge)
      shift
      copy_path "${COT_SAFETY_COLD_ROOT}/models/judges/$1" "${COT_SAFETY_JUDGE_ROOT}/$1"
      ;;
    --output)
      shift
      copy_path "${COT_SAFETY_COLD_ROOT}/outputs/$1" "${COT_SAFETY_OUTPUT_ROOT}/$1"
      ;;
    --data)
      shift
      copy_path "${COT_SAFETY_COLD_ROOT}/data/$1" "${COT_SAFETY_DATA_ROOT}/$1"
      ;;
    --check-only)
      print_check
      ;;
    --print-env)
      print_env
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

echo "Hot storage staging complete."
print_check
