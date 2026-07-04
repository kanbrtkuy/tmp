#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_base_env.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/runpod_sync_hot_to_cold.sh [options]

Options:
  --output PATH      Copy $COT_SAFETY_OUTPUT_ROOT/PATH to /workspace/outputs/PATH.
                     May be repeated.
  --data PATH        Copy $COT_SAFETY_DATA_ROOT/PATH to /workspace/data/PATH.
                     May be repeated.
  --run PATH         Copy $COT_SAFETY_RUN_ROOT/PATH to /workspace/cot-safety/runs/PATH.
                     May be repeated.
  --all-outputs      Copy all hot outputs to cold outputs.
  --all-runs         Copy all hot runs to cold run root.
  --delete           Delete files on cold side that no longer exist on hot side
                     when rsync is available.
  --remove-hot-after-sync
                     After an explicit --output PATH sync succeeds, remove that
                     hot output directory/file from $COT_SAFETY_OUTPUT_ROOT.
                     This is intended for completed checkpoints after they have
                     been persisted to /workspace.
  --check-only       Print hot/cold mapping and sizes.

Examples:
  bash pipelines/runpod_sync_hot_to_cold.sh \
    --output deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-50

  bash pipelines/runpod_sync_hot_to_cold.sh --all-outputs --all-runs
USAGE
}

cold_run_root() {
  printf '%s/cot-safety/runs' "${COT_SAFETY_COLD_ROOT}"
}

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=False))
PY
}

copy_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "missing hot source: ${src}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  local src_real dst_real
  src_real="$(resolve_path "${src}")"
  dst_real="$(resolve_path "${dst}")"
  if [[ "${src_real}" == "${dst_real}" ]]; then
    echo "already cold: ${src} -> ${dst}"
    return 0
  fi

  echo "persisting: ${src} -> ${dst}"
  if command -v rsync >/dev/null 2>&1; then
    local extra=()
    local progress=()
    if [[ "${DELETE:-0}" == "1" ]]; then
      extra+=(--delete)
    fi
    if rsync --help 2>/dev/null | grep -q -- '--info='; then
      progress+=(--info=progress2)
    fi
    if [[ -d "${src}" ]]; then
      mkdir -p "${dst}"
      rsync -a "${extra[@]}" "${progress[@]}" "${src}/" "${dst}/"
    else
      rsync -a "${extra[@]}" "${progress[@]}" "${src}" "${dst}"
    fi
  else
    if [[ "${DELETE:-0}" == "1" ]]; then
      echo "warning: --delete ignored because rsync is unavailable" >&2
    fi
    if [[ -d "${src}" ]]; then
      mkdir -p "${dst}"
      cp -a "${src}/." "${dst}/"
    else
      cp -a "${src}" "${dst}"
    fi
  fi

  if [[ -d "${dst}" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${dst}/.synced_to_cold"
  fi
}

remove_hot_after_sync() {
  local src="$1"
  local dst="$2"
  local src_real dst_real

  src_real="$(resolve_path "${src}")"
  dst_real="$(resolve_path "${dst}")"
  if [[ "${src_real}" == "${dst_real}" ]]; then
    echo "not removing hot path because it is already cold: ${src}"
    return 0
  fi

  if [[ -d "${dst}" && ! -f "${dst}/.synced_to_cold" ]]; then
    echo "refusing to remove hot path without cold marker: ${dst}" >&2
    return 1
  fi
  if [[ ! -e "${dst}" ]]; then
    echo "refusing to remove hot path; cold copy is missing: ${dst}" >&2
    return 1
  fi

  echo "removing synced hot path: ${src}"
  rm -rf -- "${src}"
}

sync_output_path() {
  local path="$1"
  local src="${COT_SAFETY_OUTPUT_ROOT}/${path}"
  local dst="${COT_SAFETY_COLD_ROOT}/outputs/${path}"

  copy_path "${src}" "${dst}"
  if [[ "${REMOVE_HOT_AFTER_SYNC}" == "1" ]]; then
    remove_hot_after_sync "${src}" "${dst}"
  fi
}

print_check() {
  echo "Cold root: ${COT_SAFETY_COLD_ROOT}"
  echo "Hot root:  ${COT_SAFETY_HOT_ROOT:-disabled}"
  echo "Hot outputs:  ${COT_SAFETY_OUTPUT_ROOT}"
  echo "Cold outputs: ${COT_SAFETY_COLD_ROOT}/outputs"
  echo "Hot data:     ${COT_SAFETY_DATA_ROOT}"
  echo "Cold data:    ${COT_SAFETY_COLD_ROOT}/data"
  echo "Hot runs:     ${COT_SAFETY_RUN_ROOT}"
  echo "Cold runs:    $(cold_run_root)"
  df -h "${COT_SAFETY_HOT_ROOT:-${COT_SAFETY_COLD_ROOT}}" "${COT_SAFETY_COLD_ROOT}" 2>/dev/null || true
}

OUTPUTS=()
DATA_PATHS=()
RUN_PATHS=()
ALL_OUTPUTS=0
ALL_RUNS=0
DELETE=0
CHECK_ONLY=0
REMOVE_HOT_AFTER_SYNC=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output)
      shift
      OUTPUTS+=("$1")
      ;;
    --data)
      shift
      DATA_PATHS+=("$1")
      ;;
    --run)
      shift
      RUN_PATHS+=("$1")
      ;;
    --all-outputs)
      ALL_OUTPUTS=1
      ;;
    --all-runs)
      ALL_RUNS=1
      ;;
    --delete)
      DELETE=1
      ;;
    --remove-hot-after-sync)
      REMOVE_HOT_AFTER_SYNC=1
      ;;
    --check-only)
      CHECK_ONLY=1
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

if [[ "${CHECK_ONLY}" == "1" ]]; then
  print_check
  exit 0
fi

if [[ "${ALL_OUTPUTS}" == "1" ]]; then
  copy_path "${COT_SAFETY_OUTPUT_ROOT}" "${COT_SAFETY_COLD_ROOT}/outputs"
fi
if [[ "${ALL_RUNS}" == "1" ]]; then
  copy_path "${COT_SAFETY_RUN_ROOT}" "$(cold_run_root)"
fi
for path in "${OUTPUTS[@]}"; do
  sync_output_path "${path}"
done
for path in "${DATA_PATHS[@]}"; do
  copy_path "${COT_SAFETY_DATA_ROOT}/${path}" "${COT_SAFETY_COLD_ROOT}/data/${path}"
done
for path in "${RUN_PATHS[@]}"; do
  copy_path "${COT_SAFETY_RUN_ROOT}/${path}" "$(cold_run_root)/${path}"
done

echo "Hot-to-cold sync complete."
print_check
