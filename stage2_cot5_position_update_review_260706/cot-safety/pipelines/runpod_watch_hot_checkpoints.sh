#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_base_env.sh"

OUTPUT_PATH=""
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
STOP_PID_FILE=""
ONCE=0
REMOVE_HOT_AFTER_SYNC=0
KEEP_LATEST_HOT=0
KEEP_BEST_HOT=0
SYNC_OUTPUT_AFTER_STOP=0
REMOVE_HOT_OUTPUT_AFTER_STOP=0

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/runpod_watch_hot_checkpoints.sh --output PATH [options]

Options:
  --output PATH       Output directory relative to $COT_SAFETY_OUTPUT_ROOT.
                      Example: deepseek_8b_run_name
  --interval SECONDS  Poll interval. Default: 60.
  --stop-pid-file P   Exit after this PID file's process ends, after one final sync.
  --remove-hot-after-sync
                      Remove each hot checkpoint after its cold copy is marked
                      synced. Use this when /dev/shm is the hot root.
  --keep-latest-hot N
                      When removing hot checkpoints, always keep the latest N
                      complete hot checkpoints. Default: 0.
  --keep-best-hot    When removing hot checkpoints, keep the checkpoint named by
                      the newest trainer_state.json best_model_checkpoint. Use
                      this with load_best_model_at_end / early stopping.
  --sync-output-after-stop
                      After the watched training process exits, sync the full
                      hot output directory to cold storage. This captures final/.
  --remove-hot-output-after-stop
                      After the final full-output sync, remove the whole hot
                      output directory. Intended for Stage 2 runs that continue
                      from /workspace outputs.
  --once              Sync currently complete checkpoints once, then exit.

The script persists complete hot checkpoints to /workspace/outputs/PATH.
A checkpoint is considered complete when trainer_state.json and
one model weight artifact both exist.
USAGE
}

checkpoint_complete() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/trainer_state.json" ]] || return 1
  [[ -f "${checkpoint_dir}/model.safetensors.index.json" ]] && return 0
  [[ -f "${checkpoint_dir}/model.safetensors" ]] && return 0
  [[ -f "${checkpoint_dir}/pytorch_model.bin.index.json" ]] && return 0
  [[ -f "${checkpoint_dir}/pytorch_model.bin" ]] && return 0
  [[ -f "${checkpoint_dir}/adapter_model.safetensors" ]] && return 0
  return 1
}

protected_hot_checkpoints() {
  local hot_dir="${COT_SAFETY_OUTPUT_ROOT}/${OUTPUT_PATH}"
  [[ -d "${hot_dir}" ]] || return 0

  if [[ "${KEEP_LATEST_HOT}" -gt 0 ]]; then
    python - "$hot_dir" "$KEEP_LATEST_HOT" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

hot_dir = Path(sys.argv[1])
keep = int(sys.argv[2])
items = []
for path in hot_dir.glob("checkpoint-*"):
    if not path.is_dir():
        continue
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match:
        items.append((int(match.group(1)), path.name))
for _, name in sorted(items)[-keep:]:
    print(name)
PY
  fi

  if [[ "${KEEP_BEST_HOT}" == "1" ]]; then
    python - "$hot_dir" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

hot_dir = Path(sys.argv[1])
states = []
for path in hot_dir.glob("checkpoint-*/trainer_state.json"):
    match = re.search(r"checkpoint-(\d+)$", str(path.parent))
    try:
        state = json.loads(path.read_text())
    except Exception:
        continue
    step = int(state.get("global_step") or (match.group(1) if match else 0))
    states.append((step, state))
if not states:
    raise SystemExit(0)
_, newest = max(states, key=lambda item: item[0])
best = newest.get("best_model_checkpoint")
if best:
    print(os.path.basename(str(best).rstrip("/")))
PY
  fi
}

hot_checkpoint_is_protected() {
  local name="$1"
  protected_hot_checkpoints | grep -Fxq "${name}"
}

remove_hot_checkpoint_if_allowed() {
  local name="$1"
  local hot_dir="${COT_SAFETY_OUTPUT_ROOT}/${OUTPUT_PATH}/${name}"
  local cold_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/${name}"

  [[ "${REMOVE_HOT_AFTER_SYNC}" == "1" ]] || return 0
  [[ -d "${hot_dir}" ]] || return 0

  if hot_checkpoint_is_protected "${name}"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keeping protected hot checkpoint ${OUTPUT_PATH}/${name}"
    return 0
  fi
  if [[ ! -f "${cold_dir}/.synced_to_cold" ]]; then
    echo "refusing to remove hot checkpoint without cold marker: ${cold_dir}" >&2
    return 1
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] removing synced hot checkpoint ${OUTPUT_PATH}/${name}"
  rm -rf -- "${hot_dir}"
}

sync_checkpoint() {
  local name="$1"
  local hot_dir="${COT_SAFETY_OUTPUT_ROOT}/${OUTPUT_PATH}/${name}"
  local cold_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/${name}"
  local marker="${cold_dir}/.synced_to_cold"

  if [[ -f "${marker}" ]]; then
    remove_hot_checkpoint_if_allowed "${name}"
    return 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] syncing ${OUTPUT_PATH}/${name}"
  local sync_args=(
    --output "${OUTPUT_PATH}/${name}"
    --all-runs
  )
  ROOT="${ROOT}" bash "${ROOT}/pipelines/runpod_sync_hot_to_cold.sh" \
    "${sync_args[@]}"
  mkdir -p "${cold_dir}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${marker}"
  remove_hot_checkpoint_if_allowed "${name}"
}

sync_complete_checkpoints() {
  local hot_dir="${COT_SAFETY_OUTPUT_ROOT}/${OUTPUT_PATH}"
  [[ -d "${hot_dir}" ]] || return 0

  local checkpoint_dir name
  for checkpoint_dir in "${hot_dir}"/checkpoint-*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    checkpoint_complete "${checkpoint_dir}" || continue
    name="$(basename "${checkpoint_dir}")"
    sync_checkpoint "${name}"
  done
}

training_process_alive() {
  [[ -n "${STOP_PID_FILE}" ]] || return 0
  [[ -f "${STOP_PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${STOP_PID_FILE}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output)
      shift
      OUTPUT_PATH="$1"
      ;;
    --interval)
      shift
      INTERVAL_SECONDS="$1"
      ;;
    --stop-pid-file)
      shift
      STOP_PID_FILE="$1"
      ;;
    --remove-hot-after-sync)
      REMOVE_HOT_AFTER_SYNC=1
      ;;
    --keep-latest-hot)
      shift
      KEEP_LATEST_HOT="$1"
      ;;
    --keep-best-hot)
      KEEP_BEST_HOT=1
      ;;
    --sync-output-after-stop)
      SYNC_OUTPUT_AFTER_STOP=1
      ;;
    --remove-hot-output-after-stop)
      REMOVE_HOT_OUTPUT_AFTER_STOP=1
      SYNC_OUTPUT_AFTER_STOP=1
      ;;
    --once)
      ONCE=1
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

if [[ -z "${OUTPUT_PATH}" ]]; then
  echo "missing --output" >&2
  usage >&2
  exit 2
fi

while true; do
  sync_complete_checkpoints
  if [[ "${ONCE}" == "1" ]]; then
    exit 0
  fi
  if [[ -n "${STOP_PID_FILE}" ]] && ! training_process_alive; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training process ended; final sync"
    KEEP_LATEST_HOT=0
    KEEP_BEST_HOT=0
    sync_complete_checkpoints
    if [[ "${SYNC_OUTPUT_AFTER_STOP}" == "1" ]]; then
      sync_args=(--output "${OUTPUT_PATH}")
      if [[ "${REMOVE_HOT_OUTPUT_AFTER_STOP}" == "1" ]]; then
        sync_args+=(--remove-hot-after-sync)
      fi
      ROOT="${ROOT}" bash "${ROOT}/pipelines/runpod_sync_hot_to_cold.sh" "${sync_args[@]}"
    fi
    ROOT="${ROOT}" bash "${ROOT}/pipelines/runpod_sync_hot_to_cold.sh" --all-runs
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
