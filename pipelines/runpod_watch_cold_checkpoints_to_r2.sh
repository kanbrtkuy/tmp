#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_base_env.sh"

OUTPUT_PATH=""
R2_ROOT="${R2_ROOT:-}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
STOP_PID_FILE=""
STATE_DIR="${STATE_DIR:-${COT_SAFETY_COLD_ROOT}/cot-safety/runs/r2_checkpoint_sync}"
LEDGER_FILE=""
ONCE=0
REMOVE_COLD_AFTER_UPLOAD=0
KEEP_LATEST_COLD=0
KEEP_BEST_COLD=0
SYNC_FINAL_AFTER_STOP=0

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/runpod_watch_cold_checkpoints_to_r2.sh --output PATH --r2-root REMOTE [options]

Options:
  --output PATH       Output directory relative to /workspace/outputs.
  --r2-root REMOTE    Rclone remote prefix, for example:
                      cloudflare_r2_cot_safety:cot-safety/stage2-stage3/RUN
  --interval SECONDS  Poll interval. Default: 60.
  --stop-pid-file P   Exit after this PID file's process ends, after one final sync.
  --state-dir DIR     Local marker directory. Default:
                      /workspace/cot-safety/runs/r2_checkpoint_sync.
  --ledger FILE       JSONL upload ledger. Default: STATE_DIR/OUTPUT.r2.jsonl.
  --remove-cold-after-upload
                      Delete uploaded /workspace checkpoint directories after
                      rclone check succeeds. Protected latest/best checkpoints
                      are kept.
  --keep-latest-cold N
                      When deleting cold checkpoints, always keep the latest N
                      complete checkpoints. Default: 0.
  --keep-best-cold    Keep the checkpoint named by newest trainer_state.json
                      best_model_checkpoint.
  --sync-final-after-stop
                      After the watched process exits, also sync final/.
  --once              Sync currently complete checkpoints once, then exit.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

json_escape() {
  python - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
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

state_key() {
  printf '%s__%s' "${OUTPUT_PATH//\//__}" "$1"
}

marker_path() {
  printf '%s/%s.r2_uploaded.ok' "${STATE_DIR}" "$(state_key "$1")"
}

append_ledger() {
  local name="$1"
  local src="$2"
  local dst="$3"
  local event="$4"
  mkdir -p "$(dirname "${LEDGER_FILE}")"
  printf '{"time":"%s","event":%s,"output":%s,"name":%s,"src":%s,"dst":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(json_escape "${event}")" \
    "$(json_escape "${OUTPUT_PATH}")" \
    "$(json_escape "${name}")" \
    "$(json_escape "${src}")" \
    "$(json_escape "${dst}")" >> "${LEDGER_FILE}"
}

protected_cold_checkpoints() {
  local cold_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}"
  [[ -d "${cold_dir}" ]] || return 0

  if [[ "${KEEP_LATEST_COLD}" -gt 0 ]]; then
    python - "$cold_dir" "$KEEP_LATEST_COLD" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

cold_dir = Path(sys.argv[1])
keep = int(sys.argv[2])
items = []
for path in cold_dir.glob("checkpoint-*"):
    if not path.is_dir():
        continue
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match:
        items.append((int(match.group(1)), path.name))
for _, name in sorted(items)[-keep:]:
    print(name)
PY
  fi

  if [[ "${KEEP_BEST_COLD}" == "1" ]]; then
    python - "$cold_dir" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

cold_dir = Path(sys.argv[1])
states = []
for path in cold_dir.glob("checkpoint-*/trainer_state.json"):
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

cold_checkpoint_is_protected() {
  local name="$1"
  protected_cold_checkpoints | grep -Fxq "${name}"
}

upload_path_to_r2() {
  local name="$1"
  local src="$2"
  local dst="$3"
  local marker
  marker="$(marker_path "${name}")"

  if [[ -f "${marker}" ]]; then
    return 0
  fi

  log "uploading ${src} -> ${dst}"
  rclone copy "${src}" "${dst}" --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list
  rclone check "${src}" "${dst}" --one-way --size-only --s3-no-check-bucket --checkers 16
  mkdir -p "${STATE_DIR}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${marker}"
  append_ledger "${name}" "${src}" "${dst}" "uploaded"
}

remove_cold_checkpoint_if_allowed() {
  local name="$1"
  local cold_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/${name}"
  local marker
  marker="$(marker_path "${name}")"

  [[ "${REMOVE_COLD_AFTER_UPLOAD}" == "1" ]] || return 0
  [[ -d "${cold_dir}" ]] || return 0
  [[ -f "${marker}" ]] || return 0

  if cold_checkpoint_is_protected "${name}"; then
    log "keeping protected cold checkpoint ${OUTPUT_PATH}/${name}"
    return 0
  fi

  log "removing uploaded cold checkpoint ${OUTPUT_PATH}/${name}"
  rm -rf -- "${cold_dir}"
  append_ledger "${name}" "${cold_dir}" "${R2_ROOT}/workspace/outputs/${OUTPUT_PATH}/${name}" "removed_cold"
}

sync_checkpoint() {
  local name="$1"
  local cold_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/${name}"
  local dst="${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}/${name}"
  upload_path_to_r2 "${name}" "${cold_dir}" "${dst}"
  remove_cold_checkpoint_if_allowed "${name}"
}

sync_complete_checkpoints() {
  local cold_root="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}"
  local checkpoint_dir name
  [[ -d "${cold_root}" ]] || return 0

  for checkpoint_dir in "${cold_root}"/checkpoint-*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    checkpoint_complete "${checkpoint_dir}" || continue
    name="$(basename "${checkpoint_dir}")"
    sync_checkpoint "${name}"
  done
}

sync_final_dir() {
  local final_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/final"
  [[ -d "${final_dir}" ]] || return 0
  upload_path_to_r2 "final" "${final_dir}" "${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}/final"
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
    --r2-root)
      shift
      R2_ROOT="$1"
      ;;
    --interval)
      shift
      INTERVAL_SECONDS="$1"
      ;;
    --stop-pid-file)
      shift
      STOP_PID_FILE="$1"
      ;;
    --state-dir)
      shift
      STATE_DIR="$1"
      ;;
    --ledger)
      shift
      LEDGER_FILE="$1"
      ;;
    --remove-cold-after-upload)
      REMOVE_COLD_AFTER_UPLOAD=1
      ;;
    --keep-latest-cold)
      shift
      KEEP_LATEST_COLD="$1"
      ;;
    --keep-best-cold)
      KEEP_BEST_COLD=1
      ;;
    --sync-final-after-stop)
      SYNC_FINAL_AFTER_STOP=1
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

if [[ -z "${OUTPUT_PATH}" || -z "${R2_ROOT}" ]]; then
  echo "missing --output or --r2-root" >&2
  usage >&2
  exit 2
fi

if [[ -z "${LEDGER_FILE}" ]]; then
  LEDGER_FILE="${STATE_DIR}/${OUTPUT_PATH//\//__}.r2.jsonl"
fi

mkdir -p "${STATE_DIR}"

while true; do
  sync_complete_checkpoints
  if [[ "${ONCE}" == "1" ]]; then
    exit 0
  fi
  if [[ -n "${STOP_PID_FILE}" ]] && ! training_process_alive; then
    log "training process ended; final R2 sync"
    KEEP_LATEST_COLD=0
    KEEP_BEST_COLD=0
    sync_complete_checkpoints
    if [[ "${SYNC_FINAL_AFTER_STOP}" == "1" ]]; then
      sync_final_dir
    fi
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
