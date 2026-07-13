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
SYNC_OUTPUT_METADATA_AFTER_STOP=0
REMOVE_COLD_OUTPUT_AFTER_UPLOAD=0
CHECKPOINT_INTEGRITY_STRICT="${CHECKPOINT_INTEGRITY_STRICT:-0}"
CHECKPOINT_INTEGRITY_CLI="${CHECKPOINT_INTEGRITY_CLI:-${ROOT}/scripts/checkpoint_integrity.py}"

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
  --sync-output-metadata-after-stop
                      After the watched process exits, upload root-level run
                      provenance/log/config files while excluding checkpoint-*
                      and final/ payloads.
  --remove-cold-output-after-upload
                      After verified checkpoint, final, and metadata uploads,
                      remove the complete /workspace output directory.
  --once              Sync currently complete checkpoints once, then exit.

Environment:
  CHECKPOINT_INTEGRITY_STRICT=1
                      Canonical fail-closed mode. Consume only locally verified
                      cold receipts, upload via a manifest-bound partial key,
                      use rclone --download SHA verification, and commit the R2
                      receipt object last. Size-only checks are forbidden.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

json_escape() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

checkpoint_complete() {
  local checkpoint_dir="$1"
  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" ]]; then
    local cold_receipt="${checkpoint_dir}/.cold_complete.json"
    [[ -f "${checkpoint_dir}/.checkpoint_complete.json" ]] || return 1
    [[ -f "${cold_receipt}" ]] || return 1
    if ! python3 "${CHECKPOINT_INTEGRITY_CLI}" verify-receipt \
      --checkpoint "${checkpoint_dir}" \
      --receipt "${cold_receipt}" \
      --kind cold \
      --destination "${checkpoint_dir}" >/dev/null; then
      echo "cold checkpoint receipt failed integrity verification: ${checkpoint_dir}" >&2
      return 2
    fi
    return 0
  fi
  [[ -f "${checkpoint_dir}/trainer_state.json" ]] || return 1
  [[ -f "${checkpoint_dir}/model.safetensors.index.json" ]] && return 0
  [[ -f "${checkpoint_dir}/model.safetensors" ]] && return 0
  [[ -f "${checkpoint_dir}/pytorch_model.bin.index.json" ]] && return 0
  [[ -f "${checkpoint_dir}/pytorch_model.bin" ]] && return 0
  [[ -f "${checkpoint_dir}/adapter_model.safetensors" ]] && return 0
  return 1
}

state_key() {
  local destination_hash
  destination_hash="$(python3 - "${R2_ROOT}" <<'PY'
import hashlib
import sys
print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:16])
PY
)"
  printf '%s__%s__r2_%s' "${OUTPUT_PATH//\//__}" "$1" "${destination_hash}"
}

marker_path() {
  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" && "$1" == checkpoint-* ]]; then
    printf '%s/%s.r2_uploaded.ok.json' "${STATE_DIR}" "$(state_key "$1")"
  else
    printf '%s/%s.r2_uploaded.ok' "${STATE_DIR}" "$(state_key "$1")"
  fi
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
    python3 - "$cold_dir" "$KEEP_LATEST_COLD" <<'PY'
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
    python3 - "$cold_dir" <<'PY'
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

  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" && "${name}" == checkpoint-* ]]; then
    local manifest_sha partial_dst pending_dir pending_receipt remote_receipt receipt_check_dir
    remote_receipt="${dst%/}/.r2_complete.json"

    python3 "${CHECKPOINT_INTEGRITY_CLI}" verify-receipt \
      --checkpoint "${src}" \
      --receipt "${src}/.cold_complete.json" \
      --kind cold \
      --destination "${src}" >/dev/null

    if [[ -f "${marker}" ]]; then
      python3 "${CHECKPOINT_INTEGRITY_CLI}" verify-receipt \
        --checkpoint "${src}" \
        --receipt "${marker}" \
        --kind r2 \
        --destination "${dst}" >/dev/null
      # A local receipt proves a historical upload, not that the destination
      # still exists. Recheck both payload and receipt before any retry can
      # delete the remaining cold checkpoint.
      rclone check "${src}" "${dst}" \
        --exclude '/.r2_complete.json' \
        --download --s3-no-check-bucket --checkers 16
      receipt_check_dir="${STATE_DIR}/.$(state_key "${name}").receipt-recheck.$$"
      if [[ -e "${receipt_check_dir}" ]]; then
        echo "refusing pre-existing receipt recheck directory: ${receipt_check_dir}" >&2
        return 1
      fi
      mkdir -p "${receipt_check_dir}"
      cp "${marker}" "${receipt_check_dir}/.r2_complete.json"
      if ! rclone check "${receipt_check_dir}" "${dst}" \
        --include '/.r2_complete.json' \
        --one-way --download --s3-no-check-bucket --checkers 1; then
        rm -rf -- "${receipt_check_dir}"
        return 1
      fi
      rm -rf -- "${receipt_check_dir}"
      return 0
    fi

    manifest_sha="$(python3 "${CHECKPOINT_INTEGRITY_CLI}" manifest-sha256 "${src}")"
    partial_dst="${dst}.partial-${manifest_sha:0:16}"
    log "uploading ${src} -> ${partial_dst} (strict partial)"
    rclone copy "${src}" "${partial_dst}" \
      --exclude '/.r2_complete.json' \
      --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list
    rclone check "${src}" "${partial_dst}" \
      --exclude '/.r2_complete.json' \
      --download --s3-no-check-bucket --checkers 16

    # Promote the verified partial payload, then independently rehash the final
    # destination. The receipt object is deliberately absent during both checks.
    rclone copy "${partial_dst}" "${dst}" \
      --exclude '/.r2_complete.json' \
      --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list
    rclone check "${src}" "${dst}" \
      --exclude '/.r2_complete.json' \
      --download --s3-no-check-bucket --checkers 16

    mkdir -p "${STATE_DIR}"
    pending_dir="${STATE_DIR}/.$(state_key "${name}").receipt-partial.$$"
    pending_receipt="${pending_dir}/.r2_complete.json"
    if [[ -e "${pending_dir}" ]]; then
      echo "refusing pre-existing local receipt partial: ${pending_dir}" >&2
      return 1
    fi
    mkdir -p "${pending_dir}"
    python3 "${CHECKPOINT_INTEGRITY_CLI}" write-receipt \
      --checkpoint "${src}" \
      --kind r2 \
      --destination "${dst}" \
      --verification-tool rclone_check_download_sha256 \
      --output "${pending_receipt}" >/dev/null
    rclone copyto "${pending_receipt}" "${remote_receipt}" \
      --s3-no-check-bucket
    rclone check "${pending_dir}" "${dst}" \
      --include '/.r2_complete.json' \
      --one-way --download --s3-no-check-bucket --checkers 1
    mv "${pending_receipt}" "${marker}"
    rmdir "${pending_dir}"
    python3 "${CHECKPOINT_INTEGRITY_CLI}" verify-receipt \
      --checkpoint "${src}" \
      --receipt "${marker}" \
      --kind r2 \
      --destination "${dst}" >/dev/null
    append_ledger "${name}" "${src}" "${dst}" "uploaded_strong_sha256"
    if ! rclone purge "${partial_dst}" --s3-no-check-bucket; then
      log "warning: verified R2 payload committed, but partial cleanup failed: ${partial_dst}"
    fi
    return 0
  fi

  if [[ -f "${marker}" ]]; then
    if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" ]]; then
      rclone check "${src}" "${dst}" --one-way --download \
        --s3-no-check-bucket --checkers 16
    fi
    return 0
  fi

  log "uploading ${src} -> ${dst}"
  rclone copy "${src}" "${dst}" --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list
  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" ]]; then
    rclone check "${src}" "${dst}" --one-way --download --s3-no-check-bucket --checkers 16
  else
    rclone check "${src}" "${dst}" --one-way --size-only --s3-no-check-bucket --checkers 16
  fi
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

  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" && "${name}" == checkpoint-* ]]; then
    python3 "${CHECKPOINT_INTEGRITY_CLI}" verify-receipt \
      --checkpoint "${cold_dir}" \
      --receipt "${marker}" \
      --kind r2 \
      --destination "${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}/${name}" >/dev/null
  fi

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
  local checkpoint_dir name checkpoint_status
  [[ -d "${cold_root}" ]] || return 0

  for checkpoint_dir in "${cold_root}"/checkpoint-*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    if checkpoint_complete "${checkpoint_dir}"; then
      checkpoint_status=0
    else
      checkpoint_status=$?
    fi
    [[ "${checkpoint_status}" -eq 0 ]] || {
      [[ "${checkpoint_status}" -eq 1 ]] && continue
      return "${checkpoint_status}"
    }
    name="$(basename "${checkpoint_dir}")"
    sync_checkpoint "${name}"
  done
}

sync_final_dir() {
  local final_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}/final"
  local final_marker
  final_marker="$(marker_path "final")"
  if [[ ! -d "${final_dir}" ]]; then
    echo "required terminal final export is missing: ${final_dir}" >&2
    return 1
  fi
  upload_path_to_r2 "final" "${final_dir}" "${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}/final"
  if [[ "${REMOVE_COLD_OUTPUT_AFTER_UPLOAD}" == "1" ]]; then
    [[ -f "${final_marker}" ]] || {
      echo "refusing to remove final export without verified R2 marker" >&2
      return 1
    }
    log "removing verified final export ${final_dir}"
    rm -rf -- "${final_dir}"
  fi
}

sync_output_metadata() {
  local source_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}"
  local destination="${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}"
  local marker
  marker="$(marker_path "run_metadata")"
  [[ -d "${source_dir}" ]] || return 0
  log "uploading run metadata ${source_dir} -> ${destination}"
  rclone copy "${source_dir}" "${destination}" \
    --exclude '/checkpoint-*/**' \
    --exclude '/final/**' \
    --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list
  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" == "1" ]]; then
    rclone check "${source_dir}" "${destination}" \
      --exclude '/checkpoint-*/**' \
      --exclude '/final/**' \
      --one-way --download --s3-no-check-bucket --checkers 16
  else
    rclone check "${source_dir}" "${destination}" \
      --exclude '/checkpoint-*/**' \
      --exclude '/final/**' \
      --one-way --size-only --s3-no-check-bucket --checkers 16
  fi
  mkdir -p "${STATE_DIR}"
  printf '{"destination":%s,"event":"metadata_uploaded_and_verified","time":"%s"}\n' \
    "$(json_escape "${destination}")" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${marker}"
  append_ledger "run_metadata" "${source_dir}" "${destination}" "metadata_uploaded_strong_sha256"
}

remove_cold_output_if_complete() {
  local source_dir="${COT_SAFETY_COLD_ROOT}/outputs/${OUTPUT_PATH}"
  [[ "${REMOVE_COLD_OUTPUT_AFTER_UPLOAD}" == "1" ]] || return 0
  [[ -d "${source_dir}" ]] || return 0
  if find "${source_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    echo "refusing to remove cold output while checkpoint directories remain: ${source_dir}" >&2
    return 1
  fi
  [[ -f "$(marker_path "run_metadata")" ]] || {
    echo "refusing to remove cold output without verified metadata marker" >&2
    return 1
  }
  [[ -f "$(marker_path "final")" ]] || {
    echo "refusing to remove cold output without verified final marker" >&2
    return 1
  }
  [[ ! -d "${source_dir}/final" ]] || {
    echo "refusing to remove cold output before verified final cleanup" >&2
    return 1
  }
  log "removing fully uploaded cold output ${source_dir}"
  rm -rf -- "${source_dir}"
  append_ledger "run_output" "${source_dir}" "${R2_ROOT%/}/workspace/outputs/${OUTPUT_PATH}" "removed_cold_output"
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
    --sync-output-metadata-after-stop)
      SYNC_OUTPUT_METADATA_AFTER_STOP=1
      ;;
    --remove-cold-output-after-upload)
      REMOVE_COLD_OUTPUT_AFTER_UPLOAD=1
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

if [[ "${REMOVE_COLD_OUTPUT_AFTER_UPLOAD}" == "1" ]]; then
  if [[ "${CHECKPOINT_INTEGRITY_STRICT}" != "1" ]]; then
    echo "--remove-cold-output-after-upload requires CHECKPOINT_INTEGRITY_STRICT=1" >&2
    exit 2
  fi
  if [[ "${REMOVE_COLD_AFTER_UPLOAD}" != "1" ]]; then
    echo "--remove-cold-output-after-upload requires --remove-cold-after-upload" >&2
    exit 2
  fi
  if [[ "${SYNC_FINAL_AFTER_STOP}" != "1" ]]; then
    echo "--remove-cold-output-after-upload requires --sync-final-after-stop" >&2
    exit 2
  fi
  if [[ "${SYNC_OUTPUT_METADATA_AFTER_STOP}" != "1" ]]; then
    echo "--remove-cold-output-after-upload requires --sync-output-metadata-after-stop" >&2
    exit 2
  fi
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
    if [[ "${SYNC_OUTPUT_METADATA_AFTER_STOP}" == "1" ]]; then
      sync_output_metadata
    fi
    remove_cold_output_if_complete
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
