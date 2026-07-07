#!/usr/bin/env bash
set -euo pipefail

# Wait for the 1.5B Stage2.1-pure full SFT and R2 checkpoint watcher to finish,
# then select the best checkpoint on the disjoint selection-dev gate and run the
# full natural pause-emission gate on the selected checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if [[ "${PYTHON_BIN}" = /* ]]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
fi
export COT_SAFETY_EVAL_PYTHON="${COT_SAFETY_EVAL_PYTHON:-${PYTHON_BIN}}"

TRAIN_PID_FILE="${TRAIN_PID_FILE:-${ROOT}/runs/stage21_pure_1p5b_full_bs4_ga2_260707.pid}"
WATCH_PID_FILE="${WATCH_PID_FILE:-${ROOT}/runs/r2_watch_stage21_pure_1p5b_full_bs4_ga2_260707.pid}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml}"
SELECTION_EVAL_CONFIG="${SELECTION_EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml}"
FULL_EVAL_CONFIG="${FULL_EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selected_2xa6000.yaml}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000}"
SELECTION_ROOT="${SELECTION_ROOT:-${ROOT}/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000}"
FULL_EVAL_ROOT="${FULL_EVAL_ROOT:-${ROOT}/runs/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_selected_2xa6000}"
R2_ROOT="${R2_ROOT:-cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2}"

STRIDE_STEPS="${STRIDE_STEPS:-50}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
MAX_STEP="${MAX_STEP:-}"
SELECTION_CONDITION="${SELECTION_CONDITION:-stage21_pure_cot5_natural}"
FULL_CONDITIONS="${FULL_CONDITIONS:-stage21_pure_cot5_natural,stage21_pure_cot5_forced}"
NATURAL_CONDITION="${NATURAL_CONDITION:-stage21_pure_cot5_natural}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

log() {
  printf '[stage21-postselect] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

pid_from_file() {
  local pid_file="$1"
  [[ -s "${pid_file}" ]] || return 1
  tr -cd '0-9' < "${pid_file}"
}

wait_for_pid_file() {
  local label="$1"
  local pid_file="$2"
  local pid=""
  log "waiting for ${label}: ${pid_file}"
  while true; do
    if ! pid="$(pid_from_file "${pid_file}")" || [[ -z "${pid}" ]]; then
      log "${label} pid file absent/empty; treating as finished"
      return 0
    fi
    if ! ps -p "${pid}" >/dev/null 2>&1; then
      log "${label} pid ${pid} finished"
      return 0
    fi
    sleep 60
  done
}

run_selection() {
  local cmd=(
    "${PYTHON_BIN}" scripts/select_stage21_checkpoint.py
    --eval_config "${SELECTION_EVAL_CONFIG}"
    --train_config "${TRAIN_CONFIG}"
    --checkpoint_root "${CHECKPOINT_ROOT}"
    --output_root "${SELECTION_ROOT}"
    --r2_root "${R2_ROOT}"
    --stride_steps "${STRIDE_STEPS}"
    --natural_condition "${SELECTION_CONDITION}"
    --remove_downloaded
  )
  if [[ -n "${MAX_CANDIDATES}" ]]; then
    cmd+=(--max_candidates "${MAX_CANDIDATES}")
  fi
  if [[ -n "${MAX_STEP}" ]]; then
    cmd+=(--max_step "${MAX_STEP}")
  fi
  log "running checkpoint selection sweep"
  "${cmd[@]}"
}

run_full_gate() {
  local selected_file="${SELECTION_ROOT}/selected_checkpoint.txt"
  if [[ ! -s "${selected_file}" ]]; then
    log "missing selected checkpoint file: ${selected_file}"
    return 1
  fi
  export STAGE21_PURE_1P5B_CHECKPOINT
  STAGE21_PURE_1P5B_CHECKPOINT="$(head -n 1 "${selected_file}")"
  export STAGE21_FULL_EVAL_ROOT="${FULL_EVAL_ROOT}"

  log "selected checkpoint: ${STAGE21_PURE_1P5B_CHECKPOINT}"
  log "running full eval prepare/generate/gate at ${FULL_EVAL_ROOT}"
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${FULL_EVAL_CONFIG}" \
    --phase prepare
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${FULL_EVAL_CONFIG}" \
    --phase generate \
    --conditions "${FULL_CONDITIONS}"
  "${PYTHON_BIN}" scripts/diag_stage2_checkpoint.py \
    --config "${TRAIN_CONFIG}" \
    --input_jsonl "${FULL_EVAL_ROOT}/generations/${NATURAL_CONDITION}_capability.jsonl" \
    --input_jsonl "${FULL_EVAL_ROOT}/generations/${NATURAL_CONDITION}_safety.jsonl" \
    --output_json "${FULL_EVAL_ROOT}/stage21_pure_natural_gate.json" \
    --generation_field generated \
    --use_existing_metrics \
    --strict
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${FULL_EVAL_CONFIG}" \
    --phase summary
}

cd "${ROOT}"
log "ROOT=${ROOT}"
log "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
log "R2_ROOT=${R2_ROOT}"
wait_for_pid_file "training wrapper" "${TRAIN_PID_FILE}"
wait_for_pid_file "R2 checkpoint watcher" "${WATCH_PID_FILE}"
sleep 30
run_selection
run_full_gate
log "done"
