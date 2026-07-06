#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="${RUN_DIR:-/workspace/cot-safety/runs/stage2_8b_full_mb4_ga2_260706}"
LOG="${LOG:-${RUN_DIR}/post_sft_eval.log}"
STATUS="${STATUS:-${RUN_DIR}/post_sft_eval.status}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${RUN_DIR}/train_wrapper.pid}"
OUTPUT_NAME="${OUTPUT_NAME:-deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100}"
FINAL_DIR="${FINAL_DIR:-/workspace/outputs/${OUTPUT_NAME}/final}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100.yaml}"
EVAL_DIR="${EVAL_DIR:-/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100}"
R2_ROOT="${R2_ROOT:-cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2}"
JUDGE_LOG="${JUDGE_LOG:-/workspace/logs/restore_judges_from_r2_260706.log}"
PY="${PY:-/workspace/venvs/stage2/bin/python}"

mkdir -p "${RUN_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG}"
}

fail() {
  log "FAILED: $*"
  printf 'FAILED\n' > "${STATUS}"
  exit 1
}

wait_for_training() {
  if [[ -f "${TRAIN_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${TRAIN_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      while kill -0 "${pid}" 2>/dev/null; do
        log "training still running pid=${pid}"
        sleep 300
      done
      log "training wrapper pid=${pid} exited"
      return 0
    fi
  fi

  while pgrep -af "scripts/run_stage2_sft.py .*stage2_intra_pause_kl_transparent_emit_8b_cot5_full_save25_2xa100.yaml|trl_train.py .*${OUTPUT_NAME}" >/dev/null 2>&1; do
    log "training process still present"
    sleep 300
  done
  log "no training process detected"
}

wait_for_final_model() {
  local attempt
  for attempt in $(seq 1 720); do
    if [[ -f "${FINAL_DIR}/config.json" ]] && find "${FINAL_DIR}" -maxdepth 1 \( -name '*.safetensors' -o -name 'model-*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
      log "final model is present at ${FINAL_DIR}"
      return 0
    fi
    log "waiting for final model at ${FINAL_DIR} attempt=${attempt}"
    sleep 60
  done
  fail "final model did not appear"
}

judge_models_present() {
  [[ -f /workspace/models/judges/wildguard_vllm_head_dim128/config.json ]] &&
  [[ -f /workspace/models/judges/Llama-Guard-3-8B/config.json ]] &&
  [[ -f /workspace/models/judges/HarmBench-Llama-2-13b-cls/config.json ]]
}

wait_for_judges() {
  local attempt
  for attempt in $(seq 1 720); do
    if grep -q "RESTORE_JUDGES_RC=0" "${JUDGE_LOG}" 2>/dev/null && judge_models_present; then
      log "judge restore complete and required judge configs are present"
      return 0
    fi
    if judge_models_present && ! pgrep -af "rclone copy .*workspace/models/judges|restore_judges_from_r2" >/dev/null 2>&1; then
      log "judge model files are present and restore process is no longer running"
      return 0
    fi
    log "waiting for judge models attempt=${attempt}"
    sleep 60
  done
  fail "judge models did not become ready"
}

run_eval() {
  cd /workspace/cot-safety || fail "cannot cd to repo"

  if [[ -f /workspace/secrets/hf.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source /workspace/secrets/hf.env
    set +a
  fi

  export HF_HOME=/workspace/hf_cache
  export COT_SAFETY_MODEL_ROOT=/workspace/models
  export COT_SAFETY_OUTPUT_ROOT=/workspace/outputs
  export COT_SAFETY_DATA_ROOT=/workspace/data
  export COT_SAFETY_RUN_ROOT=/workspace/cot-safety/runs
  export COT_SAFETY_JUDGE_ROOT=/workspace/models/judges
  export COT_SAFETY_WILDGUARD_MODEL=/workspace/models/judges/wildguard_vllm_head_dim128
  export STAGE2_8B_COT5_MB4_GA2_CHECKPOINT="${FINAL_DIR}"

  log "running Stage2 eval dry-run"
  "${PY}" scripts/run_model_comparison_eval.py --config "${EVAL_CONFIG}" --dry_run >> "${LOG}" 2>&1 || fail "Stage2 eval dry-run failed"

  log "running Stage2 eval/judge"
  "${PY}" scripts/run_model_comparison_eval.py --config "${EVAL_CONFIG}" --phase all >> "${LOG}" 2>&1 || fail "Stage2 eval/judge failed"

  if [[ -d "${EVAL_DIR}" ]]; then
    log "syncing Stage2 eval results to R2"
    rclone copy "${EVAL_DIR}" "${R2_ROOT}/workspace/cot-safety/runs/eval/$(basename "${EVAL_DIR}")" --s3-no-check-bucket --transfers 8 --checkers 16 --fast-list >> "${LOG}" 2>&1 || fail "R2 eval sync failed"
  else
    fail "eval output dir missing: ${EVAL_DIR}"
  fi

  log "Stage2 eval complete; Stage3 is gated on Fable review"
  printf 'DONE_PENDING_FABLE\n' > "${STATUS}"
}

if [[ -f "${STATUS}" ]] && grep -q '^DONE_PENDING_FABLE$' "${STATUS}"; then
  log "post-SFT eval already completed"
  exit 0
fi

printf 'RUNNING\n' > "${STATUS}"
log "post-SFT eval watcher started"
wait_for_training
wait_for_final_model
wait_for_judges
run_eval
