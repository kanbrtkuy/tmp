#!/usr/bin/env bash
set -euo pipefail

# Stage2.1-pure 8B formal run:
#   pytest -> data prep -> full SFT -> generation -> strict natural gate
#   -> judge -> summary.  The strict gate runs before judge so a bad natural
#   pause checkpoint does not spend extra judge GPU time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/experiment/stage21_pause_pure_dagger_8b_full_2xa100.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_8b_stage21_pure_cot5_2xa100.yaml}"
RUN_PYTEST="${RUN_PYTEST:-1}"
RUN_DATA_PREP="${RUN_DATA_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_GATE="${RUN_GATE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_SYNC_COLD="${RUN_SYNC_COLD:-1}"
CONDITIONS="${CONDITIONS:-base_natural,stage21_pure_cot5_natural,stage21_pure_cot5_forced}"
NATURAL_CONDITION="${NATURAL_CONDITION:-stage21_pure_cot5_natural}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_stage2_env.sh"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export STAGE21_PURE_8B_CHECKPOINT="${STAGE21_PURE_8B_CHECKPOINT:-${COT_SAFETY_COLD_ROOT}/outputs/deepseek_8b_stage21_pause_pure_cot5_full_2xa100/final}"

EVAL_ROOT="${EVAL_ROOT:-${COT_SAFETY_RUN_ROOT}/eval/stage2_model_comparison_deepseek_8b_stage21_pure_cot5_2xa100}"
GATE_JSON="${GATE_JSON:-${EVAL_ROOT}/stage21_pure_natural_gate.json}"
SYNC_COLD_DONE=0

sync_cold_runs() {
  if [[ "${RUN_SYNC_COLD}" != "1" || "${SYNC_COLD_DONE}" == "1" ]]; then
    return 0
  fi
  SYNC_COLD_DONE=1
  bash pipelines/runpod_sync_hot_to_cold.sh --all-runs || {
    echo "[stage21-pure-8b] warning: hot-to-cold run sync failed" >&2
    return 0
  }
}

checkpoint_ready() {
  [[ -f "${STAGE21_PURE_8B_CHECKPOINT}/config.json" ]] || return 1
  find "${STAGE21_PURE_8B_CHECKPOINT}" -maxdepth 1 \
    \( -name '*.safetensors' -o -name 'model-*.safetensors' -o -name 'pytorch_model*.bin' \) \
    -print -quit | grep -q .
}

if [[ "${RUN_SYNC_COLD}" == "1" ]]; then
  trap sync_cold_runs EXIT
fi

mkdir -p "${COT_SAFETY_RUN_ROOT}" logs

echo "[stage21-pure-8b] TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[stage21-pure-8b] EVAL_CONFIG=${EVAL_CONFIG}"
echo "[stage21-pure-8b] STAGE21_PURE_8B_CHECKPOINT=${STAGE21_PURE_8B_CHECKPOINT}"
echo "[stage21-pure-8b] EVAL_ROOT=${EVAL_ROOT}"

if [[ "${RUN_PYTEST}" == "1" ]]; then
  "${PYTHON_BIN}" -m pytest \
    tests/test_stage2_pause_kl_trainer.py \
    tests/test_stage2_natural_pause_metrics.py \
    tests/test_stage2_onpolicy_mining.py \
    tests/test_stage2_dagger_mix.py \
    tests/test_stage2_checkpoint_diag.py \
    -q
fi

if [[ "${RUN_DATA_PREP}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_stage2_sft.py \
    --config "${TRAIN_CONFIG}" \
    --skip_train
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_stage2_sft.py \
    --config "${TRAIN_CONFIG}" \
    --skip_data_prep
fi

if [[ "${RUN_GENERATE}" == "1" ]] && ! checkpoint_ready; then
  echo "[stage21-pure-8b] missing or incomplete checkpoint: ${STAGE21_PURE_8B_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${RUN_GENERATE}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${EVAL_CONFIG}" \
    --phase prepare
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${EVAL_CONFIG}" \
    --phase generate \
    --conditions "${CONDITIONS}"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/diag_stage2_checkpoint.py \
    --config "${TRAIN_CONFIG}" \
    --input_jsonl "${EVAL_ROOT}/generations/${NATURAL_CONDITION}_capability.jsonl" \
    --input_jsonl "${EVAL_ROOT}/generations/${NATURAL_CONDITION}_safety.jsonl" \
    --output_json "${GATE_JSON}" \
    --generation_field generated \
    --use_existing_metrics \
    --strict
  echo "[stage21-pure-8b] gate written to ${GATE_JSON}"
fi

if [[ "${RUN_JUDGE}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${EVAL_CONFIG}" \
    --phase judge \
    --conditions "${CONDITIONS}"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_model_comparison_eval.py \
    --config "${EVAL_CONFIG}" \
    --phase summary
fi

sync_cold_runs
