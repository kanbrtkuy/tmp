#!/usr/bin/env bash
set -euo pipefail

# Stage2.1-pure 1.5B full run:
#   pytest -> data prep -> full rows-only SFT -> generation -> strict natural
#   exact-3/location gate -> optional judge/summary.
#
# This wrapper defaults to cold /workspace storage because the 2x A6000 RunPod
# nodes often expose only a small /dev/shm, while full checkpoint sweeps need
# durable space.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_2xa6000.yaml}"
RUN_PYTEST="${RUN_PYTEST:-1}"
RUN_DATA_PREP="${RUN_DATA_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_GATE="${RUN_GATE:-1}"
RUN_JUDGE="${RUN_JUDGE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_SYNC_COLD="${RUN_SYNC_COLD:-0}"
CONDITIONS="${CONDITIONS:-base_natural,stage21_pure_cot5_natural,stage21_pure_cot5_forced}"
NATURAL_CONDITION="${NATURAL_CONDITION:-stage21_pure_cot5_natural}"

export COT_SAFETY_USE_HOT_STORAGE="${COT_SAFETY_USE_HOT_STORAGE:-0}"
export COT_SAFETY_STAGE2_CHECK_BNB="${COT_SAFETY_STAGE2_CHECK_BNB:-0}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_stage2_env.sh"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export STAGE21_PURE_1P5B_CHECKPOINT="${STAGE21_PURE_1P5B_CHECKPOINT:-${COT_SAFETY_COLD_ROOT}/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000/final}"

EVAL_ROOT="${EVAL_ROOT:-${COT_SAFETY_RUN_ROOT}/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_2xa6000}"
GATE_JSON="${GATE_JSON:-${EVAL_ROOT}/stage21_pure_natural_gate.json}"
SYNC_COLD_DONE=0

STAGE2_SFT_OVERRIDES=()
if [[ -n "${PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE:-}" ]]; then
  STAGE2_SFT_OVERRIDES+=(--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${PER_DEVICE_EVAL_BATCH_SIZE_OVERRIDE:-}" ]]; then
  STAGE2_SFT_OVERRIDES+=(--per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS_OVERRIDE:-}" ]]; then
  STAGE2_SFT_OVERRIDES+=(--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS_OVERRIDE}")
fi
if [[ -n "${DATALOADER_NUM_WORKERS_OVERRIDE:-}" ]]; then
  STAGE2_SFT_OVERRIDES+=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS_OVERRIDE}")
fi
if [[ -n "${OPTIM_OVERRIDE:-}" ]]; then
  STAGE2_SFT_OVERRIDES+=(--optim "${OPTIM_OVERRIDE}")
fi

sync_cold_runs() {
  if [[ "${RUN_SYNC_COLD}" != "1" || "${SYNC_COLD_DONE}" == "1" ]]; then
    return 0
  fi
  SYNC_COLD_DONE=1
  bash pipelines/runpod_sync_hot_to_cold.sh --all-runs || {
    echo "[stage21-pure-1p5b] warning: hot-to-cold run sync failed" >&2
    return 0
  }
}

checkpoint_ready() {
  [[ -f "${STAGE21_PURE_1P5B_CHECKPOINT}/config.json" ]] || return 1
  find "${STAGE21_PURE_1P5B_CHECKPOINT}" -maxdepth 1 \
    \( -name '*.safetensors' -o -name 'model-*.safetensors' -o -name 'pytorch_model*.bin' \) \
    -print -quit | grep -q .
}

if [[ "${RUN_SYNC_COLD}" == "1" ]]; then
  trap sync_cold_runs EXIT
fi

mkdir -p "${COT_SAFETY_RUN_ROOT}" logs

echo "[stage21-pure-1p5b] TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[stage21-pure-1p5b] EVAL_CONFIG=${EVAL_CONFIG}"
echo "[stage21-pure-1p5b] STAGE21_PURE_1P5B_CHECKPOINT=${STAGE21_PURE_1P5B_CHECKPOINT}"
echo "[stage21-pure-1p5b] EVAL_ROOT=${EVAL_ROOT}"
echo "[stage21-pure-1p5b] COT_SAFETY_USE_HOT_STORAGE=${COT_SAFETY_USE_HOT_STORAGE}"
echo "[stage21-pure-1p5b] SFT_OVERRIDES=${STAGE2_SFT_OVERRIDES[*]:-none}"

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
    "${STAGE2_SFT_OVERRIDES[@]}" \
    --skip_train
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_stage2_sft.py \
    --config "${TRAIN_CONFIG}" \
    "${STAGE2_SFT_OVERRIDES[@]}" \
    --skip_data_prep
fi

if [[ "${RUN_GENERATE}" == "1" ]] && ! checkpoint_ready; then
  echo "[stage21-pure-1p5b] missing or incomplete checkpoint: ${STAGE21_PURE_1P5B_CHECKPOINT}" >&2
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
  echo "[stage21-pure-1p5b] gate written to ${GATE_JSON}"
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
