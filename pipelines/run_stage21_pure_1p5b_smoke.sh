#!/usr/bin/env bash
set -euo pipefail

# Stage2.1-pure 1.5B smoke:
#   pytest -> data prep -> 25-step rows-only SFT -> small natural generation
#   -> exact-3/location gate. Judge/summary can be enabled after the gate.

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_STEPS="${MAX_STEPS:-25}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/experiment/stage21_pause_pure_dagger_1p5b.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_smoke_2xa6000.yaml}"
RUN_PYTEST="${RUN_PYTEST:-1}"
RUN_DATA_PREP="${RUN_DATA_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_GATE="${RUN_GATE:-1}"
RUN_JUDGE="${RUN_JUDGE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"
CONDITIONS="${CONDITIONS:-base_natural,stage21_pure_cot5_natural,stage21_pure_cot5_forced}"
NATURAL_CONDITION="${NATURAL_CONDITION:-stage21_pure_cot5_natural}"

export STAGE21_PURE_1P5B_SMOKE_CHECKPOINT="${STAGE21_PURE_1P5B_SMOKE_CHECKPOINT:-/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_save25_max400_2xa6000/checkpoint-${MAX_STEPS}}"

RUN_ROOT="${COT_SAFETY_RUN_ROOT:-/workspace/cot-safety/runs}"
SMOKE_EVAL_ROOT="${SMOKE_EVAL_ROOT:-${RUN_ROOT}/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_smoke_2xa6000}"
GATE_JSON="${GATE_JSON:-${SMOKE_EVAL_ROOT}/stage21_pure_natural_gate.json}"

echo "[stage21-pure-smoke] TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[stage21-pure-smoke] EVAL_CONFIG=${EVAL_CONFIG}"
echo "[stage21-pure-smoke] STAGE21_PURE_1P5B_SMOKE_CHECKPOINT=${STAGE21_PURE_1P5B_SMOKE_CHECKPOINT}"
echo "[stage21-pure-smoke] SMOKE_EVAL_ROOT=${SMOKE_EVAL_ROOT}"

if [[ "${RUN_PYTEST}" == "1" ]]; then
  "${PYTHON_BIN}" -m pytest \
    tests/test_stage2_pause_kl_trainer.py \
    tests/test_stage2_natural_pause_metrics.py \
    tests/test_stage2_onpolicy_mining.py \
    tests/test_stage2_dagger_mix.py \
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
    --skip_data_prep \
    --max_steps "${MAX_STEPS}"
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
    --input_jsonl "${SMOKE_EVAL_ROOT}/generations/${NATURAL_CONDITION}_capability.jsonl" \
    --input_jsonl "${SMOKE_EVAL_ROOT}/generations/${NATURAL_CONDITION}_safety.jsonl" \
    --output_json "${GATE_JSON}" \
    --generation_field generated \
    --use_existing_metrics \
    --strict
  echo "[stage21-pure-smoke] gate written to ${GATE_JSON}"
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
