#!/usr/bin/env bash
set -euo pipefail

# Base-model hard-safe / unsafe reference evaluation.
#
# This launcher evaluates the original DeepSeek-R1-Distill-Qwen-1.5B model on
# the same prompt-only hard-safe audit rows used by the intra-pause steering
# audit.  It does not insert pause tokens and does not apply steering.  This is
# a model-level reference baseline, not the alpha=0 steering ablation baseline.

ROOT="${ROOT:-/workspace/PauseProbe}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B}"
FALLBACK_MODEL="${FALLBACK_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_LABEL="${MODEL_LABEL:-deepseek_r1_distill_qwen_1p5b_base}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/intra_pause_hardsafe_steering_eval_260621}"
OUT_ROOT="${OUT_ROOT:-/workspace/PauseProbe/runs/steering/base_model_hardsafe_reference_eval_260621}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PREPARE_DATA="${PREPARE_DATA:-0}"
DEVICES="${DEVICES:-0,1}"
SEEDS="${SEEDS:-260621 260622 260623}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

JUDGES="${JUDGES:-wildguard}"
NORMALIZED_FILENAME="${NORMALIZED_FILENAME:-open_judges_normalized.jsonl}"
RAW_FILENAME="${RAW_FILENAME:-open_judges_raw.jsonl}"
JUDGE_STRATEGY="${JUDGE_STRATEGY:-conservative}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
ALLOW_MISSING_DATASETS="${ALLOW_MISSING_DATASETS:-0}"
MAX_PARALLEL_GENERATION_JOBS="${MAX_PARALLEL_GENERATION_JOBS:-}"
MAX_PARALLEL_JUDGE_JOBS="${MAX_PARALLEL_JUDGE_JOBS:-}"

cd "${ROOT}"
mkdir -p "${DATA_ROOT}" "${OUT_ROOT}" "${HF_HOME}" "${OUT_ROOT}/logs"
export HF_HOME OUT_ROOT

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON}" scripts/data/prepare_intra_pause_hardsafe_steering_data.py \
    --output_dir "${DATA_ROOT}"
fi

if [[ ! -e "${MODEL}" ]]; then
  echo "[model fallback] ${MODEL} not found; using ${FALLBACK_MODEL}" | tee -a "${OUT_ROOT}/logs/model_fallback.log"
  MODEL="${FALLBACK_MODEL}"
fi

DATASET_SPECS_FILE="${DATASET_SPECS_FILE:-${DATA_ROOT}/dataset_specs.tsv}"

IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
if [[ "${#DEVICE_ARRAY[@]}" -lt 1 ]]; then
  echo "DEVICES must contain at least one GPU id." >&2
  exit 2
fi
if [[ -z "${MAX_PARALLEL_GENERATION_JOBS}" ]]; then
  MAX_PARALLEL_GENERATION_JOBS="${#DEVICE_ARRAY[@]}"
fi
if [[ -z "${MAX_PARALLEL_JUDGE_JOBS}" ]]; then
  MAX_PARALLEL_JUDGE_JOBS="${#DEVICE_ARRAY[@]}"
fi

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a JUDGE_ARRAY <<< "${JUDGES}"

count_lines() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    wc -l < "${file}" | tr -d ' '
  else
    echo 0
  fi
}

generation_complete() {
  local gen_file="$1"
  local manifest_file="${gen_file%.jsonl}.manifest.json"
  [[ -f "${gen_file}" && -f "${manifest_file}" ]] || return 1
  "${PYTHON}" - "${gen_file}" "${manifest_file}" <<'PY'
import json
import sys
from pathlib import Path

gen_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
line_count = sum(1 for line in gen_path.open(encoding="utf-8") if line.strip())
manifest = json.load(manifest_path.open(encoding="utf-8"))
expected = int(manifest.get("num_generations") or 0)
if expected > 0 and line_count == expected:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

judge_complete() {
  local gen_file="$1"
  local norm_file="$2"
  [[ -f "${gen_file}" && -f "${norm_file}" ]] || return 1
  local gen_rows
  local norm_rows
  gen_rows="$(count_lines "${gen_file}")"
  norm_rows="$(count_lines "${norm_file}")"
  [[ "${gen_rows}" -gt 0 && "${gen_rows}" -eq "${norm_rows}" ]]
}

dataset_specs() {
  grep -v '^[[:space:]]*#' "${DATASET_SPECS_FILE}" | grep -v '^[[:space:]]*$'
}

run_generation_job() {
  local gpu="$1"
  local dataset="$2"
  local input_file="$3"
  local label_filter="$4"
  local rows_per_label="$5"
  local seed="$6"
  local out_dir="${OUT_ROOT}/${dataset}/seed_${seed}/alpha_0"
  local gen_file="${out_dir}/generations.jsonl"
  local log_file="${out_dir}/generate.log"
  mkdir -p "${out_dir}"

  if [[ ! -f "${input_file}" ]]; then
    if [[ "${ALLOW_MISSING_DATASETS}" == "1" ]]; then
      echo "[generation skip missing] ${dataset}: ${input_file}" | tee -a "${OUT_ROOT}/logs/missing_datasets.log"
      return 0
    fi
    echo "[generation missing] ${dataset}: ${input_file}" >&2
    return 2
  fi

  if generation_complete "${gen_file}"; then
    echo "[generation skip complete] dataset=${dataset} seed=${seed} rows=$(count_lines "${gen_file}")"
    return 0
  fi

  rm -f "${gen_file}" "${gen_file%.jsonl}.manifest.json"
  echo "[generation start] gpu=${gpu} dataset=${dataset} label=${label_filter} seed=${seed}"
  HF_HOME="${HF_HOME}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/steering/run_base_model_generation.py \
    --model "${MODEL}" \
    --input_file "${input_file}" \
    --output_jsonl "${gen_file}" \
    --model_label "${MODEL_LABEL}" \
    --run_label "base_reference_${dataset}_seed${seed}" \
    --rows_per_label "${rows_per_label}" \
    --label_filter "${label_filter}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${seed}" \
    --torch_dtype "${TORCH_DTYPE}" \
    > "${log_file}" 2>&1
  echo "[generation done] dataset=${dataset} seed=${seed} rows=$(count_lines "${gen_file}")"
}

run_judge_job() {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local out_dir="${OUT_ROOT}/${dataset}/seed_${seed}/alpha_0"
  local gen_file="${out_dir}/generations.jsonl"
  local raw_file="${out_dir}/${RAW_FILENAME}"
  local norm_file="${out_dir}/${NORMALIZED_FILENAME}"
  local judge_log="${out_dir}/judge.log"
  local normalize_log="${out_dir}/normalize.log"

  if [[ ! -f "${gen_file}" ]]; then
    echo "[judge skip missing generation] dataset=${dataset} seed=${seed}" >&2
    return 0
  fi

  if judge_complete "${gen_file}" "${norm_file}"; then
    echo "[judge skip complete] dataset=${dataset} seed=${seed} rows=$(count_lines "${norm_file}")"
    return 0
  fi

  rm -f "${raw_file}" "${raw_file%.jsonl}.manifest.json" "${norm_file}" "${norm_file%.jsonl}.manifest.json"
  echo "[judge start] gpu=${gpu} dataset=${dataset} seed=${seed} judges=${JUDGES}"
  HF_HOME="${HF_HOME}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/judge/run_open_judges.py \
    --input_file "${gen_file}" \
    --output_jsonl "${raw_file}" \
    --judges "${JUDGE_ARRAY[@]}" \
    --batch_size "${JUDGE_BATCH_SIZE}" \
    --max_input_length "${JUDGE_MAX_INPUT_LENGTH}" \
    --torch_dtype "${TORCH_DTYPE}" \
    > "${judge_log}" 2>&1

  "${PYTHON}" scripts/judge/normalize_judge_outputs.py \
    --input_file "${raw_file}" \
    --output_jsonl "${norm_file}" \
    --strategy "${JUDGE_STRATEGY}" \
    > "${normalize_log}" 2>&1
  echo "[judge done] dataset=${dataset} seed=${seed} rows=$(count_lines "${norm_file}")"
}

run_pool() {
  local stage="$1"
  local limit="$2"
  local active=0
  local failed=0
  local job_index=0

  while IFS='|' read -r dataset input_file label_filter rows_per_label; do
    [[ -n "${dataset}" ]] || continue
    for seed in "${SEED_ARRAY[@]}"; do
      local gpu="${DEVICE_ARRAY[$((job_index % ${#DEVICE_ARRAY[@]}))]}"
      if [[ "${stage}" == "generation" ]]; then
        run_generation_job "${gpu}" "${dataset}" "${input_file}" "${label_filter}" "${rows_per_label}" "${seed}" &
      else
        run_judge_job "${gpu}" "${dataset}" "${seed}" &
      fi
      active=$((active + 1))
      job_index=$((job_index + 1))
      if [[ "${active}" -ge "${limit}" ]]; then
        if ! wait -n; then
          failed=1
        fi
        active=$((active - 1))
      fi
    done
  done < <(dataset_specs)

  while [[ "${active}" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[${stage}] one or more jobs failed" >&2
    exit 1
  fi
}

{
  echo "root=${ROOT}"
  echo "model=${MODEL}"
  echo "model_label=${MODEL_LABEL}"
  echo "out_root=${OUT_ROOT}"
  echo "data_root=${DATA_ROOT}"
  echo "dataset_specs_file=${DATASET_SPECS_FILE}"
  echo "devices=${DEVICES}"
  echo "seeds=${SEEDS}"
  echo "judges=${JUDGES}"
  echo "baseline_role=model_level_reference"
  echo "notes=base model, no pause insertion, no steering; compare against intra-pause SFT alpha=0 and alpha>0 separately"
  echo "dataset_specs:"
  dataset_specs
} > "${OUT_ROOT}/run_config.txt"

if [[ "${RUN_GENERATION}" == "1" ]]; then
  run_pool generation "${MAX_PARALLEL_GENERATION_JOBS}"
fi

if [[ "${RUN_JUDGE}" == "1" ]]; then
  run_pool judge "${MAX_PARALLEL_JUDGE_JOBS}"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  "${PYTHON}" scripts/steering/summarize_intra_pause_full_steering_eval.py \
    --out_root "${OUT_ROOT}" \
    --normalized_filename "${NORMALIZED_FILENAME}"
fi

echo "[done] ${OUT_ROOT}"
