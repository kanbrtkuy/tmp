#!/usr/bin/env bash
set -euo pipefail

# Full intra-pause steering evaluation launcher.
#
# This launcher is designed for independent GPUs.  It shards work by
# dataset x seed x alpha, applies interventions only on pause_0/1/2 via
# run_intra_pause_steered_generation.py, runs open judges, and then summarizes
# all shards.  It never modifies pre_pause_* or post_pause_* positions.

ROOT="${ROOT:-/workspace/PauseProbe}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-/workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final}"
DELTA="${DELTA:-/workspace/PauseProbe/runs/steering/intra_pause_learned_delta_260618/zero_l14_steps80/learned_delta.pt}"
OUT_ROOT="${OUT_ROOT:-/workspace/PauseProbe/runs/steering/intra_pause_full_steering_eval_260621}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
BASE_OUT_ROOT="${OUT_ROOT}"

DEVICES="${DEVICES:-0,1}"
SEEDS="${SEEDS:-260621 260622 260623}"
ALPHAS="${ALPHAS:-0,1,2}"
TARGET_SPECS="${TARGET_SPECS:-\
all3|pause_0,pause_1,pause_2
pause0_only|pause_0
pause1_only|pause_1
pause2_only|pause_2}"
TARGET_NAME="${TARGET_NAME:-all3}"
TARGET_POSITIONS="${TARGET_POSITIONS:-pause_0,pause_1,pause_2}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-4096}"
LAYER="${LAYER:-14}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

STAGE4_JUDGE_BACKEND="${STAGE4_JUDGE_BACKEND:-vllm}"
JUDGES="${JUDGES:-wildguard}"
NORMALIZED_FILENAME="${NORMALIZED_FILENAME:-open_judges_normalized.jsonl}"
RAW_FILENAME="${RAW_FILENAME:-open_judges_raw.jsonl}"
JUDGE_STRATEGY="${JUDGE_STRATEGY:-conservative}"
VLLM_JUDGE_WORKER_SCRIPT="${VLLM_JUDGE_WORKER_SCRIPT:-scripts/judge/run_vllm_dynamic_open_judges.py}"
WILDGUARD_MODEL="${WILDGUARD_MODEL:-${COT_SAFETY_JUDGE_ROOT:-/workspace/models/judges}/wildguard}"
LLAMAGUARD_MODEL="${LLAMAGUARD_MODEL:-${COT_SAFETY_JUDGE_ROOT:-/workspace/models/judges}/Llama-Guard-3-8B}"
HARMBENCH_MODEL="${HARMBENCH_MODEL:-${COT_SAFETY_JUDGE_ROOT:-/workspace/models/judges}/HarmBench-Llama-2-13b-cls}"
VLLM_JUDGE_GPU_MEMORY_UTILIZATION="${VLLM_JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_JUDGE_MAX_NUM_SEQS="${VLLM_JUDGE_MAX_NUM_SEQS:-32}"
VLLM_JUDGE_DTYPE="${VLLM_JUDGE_DTYPE:-${TORCH_DTYPE}}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
ALLOW_MISSING_DATASETS="${ALLOW_MISSING_DATASETS:-0}"
REUSE_ALPHA0_FROM_TARGET="${REUSE_ALPHA0_FROM_TARGET:-}"

# Format: name|input_file|label_filter|rows_per_label
# label_filter is one of all/safe/unsafe.  For label_filter=all, each shard
# samples up to rows_per_label safe + rows_per_label unsafe rows.
DATASET_SPECS="${DATASET_SPECS:-}"
DATASET_SPECS_FILE="${DATASET_SPECS_FILE:-}"

cd "${ROOT}"
mkdir -p "${BASE_OUT_ROOT}" "${HF_HOME}"
export HF_HOME

if [[ -z "${DATASET_SPECS}" && -z "${DATASET_SPECS_FILE}" ]]; then
  echo "DATASET_SPECS or DATASET_SPECS_FILE must be set. Prefer scripts/run_stage4_steering.py with eval.dataset_specs in config." >&2
  exit 2
fi

IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
if [[ "${#DEVICE_ARRAY[@]}" -lt 1 ]]; then
  echo "DEVICES must contain at least one GPU id." >&2
  exit 2
fi
MAX_PARALLEL_GENERATION_JOBS="${MAX_PARALLEL_GENERATION_JOBS:-${#DEVICE_ARRAY[@]}}"
MAX_PARALLEL_JUDGE_JOBS="${MAX_PARALLEL_JUDGE_JOBS:-${#DEVICE_ARRAY[@]}}"

IFS=',' read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a JUDGE_ARRAY <<< "${JUDGES}"

if [[ "${STAGE4_JUDGE_BACKEND}" != "vllm" && "${STAGE4_JUDGE_BACKEND}" != "transformers" ]]; then
  echo "STAGE4_JUDGE_BACKEND must be vllm or transformers, got: ${STAGE4_JUDGE_BACKEND}" >&2
  exit 2
fi

alpha_slug() {
  local alpha="$1"
  alpha="${alpha//-/m}"
  alpha="${alpha//./p}"
  echo "${alpha}"
}

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

link_file_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    rm -f "${dst}"
    ln -s "${src}" "${dst}"
  fi
}

reuse_alpha0_generation_if_available() {
  local dataset="$1"
  local seed="$2"
  local gen_file="$3"
  local slug="0"

  [[ -n "${REUSE_ALPHA0_FROM_TARGET}" ]] || return 1
  [[ "${TARGET_NAME}" != "${REUSE_ALPHA0_FROM_TARGET}" ]] || return 1

  local src_dir="${BASE_OUT_ROOT}/${REUSE_ALPHA0_FROM_TARGET}/${dataset}/seed_${seed}/alpha_${slug}"
  local src_gen="${src_dir}/generations.jsonl"
  local src_manifest="${src_dir}/generations.manifest.json"

  generation_complete "${src_gen}" || return 1
  link_file_if_exists "${src_gen}" "${gen_file}"
  link_file_if_exists "${src_manifest}" "${gen_file%.jsonl}.manifest.json"
  echo "[generation reuse alpha0] target=${TARGET_NAME} from=${REUSE_ALPHA0_FROM_TARGET} dataset=${dataset} seed=${seed} rows=$(count_lines "${gen_file}")"
  return 0
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

reuse_alpha0_judge_if_available() {
  local dataset="$1"
  local seed="$2"
  local gen_file="$3"
  local raw_file="$4"
  local norm_file="$5"
  local slug="0"

  [[ -n "${REUSE_ALPHA0_FROM_TARGET}" ]] || return 1
  [[ "${TARGET_NAME}" != "${REUSE_ALPHA0_FROM_TARGET}" ]] || return 1

  local src_dir="${BASE_OUT_ROOT}/${REUSE_ALPHA0_FROM_TARGET}/${dataset}/seed_${seed}/alpha_${slug}"
  local src_gen="${src_dir}/generations.jsonl"
  local src_raw="${src_dir}/${RAW_FILENAME}"
  local src_norm="${src_dir}/${NORMALIZED_FILENAME}"

  judge_complete "${src_gen}" "${src_norm}" || return 1
  link_file_if_exists "${src_gen}" "${gen_file}"
  link_file_if_exists "${src_dir}/generations.manifest.json" "${gen_file%.jsonl}.manifest.json"
  link_file_if_exists "${src_raw}" "${raw_file}"
  link_file_if_exists "${src_raw%.jsonl}.manifest.json" "${raw_file%.jsonl}.manifest.json"
  link_file_if_exists "${src_norm}" "${norm_file}"
  link_file_if_exists "${src_norm%.jsonl}.manifest.json" "${norm_file%.jsonl}.manifest.json"
  echo "[judge reuse alpha0] target=${TARGET_NAME} from=${REUSE_ALPHA0_FROM_TARGET} dataset=${dataset} seed=${seed} rows=$(count_lines "${norm_file}")"
  return 0
}

dataset_specs() {
  if [[ -n "${DATASET_SPECS_FILE}" ]]; then
    grep -v '^[[:space:]]*#' "${DATASET_SPECS_FILE}" | grep -v '^[[:space:]]*$'
  else
    printf '%s\n' "${DATASET_SPECS}" | grep -v '^[[:space:]]*$'
  fi
}

task_safe_name() {
  printf '%s' "$*" | tr -c '[:alnum:]_.=-' '_'
}

worker_count_for_limit() {
  local limit="$1"
  local devices="${#DEVICE_ARRAY[@]}"
  if [[ "${limit}" -lt 1 ]]; then
    echo 1
  elif [[ "${limit}" -lt "${devices}" ]]; then
    echo "${limit}"
  else
    echo "${devices}"
  fi
}

claim_next_task() {
  local queue_root="$1"
  local worker_id="$2"
  local task
  mkdir -p "${queue_root}/running"
  for task in "${queue_root}/pending"/*.task; do
    [[ -e "${task}" ]] || return 1
    local claimed="${queue_root}/running/${worker_id}_$(basename "${task}")"
    if mv "${task}" "${claimed}" 2>/dev/null; then
      printf '%s\n' "${claimed}"
      return 0
    fi
  done
  return 1
}

finish_claimed_task() {
  local queue_root="$1"
  local claimed="$2"
  local status="$3"
  mkdir -p "${queue_root}/${status}"
  mv "${claimed}" "${queue_root}/${status}/$(basename "${claimed}")"
}

target_specs() {
  printf '%s\n' "${TARGET_SPECS}" | grep -v '^[[:space:]]*$'
}

run_generation_job() {
  local gpu="$1"
  local dataset="$2"
  local input_file="$3"
  local label_filter="$4"
  local rows_per_label="$5"
  local seed="$6"
  local alpha="$7"
  local slug
  slug="$(alpha_slug "${alpha}")"
  local out_dir="${OUT_ROOT}/${dataset}/seed_${seed}/alpha_${slug}"
  local gen_file="${out_dir}/generations.jsonl"
  local log_file="${out_dir}/generate.log"
  mkdir -p "${out_dir}"

  if [[ "${alpha}" == "0" || "${alpha}" == "0.0" ]]; then
    if reuse_alpha0_generation_if_available "${dataset}" "${seed}" "${gen_file}"; then
      return 0
    fi
  fi

  if [[ ! -f "${input_file}" ]]; then
    if [[ "${ALLOW_MISSING_DATASETS}" == "1" ]]; then
      echo "[generation skip missing] ${dataset}: ${input_file}" | tee -a "${OUT_ROOT}/logs/missing_datasets.log"
      return 0
    fi
    echo "[generation missing] ${dataset}: ${input_file}" >&2
    return 2
  fi

  if generation_complete "${gen_file}"; then
    echo "[generation skip complete] dataset=${dataset} seed=${seed} alpha=${alpha} rows=$(count_lines "${gen_file}")"
    return 0
  fi

  rm -f "${gen_file}" "${gen_file%.jsonl}.manifest.json"
  echo "[generation start] gpu=${gpu} dataset=${dataset} label=${label_filter} seed=${seed} alpha=${alpha}"
  HF_HOME="${HF_HOME}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/steering/run_intra_pause_steered_generation.py \
    --model "${MODEL}" \
    --delta_checkpoint "${DELTA}" \
    --input_file "${input_file}" \
    --output_jsonl "${gen_file}" \
    --model_label deepseek_intra_pause_cot3_sft \
    --run_label "full_steering_${TARGET_NAME}_${dataset}_seed${seed}_alpha${slug}" \
    --layer "${LAYER}" \
    --alphas="${alpha}" \
    --target_positions "${TARGET_POSITIONS}" \
    --rows_per_label "${rows_per_label}" \
    --label_filter "${label_filter}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --insert_pause_after_cot_tokens 3 \
    --n_insert_pauses 3 \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${seed}" \
    --torch_dtype "${TORCH_DTYPE}" \
    > "${log_file}" 2>&1
  echo "[generation done] dataset=${dataset} seed=${seed} alpha=${alpha} rows=$(count_lines "${gen_file}")"
}

run_judge_job() {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local alpha="$4"
  local slug
  slug="$(alpha_slug "${alpha}")"
  local out_dir="${OUT_ROOT}/${dataset}/seed_${seed}/alpha_${slug}"
  local gen_file="${out_dir}/generations.jsonl"
  local raw_file="${out_dir}/${RAW_FILENAME}"
  local norm_file="${out_dir}/${NORMALIZED_FILENAME}"
  local judge_log="${out_dir}/judge.log"
  local normalize_log="${out_dir}/normalize.log"

  if [[ "${alpha}" == "0" || "${alpha}" == "0.0" ]]; then
    if reuse_alpha0_judge_if_available "${dataset}" "${seed}" "${gen_file}" "${raw_file}" "${norm_file}"; then
      return 0
    fi
  fi

  if [[ ! -f "${gen_file}" ]]; then
    echo "[judge skip missing generation] dataset=${dataset} seed=${seed} alpha=${alpha}" >&2
    return 0
  fi

  if judge_complete "${gen_file}" "${norm_file}"; then
    echo "[judge skip complete] dataset=${dataset} seed=${seed} alpha=${alpha} rows=$(count_lines "${norm_file}")"
    return 0
  fi

  rm -f "${raw_file}" "${raw_file%.jsonl}.manifest.json" "${norm_file}" "${norm_file%.jsonl}.manifest.json"
  echo "[judge start] gpu=${gpu} dataset=${dataset} seed=${seed} alpha=${alpha} judges=${JUDGES}"
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
  echo "[judge done] dataset=${dataset} seed=${seed} alpha=${alpha} rows=$(count_lines "${norm_file}")"
}

run_dynamic_worker() {
  local stage="$1"
  local queue_root="$2"
  local gpu="$3"
  local worker_id="$4"
  local claimed

  while claimed="$(claim_next_task "${queue_root}" "${worker_id}")"; do
    if [[ "${stage}" == "generation" ]]; then
      local dataset input_file label_filter rows_per_label seed alpha
      IFS='|' read -r dataset input_file label_filter rows_per_label seed alpha < "${claimed}"
      if run_generation_job "${gpu}" "${dataset}" "${input_file}" "${label_filter}" "${rows_per_label}" "${seed}" "${alpha}"; then
        finish_claimed_task "${queue_root}" "${claimed}" "done"
      else
        finish_claimed_task "${queue_root}" "${claimed}" "failed"
        return 1
      fi
    else
      local dataset seed alpha
      IFS='|' read -r dataset seed alpha < "${claimed}"
      if run_judge_job "${gpu}" "${dataset}" "${seed}" "${alpha}"; then
        finish_claimed_task "${queue_root}" "${claimed}" "done"
      else
        finish_claimed_task "${queue_root}" "${claimed}" "failed"
        return 1
      fi
    fi
  done
}

run_dynamic_pool() {
  local stage="$1"
  local limit="$2"
  local queue_root="${OUT_ROOT}/logs/${stage}_queue_$$"
  local worker_count
  worker_count="$(worker_count_for_limit "${limit}")"

  rm -rf "${queue_root}"
  mkdir -p "${queue_root}/pending" "${queue_root}/running" "${queue_root}/done" "${queue_root}/failed"
  while IFS='|' read -r dataset input_file label_filter rows_per_label; do
    [[ -n "${dataset}" ]] || continue
    for seed in "${SEED_ARRAY[@]}"; do
      for alpha in "${ALPHA_ARRAY[@]}"; do
        if [[ "${stage}" == "generation" ]]; then
          printf '%s|%s|%s|%s|%s|%s\n' "${dataset}" "${input_file}" "${label_filter}" "${rows_per_label}" "${seed}" "${alpha}" \
            > "${queue_root}/pending/$(task_safe_name "${dataset}_${seed}_${alpha}").task"
        else
          printf '%s|%s|%s\n' "${dataset}" "${seed}" "${alpha}" \
            > "${queue_root}/pending/$(task_safe_name "${dataset}_${seed}_${alpha}").task"
        fi
      done
    done
  done < <(dataset_specs)

  local failed=0
  for idx in $(seq 0 $((worker_count - 1))); do
    local gpu="${DEVICE_ARRAY[$idx]}"
    run_dynamic_worker "${stage}" "${queue_root}" "${gpu}" "gpu${gpu}_${stage}" > "${OUT_ROOT}/logs/${stage}_gpu${gpu}.log" 2>&1 &
  done

  for _ in $(seq 1 "${worker_count}"); do
    if ! wait -n; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[${stage}] one or more jobs failed" >&2
    exit 1
  fi
}

run_vllm_judge_pool() {
  if [[ "${#JUDGE_ARRAY[@]}" -ne 1 ]]; then
    echo "STAGE4_JUDGE_BACKEND=vllm currently expects exactly one judge for ${NORMALIZED_FILENAME}; got: ${JUDGES}" >&2
    echo "Use STAGE4_JUDGE_BACKEND=transformers for multi-judge ensemble output, or run pipelines/run_stage4_second_judges_vllm_dynamic.sh for per-judge second passes." >&2
    exit 2
  fi
  local judge="${JUDGE_ARRAY[0]}"
  if [[ "${judge}" != "wildguard" && "${judge}" != "llamaguard" && "${judge}" != "harmbench" ]]; then
    echo "STAGE4_JUDGE_BACKEND=vllm supports wildguard/llamaguard/harmbench, got: ${judge}" >&2
    exit 2
  fi

  local queue_root="${OUT_ROOT}/logs/vllm_judge_queue_$$"
  local worker_count
  worker_count="$(worker_count_for_limit "${MAX_PARALLEL_JUDGE_JOBS}")"
  rm -rf "${queue_root}"
  mkdir -p "${queue_root}/pending/${judge}" "${queue_root}/running" "${queue_root}/done" "${queue_root}/failed"

  while IFS='|' read -r dataset _input_file _label_filter _rows_per_label; do
    [[ -n "${dataset}" ]] || continue
    for seed in "${SEED_ARRAY[@]}"; do
      for alpha in "${ALPHA_ARRAY[@]}"; do
        local slug out_dir gen_file raw_file norm_file key
        slug="$(alpha_slug "${alpha}")"
        out_dir="${OUT_ROOT}/${dataset}/seed_${seed}/alpha_${slug}"
        gen_file="${out_dir}/generations.jsonl"
        raw_file="${out_dir}/${RAW_FILENAME}"
        norm_file="${out_dir}/${NORMALIZED_FILENAME}"
        if [[ "${alpha}" == "0" || "${alpha}" == "0.0" ]]; then
          reuse_alpha0_judge_if_available "${dataset}" "${seed}" "${gen_file}" "${raw_file}" "${norm_file}" && continue
        fi
        [[ -f "${gen_file}" ]] || continue
        judge_complete "${gen_file}" "${norm_file}" && continue
        rm -f "${raw_file}" "${raw_file%.jsonl}.manifest.json" "${norm_file}" "${norm_file%.jsonl}.manifest.json"
        key="$(task_safe_name "${dataset}_${seed}_${alpha}_${judge}")"
        "${PYTHON}" - "${queue_root}/pending/${judge}/${key}.json" "${judge}" "${gen_file}" "${raw_file}" "${norm_file}" <<'PY'
import json
import sys
from pathlib import Path
path, judge, gen, raw, norm = sys.argv[1:]
Path(path).write_text(json.dumps({"judge": judge, "gen": gen, "raw": raw, "norm": norm}, indent=2), encoding="utf-8")
PY
      done
    done
  done < <(dataset_specs)

  local model_map_json="{\"wildguard\":\"${WILDGUARD_MODEL}\",\"llamaguard\":\"${LLAMAGUARD_MODEL}\",\"harmbench\":\"${HARMBENCH_MODEL}\"}"
  local max_model_len_json="{\"wildguard\":${JUDGE_MAX_INPUT_LENGTH},\"llamaguard\":${JUDGE_MAX_INPUT_LENGTH},\"harmbench\":${JUDGE_MAX_INPUT_LENGTH}}"
  local failed=0
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  for idx in $(seq 0 $((worker_count - 1))); do
    local gpu="${DEVICE_ARRAY[$idx]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${VLLM_JUDGE_WORKER_SCRIPT}" \
      --queue_root "${queue_root}" \
      --worker_id "gpu${gpu}_vllm_judge" \
      --preferred_judges "${judge}" \
      --model_map_json "${model_map_json}" \
      --max_model_len_json "${max_model_len_json}" \
      --gpu_memory_utilization "${VLLM_JUDGE_GPU_MEMORY_UTILIZATION}" \
      --max_num_seqs "${VLLM_JUDGE_MAX_NUM_SEQS}" \
      --dtype "${VLLM_JUDGE_DTYPE}" \
      --strategy "${JUDGE_STRATEGY}" \
      > "${OUT_ROOT}/logs/vllm_judge_gpu${gpu}.log" 2>&1 &
  done
  for _ in $(seq 1 "${worker_count}"); do
    if ! wait -n; then
      failed=1
    fi
  done
  local failed_count
  failed_count="$(find "${queue_root}/failed" -type f -name '*.json' | wc -l | tr -d ' ')"
  if [[ "${failed}" -ne 0 || "${failed_count}" != "0" ]]; then
    echo "[vllm judge] one or more jobs failed: ${failed_count}" >&2
    exit 1
  fi
}

run_target() {
  local target_name="$1"
  local target_positions="$2"
  TARGET_NAME="${target_name}"
  TARGET_POSITIONS="${target_positions}"
  OUT_ROOT="${BASE_OUT_ROOT}/${TARGET_NAME}"
  export OUT_ROOT TARGET_NAME TARGET_POSITIONS
  mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"

  {
    echo "root=${ROOT}"
    echo "model=${MODEL}"
    echo "delta=${DELTA}"
    echo "base_out_root=${BASE_OUT_ROOT}"
    echo "out_root=${OUT_ROOT}"
    echo "target_name=${TARGET_NAME}"
    echo "target_positions=${TARGET_POSITIONS}"
    echo "devices=${DEVICES}"
    echo "seeds=${SEEDS}"
    echo "alphas=${ALPHAS}"
    echo "judges=${JUDGES}"
    echo "stage4_judge_backend=${STAGE4_JUDGE_BACKEND}"
    echo "dataset_specs:"
    dataset_specs
  } > "${OUT_ROOT}/run_config.txt"

  if [[ "${RUN_GENERATION}" == "1" ]]; then
    run_dynamic_pool generation "${MAX_PARALLEL_GENERATION_JOBS}"
  fi

  if [[ "${RUN_JUDGE}" == "1" ]]; then
    if [[ "${STAGE4_JUDGE_BACKEND}" == "vllm" ]]; then
      run_vllm_judge_pool
    else
      run_dynamic_pool judge "${MAX_PARALLEL_JUDGE_JOBS}"
    fi
  fi

  if [[ "${RUN_SUMMARY}" == "1" ]]; then
    "${PYTHON}" scripts/steering/summarize_intra_pause_full_steering_eval.py \
      --out_root "${OUT_ROOT}" \
      --normalized_filename "${NORMALIZED_FILENAME}"
  fi

  echo "[done] ${OUT_ROOT}"
}

while IFS='|' read -r target_name target_positions; do
  [[ -n "${target_name}" && -n "${target_positions}" ]] || continue
  run_target "${target_name}" "${target_positions}"
done < <(target_specs)

echo "[done all targets] ${BASE_OUT_ROOT}"
