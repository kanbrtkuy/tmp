#!/usr/bin/env bash
set -euo pipefail

cd /workspace/cot-safety

export PATH=/workspace/venvs/stage1/bin:${PATH}
export HF_HOME=/workspace/hf-cache
export TRANSFORMERS_CACHE=/workspace/hf-cache
export COT_SAFETY_COLD_ROOT=/workspace
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
export COT_SAFETY_RUN_ROOT=/workspace/cot-safety/runs
export COT_SAFETY_MODEL_ROOT=/workspace/models
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/logs
CLEAN_HOT_AFTER_BACKUP=${CLEAN_HOT_AFTER_BACKUP:-1}
mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

backup_run() {
  local backup_root="$1"
  local run_name="$2"
  local hidden_name="$3"

  mkdir -p "${backup_root}/runs/hidden" "${backup_root}/runs/logs"
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/${run_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/${run_name}/" "${backup_root}/runs/${run_name}/"
  fi
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/hidden/${hidden_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/hidden/${hidden_name}/" "${backup_root}/runs/hidden/${hidden_name}/"
  fi
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}/" "${backup_root}/runs/logs/${run_name}/"
  fi

  if [ "${CLEAN_HOT_AFTER_BACKUP}" = "1" ]; then
    rm -rf \
      "${COT_SAFETY_HOT_ROOT}/runs/${run_name}" \
      "${COT_SAFETY_HOT_ROOT}/runs/hidden/${hidden_name}" \
      "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}"
  fi
}

run_dense_pair() {
  local label="$1"
  local data_dir="$2"
  local stage1_config="$3"
  local stage1b_config="$4"
  local backup_root="$5"
  local stage1_run="$6"
  local stage1_hidden="$7"
  local stage1b_run="$8"
  local stage1b_hidden="$9"

  export NATURAL_STAGE1_DATA_DIR="${data_dir}"

  log "Dense Stage1 suite start: ${label}"
  log "Data: ${NATURAL_STAGE1_DATA_DIR}"
  log "Stage1 config: ${stage1_config}"
  log "Stage1b config: ${stage1b_config}"

  if [ ! -f "${NATURAL_STAGE1_DATA_DIR}/cotpause/train.json" ]; then
    log "Missing Stage1 train split: ${NATURAL_STAGE1_DATA_DIR}/cotpause/train.json"
    exit 2
  fi
  if [ ! -f "${NATURAL_STAGE1_DATA_DIR}/cotpause/val.json" ]; then
    log "Missing Stage1 val split: ${NATURAL_STAGE1_DATA_DIR}/cotpause/val.json"
    exit 2
  fi
  if [ ! -f "${NATURAL_STAGE1_DATA_DIR}/cotpause/test.json" ]; then
    log "Missing Stage1 test split: ${NATURAL_STAGE1_DATA_DIR}/cotpause/test.json"
    exit 2
  fi

  /workspace/venvs/stage1/bin/python scripts/run_stage1_positionscan.py \
    --config "${stage1_config}" \
    --python /workspace/venvs/stage1/bin/python \
    --skip_existing

  log "Backing up dense Stage1 artifacts: ${label}"
  backup_run "${backup_root}" "${stage1_run}" "${stage1_hidden}"

  /workspace/venvs/stage1/bin/python scripts/run_stage1b_prompt_baseline.py \
    --config "${stage1b_config}" \
    --python /workspace/venvs/stage1/bin/python \
    --skip_existing

  log "Backing up dense Stage1b artifacts: ${label}"
  backup_run "${backup_root}" "${stage1b_run}" "${stage1b_hidden}"
  log "Dense Stage1 suite complete: ${label}"
}

/workspace/venvs/stage1/bin/python -m py_compile \
  scripts/run_stage1_positionscan.py \
  scripts/run_stage1b_prompt_baseline.py \
  legacy/PauseProbe/scripts/probe/run_position_scan_full.py \
  legacy/PauseProbe/scripts/probe/extract_hidden_states.py \
  legacy/PauseProbe/scripts/probe/run_position_scan_batched.py \
  legacy/PauseProbe/scripts/probe/train_probe.py

RUN_DENSE_TARGETS=${RUN_DENSE_TARGETS:-8bgen_8b,32bgen_8b,32bgen_32b}

IFS=',' read -r -a TARGETS <<< "${RUN_DENSE_TARGETS}"
for target in "${TARGETS[@]}"; do
  case "${target}" in
    8bgen_8b)
      run_dense_pair \
        "8B generated/generated pairs with 8B hidden extractor" \
        "/workspace/cot-safety/runs/natural_cot_pair_full_n50_v1_8b_generated_generated_stage1" \
        "configs/experiment/stage1_natural_pairs_dense_8b_a100_1x.yaml" \
        "configs/experiment/stage1b_natural_pairs_dense_8b_a100_1x.yaml" \
        "/workspace/cot-safety/runs/hot_backup_stage1_natural_pairs_dense_8b_a100_1x_latest" \
        "stage1_natural_pairs_dense_8b_a100_1x" \
        "stage1_natural_pairs_dense_8b_a100_1x" \
        "stage1b_natural_pairs_dense_8b_a100_1x" \
        "stage1b_natural_pairs_dense_8b_a100_1x"
      ;;
    32bgen_8b)
      run_dense_pair \
        "32B generated/generated pairs with 8B hidden extractor" \
        "/workspace/cot-safety/runs/natural_cot_pair_full_n50_v1_32b_generated_generated_stage1" \
        "configs/experiment/stage1_natural_pairs_32bgen_dense_8b_a100_1x.yaml" \
        "configs/experiment/stage1b_natural_pairs_32bgen_dense_8b_a100_1x.yaml" \
        "/workspace/cot-safety/runs/hot_backup_stage1_natural_pairs_32bgen_dense_8b_a100_1x_latest" \
        "stage1_natural_pairs_32bgen_dense_8b_a100_1x" \
        "stage1_natural_pairs_32bgen_dense_8b_a100_1x" \
        "stage1b_natural_pairs_32bgen_dense_8b_a100_1x" \
        "stage1b_natural_pairs_32bgen_dense_8b_a100_1x"
      ;;
    32bgen_32b)
      run_dense_pair \
        "32B generated/generated pairs with 32B hidden extractor" \
        "/workspace/cot-safety/runs/natural_cot_pair_full_n50_v1_32b_generated_generated_stage1" \
        "configs/experiment/stage1_natural_pairs_32bgen_dense_32b_a100_1x.yaml" \
        "configs/experiment/stage1b_natural_pairs_32bgen_dense_32b_a100_1x.yaml" \
        "/workspace/cot-safety/runs/hot_backup_stage1_natural_pairs_32bgen_dense_32b_a100_1x_latest" \
        "stage1_natural_pairs_32bgen_dense_32b_a100_1x" \
        "stage1_natural_pairs_32bgen_dense_32b_a100_1x" \
        "stage1b_natural_pairs_32bgen_dense_32b_a100_1x" \
        "stage1b_natural_pairs_32bgen_dense_32b_a100_1x"
      ;;
    *)
      log "Unknown dense target: ${target}"
      exit 2
      ;;
  esac
done

log "All requested dense Stage1 suites complete: ${RUN_DENSE_TARGETS}"
