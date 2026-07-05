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
export NATURAL_STAGE1_DATA_DIR=/workspace/cot-safety/runs/natural_cot_pair_full_n50_v1_8b_generated_generated_stage1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/logs
BACKUP_ROOT=/workspace/cot-safety/runs/hot_backup_stage1_natural_pairs_8b_a100_1x_latest
mkdir -p "${LOG_DIR}" "${BACKUP_ROOT}/runs/hidden" "${BACKUP_ROOT}/runs/logs"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

backup_run() {
  local run_name="$1"
  local hidden_name="$2"
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/${run_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/${run_name}/" "${BACKUP_ROOT}/runs/${run_name}/"
  fi
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/hidden/${hidden_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/hidden/${hidden_name}/" "${BACKUP_ROOT}/runs/hidden/${hidden_name}/"
  fi
  if [ -d "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}" ]; then
    rsync -a --delete "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}/" "${BACKUP_ROOT}/runs/logs/${run_name}/"
  fi
}

log "Stage1 natural generated/generated pairs with 8B hidden extractor start"
log "Data: ${NATURAL_STAGE1_DATA_DIR}"
log "Config: stage1_natural_pairs_8b_a100_1x.yaml"

/workspace/venvs/stage1/bin/python -m py_compile \
  scripts/run_stage1_positionscan.py \
  scripts/run_stage1b_prompt_baseline.py \
  legacy/PauseProbe/scripts/probe/run_position_scan_full.py \
  legacy/PauseProbe/scripts/probe/extract_hidden_states.py \
  legacy/PauseProbe/scripts/probe/run_position_scan_batched.py \
  legacy/PauseProbe/scripts/probe/train_probe.py

/workspace/venvs/stage1/bin/python scripts/run_stage1_positionscan.py \
  --config configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml \
  --python /workspace/venvs/stage1/bin/python \
  --skip_existing

log "Backing up Stage1 natural 8B-hidden hot artifacts"
backup_run "stage1_natural_pairs_8b_a100_1x" "stage1_natural_pairs_8b_a100_1x"

log "Stage1b natural 8B-hidden prompt/pre-CoT baseline start"
/workspace/venvs/stage1/bin/python scripts/run_stage1b_prompt_baseline.py \
  --config configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml \
  --python /workspace/venvs/stage1/bin/python \
  --skip_existing

log "Backing up Stage1b natural 8B-hidden hot artifacts"
backup_run "stage1b_natural_pairs_8b_a100_1x" "stage1b_natural_pairs_8b_a100_1x"

log "Stage1 + Stage1b natural generated/generated pairs with 8B hidden extractor complete"
