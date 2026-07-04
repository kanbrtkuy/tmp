#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
RUN_DIR="${RUN_DIR:-runs/natural_cot_pair_full_n50_v1}"
PILOT_PAIRS="${PILOT_PAIRS:-runs/natural_cot_pair_pilot_v1/natural_safe_pairs.jsonl}"
LOG_DIR="${LOG_DIR:-/workspace/logs/natural_cot_full_n50_v1}"

CONFIG_8B="configs/data/natural_cot_pair_full_n50.yaml"
CONFIG_32B="configs/data/natural_cot_pair_full_n50_32b.yaml"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

run_step() {
  local name="$1"
  shift
  echo "===== START ${name} $(date -Is) ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "===== END ${name} $(date -Is) ====="
}

download_32b() {
  "${VENV_DIR}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
PY
}

run_step prepare "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py --config "${CONFIG_8B}" prepare
cp "${CONFIG_8B}" "${RUN_DIR}/resolved_config_8b_generation.yaml"
cp "${CONFIG_32B}" "${RUN_DIR}/resolved_config_32b_generation.yaml"

run_step make_todo "${VENV_DIR}/bin/python" scripts/data/manage_natural_cot_full_run.py make-todo \
  --run-dir "${RUN_DIR}" \
  --inherited-pairs "${PILOT_PAIRS}" \
  --models r1-8b r1-32b

echo "===== START download_r1_32b $(date -Is) =====" | tee "${LOG_DIR}/download_r1_32b.log"
download_32b >> "${LOG_DIR}/download_r1_32b.log" 2>&1 &
DOWNLOAD_32B_PID=$!
echo "${DOWNLOAD_32B_PID}" > "${LOG_DIR}/download_r1_32b.pid"

run_step generate_r1_8b "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" generate \
  --model r1-8b \
  --prompt-manifest "${RUN_DIR}/prompt_manifest_todo_r1-8b.jsonl"

run_step judge_r1_8b "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" judge \
  --model r1-8b

echo "===== WAIT download_r1_32b $(date -Is) =====" | tee -a "${LOG_DIR}/download_r1_32b.log"
wait "${DOWNLOAD_32B_PID}"
echo "===== END download_r1_32b $(date -Is) =====" | tee -a "${LOG_DIR}/download_r1_32b.log"

run_step generate_r1_32b "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_32B}" generate \
  --model r1-32b \
  --prompt-manifest "${RUN_DIR}/prompt_manifest_todo_r1-32b.jsonl"

run_step judge_r1_32b "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_32B}" judge \
  --model r1-32b

run_step select "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" select

run_step merge "${VENV_DIR}/bin/python" scripts/data/manage_natural_cot_full_run.py merge \
  --run-dir "${RUN_DIR}"

run_step summarize_merged "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" summarize \
  --pairs "${RUN_DIR}/natural_safe_pairs_merged.jsonl"

echo "===== FULL PIPELINE DONE $(date -Is) ====="
