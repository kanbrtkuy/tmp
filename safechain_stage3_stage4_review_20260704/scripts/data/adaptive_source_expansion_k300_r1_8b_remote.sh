#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
CONFIG="${CONFIG:-configs/data/source_expansion_r1_8b_k300.yaml}"
RUN_DIR="${RUN_DIR:-runs/source_expansion_r1_8b_k300_v1}"
LOG_DIR="${LOG_DIR:-/workspace/logs/source_expansion_r1_8b_k300_v1}"
MODEL="${MODEL:-r1-8b}"
SAMPLES_TOTAL="${SAMPLES_TOTAL:-300}"
SAMPLES_PER_CALL="${SAMPLES_PER_CALL:-5}"
WAIT_FOR_32B="${WAIT_FOR_32B:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export CONFIG

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}"
LOCK_FILE="${LOCK_FILE:-${RUN_DIR}/.source_expansion_k300.lock}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: another source-expansion run appears to hold ${LOCK_FILE}" >&2
  exit 3
fi

run_step() {
  local name="$1"
  shift
  echo "===== START ${name} $(date -Is) ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "===== END ${name} $(date -Is) ====="
}

count_jsonl_rows() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo 0
  else
    wc -l < "${path}" | tr -d ' '
  fi
}

if [[ "${WAIT_FOR_32B}" == "1" ]]; then
  while pgrep -f "natural_cot_pair_full_n50_32b|adaptive_natural_cot_full_n50_32b_remote.sh" >/dev/null; do
    echo "===== waiting for existing 32B natural generation job $(date -Is) ====="
    sleep "${WAIT_SECONDS}"
  done
fi

config_samples_total=$("${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path
import importlib.util
script = Path("scripts/data/manage_source_expansion_gen_gen.py").resolve()
spec = importlib.util.spec_from_file_location("source_expansion", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
cfg = module.read_config(os.environ["CONFIG"])
print(int(cfg["generation"]["samples_per_prompt"]))
PY
)
if [[ "${config_samples_total}" != "${SAMPLES_TOTAL}" ]]; then
  echo "ERROR: config samples_per_prompt=${config_samples_total}, but SAMPLES_TOTAL=${SAMPLES_TOTAL}" >&2
  exit 2
fi

run_step "prepare_sources" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" prepare \
  --model "${MODEL}"

for ((sample_start=0; sample_start<SAMPLES_TOTAL; sample_start+=SAMPLES_PER_CALL)); do
  sample_end=$((sample_start + SAMPLES_PER_CALL - 1))
  if [[ "${sample_end}" -ge "${SAMPLES_TOTAL}" ]]; then
    sample_end=$((SAMPLES_TOTAL - 1))
  fi
  round_name=$(printf "r1_8b_source_expansion_round_%03d_%03d" "${sample_start}" "${sample_end}")
  active_manifest="${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_${round_name}.jsonl"

  run_step "active_${round_name}_before_generate" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" active-gen-gen \
    --model "${MODEL}" \
    --base-prompt-manifest "${RUN_DIR}/prompt_manifest.jsonl" \
    --sample-start "${sample_start}" \
    --sample-count "${SAMPLES_PER_CALL}" \
    --output "${active_manifest}"

  active_rows=$(count_jsonl_rows "${active_manifest}")
  echo "===== ROUND ${round_name}: active_rows=${active_rows} ====="
  if [[ "${active_rows}" -gt 0 ]]; then
    run_step "generate_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
      --config "${CONFIG}" generate \
      --model "${MODEL}" \
      --prompt-manifest "${active_manifest}" \
      --sample-start "${sample_start}" \
      --sample-count "${SAMPLES_PER_CALL}"
  else
    echo "===== SKIP generate_${round_name}: no active prompts need this sample range ====="
  fi

  run_step "judge_after_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG}" judge \
    --model "${MODEL}"

  run_step "select_after_${round_name}" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" select-gen-gen \
    --model "${MODEL}"

  run_step "active_${round_name}_after_judge" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" active-gen-gen \
    --model "${MODEL}" \
    --base-prompt-manifest "${RUN_DIR}/prompt_manifest.jsonl" \
    --output "${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_after_${round_name}.jsonl"

  remaining_rows=$(count_jsonl_rows "${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_after_${round_name}.jsonl")
  echo "===== ROUND ${round_name}: remaining_prompts_without_both_safe_and_unsafe=${remaining_rows} ====="
  if [[ "${remaining_rows}" -eq 0 ]]; then
    echo "===== EARLY STOP: all prompts have eligible safe+unsafe candidates after ${round_name} ====="
    break
  fi
done

run_step "final_select_gen_gen" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" select-gen-gen \
  --model "${MODEL}"

run_step "final_summarize_gen_gen" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" summarize

echo "===== SOURCE EXPANSION R1-8B K300 PIPELINE DONE $(date -Is) ====="
