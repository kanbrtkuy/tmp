#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
RUN_DIR="${RUN_DIR:-runs/natural_cot_pair_full_n100_8b_remaining_v1}"
LOG_DIR="${LOG_DIR:-/workspace/logs/natural_cot_full_n100_8b_remaining_v1}"

CONFIG_8B="configs/data/natural_cot_pair_full_n100_8b_remaining.yaml"
MODEL="r1-8b"
SAMPLES_PER_CALL="${SAMPLES_PER_CALL:-5}"
SAMPLES_TOTAL="${SAMPLES_TOTAL:-100}"
SAMPLE_START_INITIAL="${SAMPLE_START_INITIAL:-50}"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: missing Python environment: ${VENV_DIR}" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/prompt_manifest.jsonl" ]]; then
  echo "ERROR: missing remaining-prompt manifest: ${RUN_DIR}/prompt_manifest.jsonl" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/inherited_natural_safe_pairs.jsonl" ]]; then
  echo "ERROR: missing inherited selected safe pairs: ${RUN_DIR}/inherited_natural_safe_pairs.jsonl" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/unsafe_reference_manifest.jsonl" ]]; then
  echo "ERROR: missing unsafe reference manifest: ${RUN_DIR}/unsafe_reference_manifest.jsonl" >&2
  exit 2
fi

config_samples_total=$("${VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import importlib.util
script = Path("scripts/data/run_natural_cot_pair_pipeline.py").resolve()
spec = importlib.util.spec_from_file_location("natural_pipe", script)
pipe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipe)
cfg = pipe.read_config("configs/data/natural_cot_pair_full_n100_8b_remaining.yaml")
print(int(cfg["generation"]["samples_per_prompt"]))
PY
)
if [[ "${config_samples_total}" != "${SAMPLES_TOTAL}" ]]; then
  echo "ERROR: CONFIG_8B samples_per_prompt=${config_samples_total}, but SAMPLES_TOTAL=${SAMPLES_TOTAL}" >&2
  exit 2
fi
if (( SAMPLE_START_INITIAL < 0 || SAMPLE_START_INITIAL >= SAMPLES_TOTAL )); then
  echo "ERROR: invalid SAMPLE_START_INITIAL=${SAMPLE_START_INITIAL}" >&2
  exit 2
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

for ((sample_start=SAMPLE_START_INITIAL; sample_start<SAMPLES_TOTAL; sample_start+=SAMPLES_PER_CALL)); do
  round_name=$(printf "r1_8b_round_%03d_%03d" "${sample_start}" "$((sample_start + SAMPLES_PER_CALL - 1))")
  active_manifest="${RUN_DIR}/prompt_manifest_active_r1-8b_${round_name}.jsonl"

  run_step "active_${round_name}_before_generate" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG_8B}" active-prompts \
    --model "${MODEL}" \
    --base-prompt-manifest "${RUN_DIR}/prompt_manifest.jsonl" \
    --sample-start "${sample_start}" \
    --sample-count "${SAMPLES_PER_CALL}" \
    --output "${active_manifest}"

  active_rows=$(count_jsonl_rows "${active_manifest}")
  echo "===== ROUND ${round_name}: active_rows=${active_rows} ====="
  if [[ "${active_rows}" -gt 0 ]]; then
    run_step "generate_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
      --config "${CONFIG_8B}" generate \
      --model "${MODEL}" \
      --prompt-manifest "${active_manifest}" \
      --sample-start "${sample_start}" \
      --sample-count "${SAMPLES_PER_CALL}"
  else
    echo "===== SKIP generate_${round_name}: no active prompts needing this sample range ====="
  fi

  run_step "judge_after_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG_8B}" judge \
    --model "${MODEL}"

  run_step "active_${round_name}_after_judge" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG_8B}" active-prompts \
    --model "${MODEL}" \
    --base-prompt-manifest "${RUN_DIR}/prompt_manifest.jsonl" \
    --output "${RUN_DIR}/prompt_manifest_active_r1-8b_after_${round_name}.jsonl"

  remaining_rows=$(count_jsonl_rows "${RUN_DIR}/prompt_manifest_active_r1-8b_after_${round_name}.jsonl")
  echo "===== ROUND ${round_name}: remaining_prompts_without_safe=${remaining_rows} ====="
  if [[ "${remaining_rows}" -eq 0 ]]; then
    echo "===== EARLY STOP: all remaining prompts have an eligible safe candidate after ${round_name} ====="
    break
  fi
done

run_step select "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" select \
  --unsafe-reference-manifest "${RUN_DIR}/unsafe_reference_manifest.jsonl"

run_step merge "${VENV_DIR}/bin/python" scripts/data/manage_natural_cot_full_run.py merge \
  --run-dir "${RUN_DIR}"

run_step summarize_merged "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
  --config "${CONFIG_8B}" summarize \
  --pairs "${RUN_DIR}/natural_safe_pairs_merged.jsonl"

echo "===== ADAPTIVE 8B REMAINING PIPELINE DONE $(date -Is) ====="
