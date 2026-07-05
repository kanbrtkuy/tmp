#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
CONFIG="${CONFIG:-configs/data/source_expansion_r1_8b_k300.yaml}"
RUN_DIR="${RUN_DIR:-runs/source_expansion_r1_8b_k300_v1}"
LOG_DIR="${LOG_DIR:-/workspace/logs/source_expansion_harmbench_only_r1_8b_k300_v1}"
MODEL="${MODEL:-r1-8b}"
SOURCE_FAMILY="${SOURCE_FAMILY:-harmbench_standard}"
SAMPLES_TOTAL="${SAMPLES_TOTAL:-300}"
SAMPLES_PER_CALL="${SAMPLES_PER_CALL:-5}"
START_SAMPLE="${START_SAMPLE:-110}"
TARGET_SOURCE_PAIRS="${TARGET_SOURCE_PAIRS:-999999}"

export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}"

BASE_PROMPT_MANIFEST="${RUN_DIR}/prompt_manifest.jsonl"
if [[ ! -s "${BASE_PROMPT_MANIFEST}" ]]; then
  echo "ERROR: missing ${BASE_PROMPT_MANIFEST}" >&2
  exit 2
fi

LOCK_FILE="${RUN_DIR}/.harmbench_only_gen_gen_k300.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: another HarmBench-only gen/gen run appears to hold ${LOCK_FILE}" >&2
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

selected_source_pairs() {
  "${VENV_DIR}/bin/python" - "${RUN_DIR}/selection_gen_gen_summary.json" "${SOURCE_FAMILY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = sys.argv[2]
if not path.exists():
    print(0)
    raise SystemExit
summary = json.loads(path.read_text())
print(int((summary.get("selected_pairs_by_source") or {}).get(source, 0)))
PY
}

filter_source_manifest() {
  local input_path="$1"
  local output_path="$2"
  local source_family="$3"
  "${VENV_DIR}/bin/python" - "${input_path}" "${output_path}" "${source_family}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
source_family = sys.argv[3]

rows = []
with input_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if ((row.get("metadata") or {}).get("source_family") or "") == source_family:
            rows.append(row)

with output_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary_path = output_path.with_suffix(".summary.json")
summary = {
    "stage": "filter_source_active_manifest",
    "source_family": source_family,
    "input_manifest": str(input_path),
    "output_manifest": str(output_path),
    "n_source_active_prompts": len(rows),
    "active_prompts_by_source": dict(Counter((row.get("metadata") or {}).get("source_family", "") for row in rows)),
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

for ((sample_start=START_SAMPLE; sample_start<SAMPLES_TOTAL; sample_start+=SAMPLES_PER_CALL)); do
  current_source_pairs="$(selected_source_pairs)"
  if [[ "${current_source_pairs}" -ge "${TARGET_SOURCE_PAIRS}" ]]; then
    echo "===== TARGET STOP: ${SOURCE_FAMILY} selected_pairs=${current_source_pairs} target=${TARGET_SOURCE_PAIRS} before sample_start=${sample_start} ====="
    break
  fi

  sample_end=$((sample_start + SAMPLES_PER_CALL - 1))
  if [[ "${sample_end}" -ge "${SAMPLES_TOTAL}" ]]; then
    sample_end=$((SAMPLES_TOTAL - 1))
  fi
  round_name="$(printf "harmbench_only_r1_8b_round_%03d_%03d" "${sample_start}" "${sample_end}")"
  active_all="${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_${round_name}_all_sources.jsonl"
  active_source="${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_${round_name}.jsonl"
  active_after_all="${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_after_${round_name}_all_sources.jsonl"
  active_after_source="${RUN_DIR}/prompt_manifest_active_gen_gen_r1-8b_after_${round_name}.jsonl"

  run_step "active_${round_name}_before_filter" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" active-gen-gen \
    --model "${MODEL}" \
    --base-prompt-manifest "${BASE_PROMPT_MANIFEST}" \
    --sample-start "${sample_start}" \
    --sample-count "${SAMPLES_PER_CALL}" \
    --output "${active_all}"

  filter_source_manifest "${active_all}" "${active_source}" "${SOURCE_FAMILY}" | tee "${LOG_DIR}/filter_${round_name}.log"

  active_rows="$(count_jsonl_rows "${active_source}")"
  echo "===== ROUND ${round_name}: ${SOURCE_FAMILY}_active_rows=${active_rows} selected_pairs_before=${current_source_pairs} ====="
  if [[ "${active_rows}" -eq 0 ]]; then
    echo "===== NO GENERATE: no active ${SOURCE_FAMILY} prompts need sample_start=${sample_start}; running judge/select in case this range was already generated ====="
    run_step "judge_after_${round_name}_no_new_generation" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
      --config "${CONFIG}" judge \
      --model "${MODEL}"

    run_step "select_after_${round_name}_no_new_generation" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
      --config "${CONFIG}" select-gen-gen \
      --model "${MODEL}"

    run_step "active_${round_name}_after_select_no_new_generation" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
      --config "${CONFIG}" active-gen-gen \
      --model "${MODEL}" \
      --base-prompt-manifest "${BASE_PROMPT_MANIFEST}" \
      --output "${active_after_all}"

    filter_source_manifest "${active_after_all}" "${active_after_source}" "${SOURCE_FAMILY}" | tee "${LOG_DIR}/filter_after_${round_name}_no_new_generation.log"
    remaining_rows="$(count_jsonl_rows "${active_after_source}")"
    echo "===== ROUND ${round_name}: ${SOURCE_FAMILY}_remaining_prompts_without_both_safe_and_unsafe=${remaining_rows} ====="
    if [[ "${remaining_rows}" -eq 0 ]]; then
      echo "===== EARLY STOP: all ${SOURCE_FAMILY} prompts have eligible safe+unsafe candidates after ${round_name} ====="
      break
    fi
    continue
  fi

  run_step "generate_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG}" generate \
    --model "${MODEL}" \
    --prompt-manifest "${active_source}" \
    --sample-start "${sample_start}" \
    --sample-count "${SAMPLES_PER_CALL}"

  run_step "judge_after_${round_name}" "${VENV_DIR}/bin/python" scripts/data/run_natural_cot_pair_pipeline.py \
    --config "${CONFIG}" judge \
    --model "${MODEL}"

  run_step "select_after_${round_name}" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" select-gen-gen \
    --model "${MODEL}"

  run_step "active_${round_name}_after_select" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
    --config "${CONFIG}" active-gen-gen \
    --model "${MODEL}" \
    --base-prompt-manifest "${BASE_PROMPT_MANIFEST}" \
    --output "${active_after_all}"

  filter_source_manifest "${active_after_all}" "${active_after_source}" "${SOURCE_FAMILY}" | tee "${LOG_DIR}/filter_after_${round_name}.log"
  remaining_rows="$(count_jsonl_rows "${active_after_source}")"
  echo "===== ROUND ${round_name}: ${SOURCE_FAMILY}_remaining_prompts_without_both_safe_and_unsafe=${remaining_rows} ====="
  if [[ "${remaining_rows}" -eq 0 ]]; then
    echo "===== EARLY STOP: all ${SOURCE_FAMILY} prompts have eligible safe+unsafe candidates after ${round_name} ====="
    break
  fi
done

run_step "final_select_gen_gen" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" select-gen-gen \
  --model "${MODEL}"

run_step "final_summarize_gen_gen" "${VENV_DIR}/bin/python" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" summarize

echo "===== HARMBENCH-ONLY R1-8B K300 GEN/GEN PIPELINE DONE $(date -Is) ${SOURCE_FAMILY}_selected_pairs=$(selected_source_pairs) ====="
