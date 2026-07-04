#!/usr/bin/env bash
set -euo pipefail

# Post-HarmBench Stage 1 orchestrator.
#
# This script is safe to start while the HB-only generator is still running.  It
# waits for that wrapper to exit, freezes the primary N=100 first-budget pairs,
# runs CPU-only audits/baselines, and only launches GPU Stage1 if the human-QA
# gate has been completed and passed.

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
CONFIG="${CONFIG:-configs/data/source_expansion_r1_8b_k300.yaml}"
MODEL="${MODEL:-r1-8b}"
RUN_DIR="${RUN_DIR:-runs/source_expansion_r1_8b_k300_v1}"
STAGE1_OUT_ROOT="${STAGE1_OUT_ROOT:-runs/stage1_post_hb_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/workspace/logs/stage1_post_hb}"

WAIT_FOR_HB="${WAIT_FOR_HB:-1}"
HB_PROCESS_PATTERN="${HB_PROCESS_PATTERN:-runpod_harmbench_only_gen_gen_k300_r1_8b.sh}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

FIXED_BUDGET_SAMPLE_START="${FIXED_BUDGET_SAMPLE_START:-0}"
FIXED_BUDGET_MAX_SAMPLE_IDX="${FIXED_BUDGET_MAX_SAMPLE_IDX:-100}"
WJB_TRAINVAL_CAP="${WJB_TRAINVAL_CAP:-700}"
QA_ROWS_PER_SOURCE="${QA_ROWS_PER_SOURCE:-60}"
QA_UNSAFE_AGREEMENT_BAR="${QA_UNSAFE_AGREEMENT_BAR:-0.90}"
REQUIRED_LOSO_SOURCES="${REQUIRED_LOSO_SOURCES:-reasoningshield,strongreject_full,wildjailbreak_vanilla_harmful,harmbench_standard}"
MIN_LOSO_SOURCE_PAIRS="${MIN_LOSO_SOURCE_PAIRS:-150}"

# Extra fixed/frozen pair JSONL files to include in the LOSO freeze, separated
# by whitespace.  The source-expansion k300 run currently covers SR/HB/WJB; set
# this to the fixed ReasoningShield gen/gen pair file before claiming 4-source
# LOSO, e.g. EXTRA_LOSO_PAIR_JSONL="/path/to/reasoningshield_pairs.jsonl".
EXTRA_LOSO_PAIR_JSONL="${EXTRA_LOSO_PAIR_JSONL:-}"

# Optional whitespace-separated SOURCE=PATH entries.  These are deliberately
# explicit because safe-prompt diagnostics and external-test quarantine depend
# on project-specific prompt sources.
SAFE_PROMPT_INPUTS="${SAFE_PROMPT_INPUTS:-}"
EXTERNAL_PROMPT_JSONL="${EXTERNAL_PROMPT_JSONL:-}"

RUN_CPU_BASELINES="${RUN_CPU_BASELINES:-1}"
RUN_GPU_STAGE1="${RUN_GPU_STAGE1:-1}"
STAGE1_SEQUENCE_SCRIPT="${STAGE1_SEQUENCE_SCRIPT:-pipelines/run_stage1_sequence.sh}"

export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm-cache}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
mkdir -p "${STAGE1_OUT_ROOT}"

run_step() {
  local name="$1"
  shift
  echo "===== START ${name} $(date -Is) ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "===== END ${name} $(date -Is) ====="
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 2
  fi
}

verify_loso_sources() {
  local required_sources="$1"
  local min_pairs="$2"
  shift 2
  "${PYTHON}" - "${required_sources}" "${min_pairs}" "$@" <<'PY'
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

required = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
min_pairs = int(sys.argv[2])
paths = [Path(item) for item in sys.argv[3:]]

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def canonical(value):
    raw = clean(value).lower()
    if "reasoningshield" in raw:
        return "reasoningshield"
    if "strongreject" in raw:
        return "strongreject_full"
    if "harmbench" in raw:
        return "harmbench_standard"
    if "wildjailbreak" in raw or raw in {"wjb", "wildjailbreak_vanilla"}:
        return "wildjailbreak_vanilla_harmful"
    return raw

def source_family(row):
    metadata = row.get("metadata") or {}
    prompt_metadata = metadata.get("prompt_metadata") or {}
    provenance = metadata.get("source_provenance") or {}
    for value in (
        row.get("source_family"),
        metadata.get("source_family"),
        metadata.get("source_pair_source"),
        prompt_metadata.get("source_family"),
        provenance.get("source_family"),
        row.get("source"),
        metadata.get("source"),
    ):
        source = canonical(value)
        if source:
            return source
    return ""

source_to_pairs = defaultdict(set)
for path in paths:
    if not path.exists():
        raise SystemExit(f"missing LOSO pair input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = source_family(row)
            pair_id = clean(row.get("pair_id"))
            if source and pair_id:
                source_to_pairs[source].add(pair_id)

counts = {source: len(source_to_pairs.get(source, set())) for source in required}
missing = [source for source, count in counts.items() if count < min_pairs]
print(json.dumps({"required_sources": required, "min_pairs": min_pairs, "pair_counts": counts}, indent=2))
if missing:
    raise SystemExit(
        "LOSO source gate failed; provide/fix EXTRA_LOSO_PAIR_JSONL or lower MIN_LOSO_SOURCE_PAIRS only for a declared pilot. "
        + json.dumps({"failing_sources": missing, "pair_counts": counts}, sort_keys=True)
    )
PY
}

check_cpu_deps() {
  "${PYTHON}" - <<'PY'
import importlib
missing = []
for name in ("numpy", "sklearn", "yaml"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("missing CPU dependencies: " + ", ".join(missing))
PY
}

wait_for_hb() {
  if [[ "${WAIT_FOR_HB}" != "1" ]]; then
    return
  fi
  while pgrep -f "${HB_PROCESS_PATTERN}" >/dev/null 2>&1; do
    echo "===== WAIT $(date -Is): HB generation still running (${HB_PROCESS_PATTERN}); sleeping ${WAIT_POLL_SECONDS}s ====="
    sleep "${WAIT_POLL_SECONDS}"
  done
  echo "===== HB generation wrapper no longer running $(date -Is) ====="
}

json_bool_gate() {
  local path="$1"
  local expected_manifest="${2:-}"
  "${PYTHON}" - "$path" "$expected_manifest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
expected_manifest = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
data = json.loads(path.read_text())
if not data.get("passes"):
    raise SystemExit(f"gate failed: {path}")
if expected_manifest:
    if not expected_manifest.exists():
        raise SystemExit(f"gate failed: expected QA manifest is missing: {expected_manifest}")
    h = hashlib.sha256()
    with expected_manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    expected = h.hexdigest()
    observed = data.get("manifest_jsonl_sha256")
    if observed != expected:
        raise SystemExit(
            f"gate failed: QA summary manifest hash mismatch: observed={observed} expected={expected}"
        )
print("gate passed:", path)
PY
}

run_text_bootstrap_ci() {
  local fold_name="$1"
  local text_dir="$2"
  local output_dir="$3"
  local pred_dir="${text_dir}/predictions"
  if [[ ! -d "${pred_dir}" ]]; then
    echo "===== SKIP bootstrap ${fold_name}: no predictions dir ${pred_dir} ====="
    return
  fi

  local args=(scripts/data/run_stage1_bootstrap_ci.py --output-dir "${output_dir}" --n-bootstrap 2000)
  local pred count
  count=0
  for pred in "${pred_dir}"/*.test.predictions.jsonl; do
    [[ -e "${pred}" ]] || continue
    local base name
    base="$(basename "${pred}")"
    name="${base%.test.predictions.jsonl}"
    args+=(--prediction-jsonl "${name}=${pred}")
    count=$((count + 1))
  done
  if [[ "${count}" -eq 0 ]]; then
    echo "===== SKIP bootstrap ${fold_name}: no test prediction files ====="
    return
  fi
  if [[ -s "${pred_dir}/word_tfidf.test.predictions.jsonl" && -s "${pred_dir}/prompt_only_tfidf.test.predictions.jsonl" ]]; then
    args+=(--delta word_tfidf:prompt_only_tfidf)
  fi
  run_step "bootstrap_ci_${fold_name}" "${PYTHON}" "${args[@]}"
}

check_cpu_deps
wait_for_hb

run_step "final_select_gen_gen" "${PYTHON}" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" select-gen-gen \
  --model "${MODEL}"

run_step "final_summarize_gen_gen" "${PYTHON}" scripts/data/manage_source_expansion_gen_gen.py \
  --config "${CONFIG}" summarize

fixed_tag="$(printf "fixed_budget_samples_%03d_%03d" "${FIXED_BUDGET_SAMPLE_START}" "$((FIXED_BUDGET_MAX_SAMPLE_IDX - 1))")"
FIXED_DIR="${STAGE1_OUT_ROOT}/${fixed_tag}"
run_step "select_${fixed_tag}" "${PYTHON}" scripts/data/select_fixed_budget_gen_gen_pairs.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --sample-start "${FIXED_BUDGET_SAMPLE_START}" \
  --max-sample-idx "${FIXED_BUDGET_MAX_SAMPLE_IDX}" \
  --output-dir "${FIXED_DIR}" \
  --write-filtered-judged

PAIR_JSONL="${FIXED_DIR}/natural_generated_pairs.jsonl"
NORMALIZED_JSONL="${FIXED_DIR}/natural_generated_pairs_normalized.jsonl"
require_file "${PAIR_JSONL}"
require_file "${NORMALIZED_JSONL}"

LOSO_PAIR_JSONLS=("${PAIR_JSONL}")
for item in ${EXTRA_LOSO_PAIR_JSONL}; do
  require_file "${item}"
  LOSO_PAIR_JSONLS+=("${item}")
done
verify_loso_sources "${REQUIRED_LOSO_SOURCES}" "${MIN_LOSO_SOURCE_PAIRS}" "${LOSO_PAIR_JSONLS[@]}"

LOSO_INPUT_ARGS=()
for item in "${LOSO_PAIR_JSONLS[@]}"; do
  LOSO_INPUT_ARGS+=(--input-jsonl "${item}")
done

run_step "freeze_audit_${fixed_tag}" "${PYTHON}" scripts/data/audit_stage1_pair_freeze.py \
  "${LOSO_INPUT_ARGS[@]}" \
  --output-dir "${STAGE1_OUT_ROOT}/freeze_audit_${fixed_tag}" \
  --tokenizer-local-files-only \
  --snapshot-inputs

run_step "embedding_dedup_${fixed_tag}" "${PYTHON}" scripts/data/audit_stage1_embedding_dedup.py \
  "${LOSO_INPUT_ARGS[@]}" \
  --output-dir "${STAGE1_OUT_ROOT}/embedding_dedup_${fixed_tag}" \
  --embedding-mode tfidf \
  --allow-tfidf-fallback \
  --threshold 0.90 \
  --near-band-low 0.80 \
  --near-band-high 0.90 \
  --top-k 50

FREEZE_DIR="${STAGE1_OUT_ROOT}/loso_freeze_${fixed_tag}"
run_step "build_loso_freeze_${fixed_tag}" "${PYTHON}" scripts/data/build_stage1_loso_freeze.py \
  "${LOSO_INPUT_ARGS[@]}" \
  --output-dir "${FREEZE_DIR}" \
  --wjb-trainval-cap "${WJB_TRAINVAL_CAP}" \
  --force

QA_DIR="${STAGE1_OUT_ROOT}/human_qa_${fixed_tag}"
run_step "sample_human_qa_${fixed_tag}" "${PYTHON}" scripts/data/sample_stage1_human_qa.py \
  --normalized-jsonl "${FREEZE_DIR}/frozen_normalized_all.jsonl" \
  --output-dir "${QA_DIR}" \
  --rows-per-source "${QA_ROWS_PER_SOURCE}" \
  --include-text

if [[ -n "${SAFE_PROMPT_INPUTS}" ]]; then
  safe_args=(scripts/data/build_stage1_safe_prompt_diagnostics.py --output-dir "${STAGE1_OUT_ROOT}/safe_prompt_diagnostics")
  for item in ${SAFE_PROMPT_INPUTS}; do
    safe_args+=(--input-jsonl "${item}")
  done
  run_step "safe_prompt_diagnostics" "${PYTHON}" "${safe_args[@]}"
else
  echo "===== SKIP safe_prompt_diagnostics: set SAFE_PROMPT_INPUTS='source=path ...' to freeze S->S prompts ====="
fi

if [[ -n "${EXTERNAL_PROMPT_JSONL}" ]]; then
  quarantine_args=(
    scripts/data/quarantine_stage1_external_prompts.py
    --reference-jsonl "${FREEZE_DIR}/frozen_normalized_all.jsonl"
    --output-dir "${STAGE1_OUT_ROOT}/external_quarantine"
  )
  for item in ${EXTERNAL_PROMPT_JSONL}; do
    quarantine_args+=(--external-jsonl "${item}")
  done
  run_step "external_prompt_quarantine" "${PYTHON}" "${quarantine_args[@]}"
else
  echo "===== SKIP external_prompt_quarantine: set EXTERNAL_PROMPT_JSONL='source=path ...' before final external tests ====="
fi

if [[ "${RUN_CPU_BASELINES}" == "1" ]]; then
  for fold_dir in "${FREEZE_DIR}"/folds/*; do
    [[ -d "${fold_dir}" ]] || continue
    fold_name="$(basename "${fold_dir}")"
    text_dir="${STAGE1_OUT_ROOT}/text_baselines/${fold_name}"
    surface_dir="${STAGE1_OUT_ROOT}/surface_audit/${fold_name}"
    run_step "text_baselines_${fold_name}" "${PYTHON}" scripts/data/run_stage1_text_baselines.py \
      --export-dir "${fold_dir}" \
      --output-dir "${text_dir}" \
      --write-predictions \
      --baselines all

    run_step "surface_audit_${fold_name}" "${PYTHON}" scripts/data/run_stage1_surface_audit.py \
      --export-dir "${fold_dir}" \
      --output-dir "${surface_dir}" \
      --bootstrap-pairs \
      --bootstrap-samples 1000

    run_text_bootstrap_ci "${fold_name}" "${text_dir}" "${STAGE1_OUT_ROOT}/bootstrap_ci/${fold_name}"
  done
fi

QA_SUMMARY_JSON="${HUMAN_QA_SUMMARY_JSON:-${QA_DIR}/stage1_human_qa_summary.json}"
if [[ ! -s "${QA_SUMMARY_JSON}" ]]; then
  echo "===== STOP BEFORE GPU STAGE1 ====="
  echo "Human QA sheet was written to: ${QA_DIR}/stage1_human_qa_sheet.tsv"
  echo "After annotation, run scripts/data/summarize_stage1_human_qa.py with --qa-tsv ${QA_DIR}/stage1_human_qa_sheet.tsv and set HUMAN_QA_SUMMARY_JSON to the passing summary."
  exit 20
fi
json_bool_gate "${QA_SUMMARY_JSON}" "${QA_DIR}/stage1_human_qa_manifest.jsonl"

if [[ "${RUN_GPU_STAGE1}" == "1" ]]; then
  run_step "gpu_stage1_sequence" bash "${STAGE1_SEQUENCE_SCRIPT}"
else
  echo "===== RUN_GPU_STAGE1=0; CPU audits/baselines complete, GPU Stage1 not launched ====="
fi

echo "===== STAGE1_POST_HB_ORCHESTRATOR_DONE $(date -Is) output=${STAGE1_OUT_ROOT} ====="
