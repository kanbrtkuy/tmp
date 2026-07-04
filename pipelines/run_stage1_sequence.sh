#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
ARCHIVE_ROOT="${COT_SAFETY_STAGE1_ARCHIVE_ROOT:-/workspace/stage1-results}"

# Frozen-fold Stage 1 launcher.  The post-HB orchestrator passes
# STAGE1_FREEZE_DIR after building loso_freeze_*/folds/<source>/normalized.
STAGE1_FREEZE_DIR="${STAGE1_FREEZE_DIR:-${FREEZE_DIR:-}}"
STAGE1_PREPARED_ROOT="${STAGE1_PREPARED_ROOT:-${STAGE1_FREEZE_DIR}/stage1_prepared}"
STAGE1_CONFIG_ROOT="${STAGE1_CONFIG_ROOT:-${STAGE1_FREEZE_DIR}/stage1_gpu_configs}"
STAGE1_FOLDS="${STAGE1_FOLDS:-reasoningshield strongreject_full wildjailbreak_vanilla_harmful harmbench_standard}"

# Whitespace-separated list of template configs.  The defaults target the
# current one-A100 RunPod and use aggressive per-job parallelism in the configs.
CONFIGS="${CONFIGS:-configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml}"

# Keep hidden arrays only if you need to rerun probes without re-extraction.
CLEAN_HIDDEN_AFTER_STAGE="${CLEAN_HIDDEN_AFTER_STAGE:-1}"
STAGE1_SEQUENCE_DRY_RUN="${STAGE1_SEQUENCE_DRY_RUN:-0}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_stage1_env.sh"

cd "${ROOT}"
mkdir -p "${ARCHIVE_ROOT}" "${COT_SAFETY_RUN_ROOT}" "${STAGE1_PREPARED_ROOT}" "${STAGE1_CONFIG_ROOT}" logs

if [[ -z "${STAGE1_FREEZE_DIR}" ]]; then
  echo "STAGE1_FREEZE_DIR is required for frozen-fold Stage 1." >&2
  exit 2
fi
if [[ ! -d "${STAGE1_FREEZE_DIR}/folds" ]]; then
  echo "missing frozen folds: ${STAGE1_FREEZE_DIR}/folds" >&2
  exit 2
fi

config_value() {
  local config="$1"
  local expr="$2"
  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" - "$config" "$expr" <<'PY'
from pathlib import Path
import sys
from cot_safety.config import load_config

cfg = load_config(Path(sys.argv[1]))
value = cfg
for key in sys.argv[2].split("."):
    value = value.get(key, {}) if isinstance(value, dict) else {}
print(value if isinstance(value, str) else "")
PY
}

resolve_env_path() {
  local value="$1"
  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" - "$value" <<'PY'
import os
import re
import sys

value = sys.argv[1]
pattern = re.compile(r"\$\{([^}:]+):-([^}]+)\}")
value = pattern.sub(lambda m: os.environ.get(m.group(1), m.group(2)), value)
print(os.path.expandvars(value))
PY
}

make_fold_config() {
  local template="$1"
  local fold="$2"
  local prepared_dir="$3"
  local output_config="$4"
  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" - "$template" "$fold" "$prepared_dir" "$output_config" <<'PY'
from pathlib import Path
import sys

from cot_safety.config import dump_config, load_config

template = Path(sys.argv[1])
fold = sys.argv[2]
prepared_dir = sys.argv[3]
output_config = Path(sys.argv[4])

cfg = load_config(template)
base_run = str(cfg.get("run", {}).get("name") or template.stem)
run_name = f"{base_run}_loso_{fold}"
cfg.setdefault("run", {})["name"] = run_name
cfg["run"]["output_dir"] = f"${{COT_SAFETY_RUN_ROOT:-runs}}/{run_name}"

data = cfg.setdefault("data", {})
data["prepared_data_dir"] = prepared_dir
data["sources"] = []
data["heldout_sources"] = []

hidden_prefix = str((cfg.get("legacy") or {}).get("hidden_prefix") or run_name)
cfg["legacy"] = {
    "data_dir": prepared_dir,
    "hidden_dir": f"${{COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}}/runs/hidden/{run_name}",
    "hidden_prefix": f"{hidden_prefix}_loso_{fold}",
    "log_dir": f"${{COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}}/runs/logs/{run_name}",
    "single_scan_out_root": f"${{COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}}/runs/{run_name}/linear",
    "multilayer_out_root": f"${{COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}}/runs/{run_name}/multilayer",
}

output_config.parent.mkdir(parents=True, exist_ok=True)
output_config.write_text(dump_config(cfg), encoding="utf-8")
print(run_name)
PY
}

export_fold_for_stage1() {
  local fold="$1"
  local fold_dir="${STAGE1_FREEZE_DIR}/folds/${fold}"
  local prepared_dir="${STAGE1_PREPARED_ROOT}/${fold}"
  if [[ ! -s "${fold_dir}/normalized/train.jsonl" || ! -s "${fold_dir}/normalized/val.jsonl" || ! -s "${fold_dir}/normalized/test.jsonl" ]]; then
    echo "missing normalized split files for fold=${fold}: ${fold_dir}" >&2
    exit 2
  fi
  "${PYTHON}" scripts/data/export_normalized_pairs_for_stage1.py \
    --input-dir "${fold_dir}" \
    --output-dir "${prepared_dir}" \
    --n-pause-tokens 0 \
    --require-pair-integrity
}

run_and_archive() {
  local config="$1"
  local run_name hidden_dir wrapper
  run_name="$(config_value "$config" "run.name")"
  hidden_dir="$(resolve_env_path "$(config_value "$config" "legacy.hidden_dir")")"

  if [[ -z "${run_name}" ]]; then
    echo "Could not resolve run.name for ${config}" >&2
    exit 1
  fi

  if [[ "${config}" == *stage1b* ]]; then
    wrapper="scripts/run_stage1b_prompt_baseline.py"
  else
    wrapper="scripts/run_stage1_positionscan.py"
  fi

  local args=(--config "${config}" --skip_existing)
  if [[ "${config}" == *stage1b* ]]; then
    args+=(--skip_data_prep)
  fi
  if [[ "${STAGE1_SEQUENCE_DRY_RUN}" == "1" ]]; then
    args+=(--dry_run)
  fi

  echo "========== START ${run_name} $(date -Is) =========="
  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" "${wrapper}" "${args[@]}" \
    2>&1 | tee "${ARCHIVE_ROOT}/${run_name}.log"
  echo "========== END ${run_name} $(date -Is) =========="

  mkdir -p "${ARCHIVE_ROOT}/${run_name}"
  rsync -a --ignore-missing-args \
    "${COT_SAFETY_HOT_ROOT}/runs/${run_name}/" \
    "${ARCHIVE_ROOT}/${run_name}/runs/" || true
  rsync -a --ignore-missing-args \
    "${COT_SAFETY_HOT_ROOT}/runs/logs/${run_name}/" \
    "${ARCHIVE_ROOT}/${run_name}/logs/" || true
  rsync -a --ignore-missing-args \
    "${ROOT}/runs/${run_name}_resolved.yaml" \
    "${ARCHIVE_ROOT}/${run_name}/" || true
  rsync -a --ignore-missing-args \
    "${config}" \
    "${ARCHIVE_ROOT}/${run_name}/" || true

  if [[ "${STAGE1_SEQUENCE_DRY_RUN}" != "1" && "${CLEAN_HIDDEN_AFTER_STAGE}" == "1" && -n "${hidden_dir}" ]]; then
    echo "Cleaning hidden arrays for ${run_name}: ${hidden_dir}"
    rm -rf "${hidden_dir}"
  fi

  df -h "${COT_SAFETY_HOT_ROOT}" "${ARCHIVE_ROOT}" 2>/dev/null || true
}

for fold in ${STAGE1_FOLDS}; do
  echo "========== PREPARE FOLD ${fold} $(date -Is) =========="
  export_fold_for_stage1 "${fold}"
  prepared_dir="${STAGE1_PREPARED_ROOT}/${fold}"
  for template in ${CONFIGS}; do
    if [[ ! -s "${template}" ]]; then
      echo "missing Stage1 template config: ${template}" >&2
      exit 2
    fi
    generated="${STAGE1_CONFIG_ROOT}/${fold}_$(basename "${template}")"
    run_name="$(make_fold_config "${template}" "${fold}" "${prepared_dir}" "${generated}")"
    echo "generated ${generated} for ${run_name}"
    run_and_archive "${generated}"
  done
done

rsync -a --ignore-missing-args "${COT_SAFETY_HOT_ROOT}/runs/logs/" "${ARCHIVE_ROOT}/all_logs/" || true
echo "ALL_STAGE1_SEQUENCE_DONE $(date -Is) freeze=${STAGE1_FREEZE_DIR}"
