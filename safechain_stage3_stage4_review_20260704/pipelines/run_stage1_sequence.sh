#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
ARCHIVE_ROOT="${COT_SAFETY_STAGE1_ARCHIVE_ROOT:-/workspace/stage1-results}"

# Whitespace-separated list of configs. Override this for a different model/order.
CONFIGS="${CONFIGS:-configs/experiment/stage1_positionscan_8b_2xa6000.yaml configs/experiment/stage1b_prompt_baseline_8b_2xa6000.yaml}"

# Keep hidden arrays only if you need to rerun probes without re-extraction.
CLEAN_HIDDEN_AFTER_STAGE="${CLEAN_HIDDEN_AFTER_STAGE:-1}"

# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_stage1_env.sh"

cd "${ROOT}"
mkdir -p "${ARCHIVE_ROOT}" "${COT_SAFETY_RUN_ROOT}" logs

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

  if [[ "${CLEAN_HIDDEN_AFTER_STAGE}" == "1" && -n "${hidden_dir}" ]]; then
    echo "Cleaning hidden arrays for ${run_name}: ${hidden_dir}"
    rm -rf "${hidden_dir}"
  fi

  df -h "${COT_SAFETY_HOT_ROOT}" "${ARCHIVE_ROOT}" 2>/dev/null || true
}

for config in ${CONFIGS}; do
  run_and_archive "${config}"
done

rsync -a --ignore-missing-args "${COT_SAFETY_HOT_ROOT}/runs/logs/" "${ARCHIVE_ROOT}/all_logs/" || true
echo "ALL_STAGE1_SEQUENCE_DONE $(date -Is)"
