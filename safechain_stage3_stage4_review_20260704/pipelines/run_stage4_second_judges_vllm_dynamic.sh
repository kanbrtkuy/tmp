#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT="${ROOT:-${REPO_ROOT}/legacy/PauseProbe}"
CONFIG="${CONFIG:-}"
SFT_ROOT="${SFT_ROOT:-}"
BASE_ROOT="${BASE_ROOT:-}"
EVAL_ROOTS="${EVAL_ROOTS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${COT_SAFETY_RUN_ROOT:-${REPO_ROOT}/runs}/logs/stage4_second_judges_vllm_dynamic_${RUN_ID}}"
QUEUE_ROOT="${LOG_ROOT}/queue"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
JUDGES_CSV="${JUDGES_CSV:-llamaguard,harmbench}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${ROOT}/scripts/judge/run_vllm_dynamic_open_judges.py}"
LLAMAGUARD_MODEL="${LLAMAGUARD_MODEL:-${COT_SAFETY_JUDGE_ROOT:-/workspace/models/judges}/Llama-Guard-3-8B}"
HARMBENCH_MODEL="${HARMBENCH_MODEL:-${COT_SAFETY_JUDGE_ROOT:-/workspace/models/judges}/HarmBench-Llama-2-13b-cls}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
HF_HOME="${HF_HOME:-${COT_SAFETY_HF_HOME:-${HOME}/.cache/huggingface}}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

HF_ENV_FILE="${COT_SAFETY_HF_ENV_FILE:-/workspace/secrets/hf.env}"
if [[ -f "${HF_ENV_FILE}" ]]; then
  # shellcheck disable=SC1091
  source "${HF_ENV_FILE}"
fi

if [[ -z "${EVAL_ROOTS}" && -n "${CONFIG}" ]]; then
  EVAL_ROOTS="$(
    PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 - "${CONFIG}" "${REPO_ROOT}" <<'PY'
import sys
from pathlib import Path

from cot_safety.config import load_config

repo_root = Path(sys.argv[2]).resolve()
config_path = Path(sys.argv[1])
if not config_path.is_absolute():
    config_path = repo_root / config_path
config = load_config(config_path)
eval_cfg = config.get("eval", {})
roots = []
for item in eval_cfg.get("second_judge_roots", []):
    if not str(item):
        continue
    root = Path(str(item))
    roots.append(str(root if root.is_absolute() else repo_root / root))
if not roots:
    out_root = config.get("run", {}).get("output_dir")
    if out_root:
        out_path = Path(str(out_root))
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        roots.append(str(out_path / str(eval_cfg.get("second_judge_target", "all3"))))
print(",".join(roots))
PY
  )"
fi

if [[ -z "${EVAL_ROOTS}" && -n "${SFT_ROOT}" && -n "${BASE_ROOT}" ]]; then
  EVAL_ROOTS="${SFT_ROOT},${BASE_ROOT}"
fi

if [[ -z "${EVAL_ROOTS}" ]]; then
  echo "Set CONFIG, EVAL_ROOTS, or both SFT_ROOT and BASE_ROOT." >&2
  exit 2
fi

IFS=',' read -r -a JUDGES <<< "${JUDGES_CSV}"
if [[ "${#JUDGES[@]}" -lt 1 ]]; then
  echo "JUDGES_CSV must contain at least one judge." >&2
  exit 2
fi
for judge in "${JUDGES[@]}"; do
  if [[ "${judge}" != "llamaguard" && "${judge}" != "harmbench" ]]; then
    echo "Unsupported vLLM judge: ${judge}; expected llamaguard or harmbench." >&2
    exit 2
  fi
  mkdir -p "${QUEUE_ROOT}/pending/${judge}"
done
mkdir -p \
  "${QUEUE_ROOT}/running" "${QUEUE_ROOT}/done" "${QUEUE_ROOT}/failed" "${LOG_ROOT}"

MODEL_MAP_JSON="{\"llamaguard\":\"${LLAMAGUARD_MODEL}\",\"harmbench\":\"${HARMBENCH_MODEL}\"}"
MAX_MODEL_LEN_JSON='{"llamaguard":4096,"harmbench":2048}'

python3 - "${EVAL_ROOTS}" "${JUDGES_CSV}" "${QUEUE_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

roots = [Path(item) for item in sys.argv[1].split(",") if item]
judges = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
queue_root = Path(sys.argv[3])
tasks = []

prompt_fields = ("input", "prompt", "question", "query", "behavior", "goal")
response_fields = ("generated_for_judge", "generated", "response", "output", "completion", "final_answer")

def clean(value):
    return "" if value is None else str(value).strip()

def count_judgeable(path):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = next((clean(row.get(k)) for k in prompt_fields if clean(row.get(k))), "")
            response = next((clean(row.get(k)) for k in response_fields if clean(row.get(k))), "")
            if prompt and response:
                count += 1
    return count

def count_jsonl(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())

def complete(gen, norm):
    expected = count_judgeable(gen)
    return expected > 0 and count_jsonl(norm) == expected

for gen in sorted(gen for root in roots for gen in root.glob("*/*/*/generations.jsonl")):
    for judge in judges:
        raw = gen.parent / f"{judge}_raw.jsonl"
        norm = gen.parent / f"{judge}_normalized.jsonl"
        if complete(gen, norm):
            continue
        key = hashlib.sha1(f"{judge}\t{gen}".encode("utf-8")).hexdigest()
        tasks.append((judge, key, {"judge": judge, "gen": str(gen), "raw": str(raw), "norm": str(norm)}))

for judge, key, task in tasks:
    path = queue_root / "pending" / judge / f"{key}.json"
    if not path.exists():
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"tasks": len(tasks), "queue_root": str(queue_root), "roots": [str(root) for root in roots], "judges": judges}, indent=2))
PY

launch_worker() {
  local gpu="$1"
  local worker_id="gpu${gpu}"
  local order="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${WORKER_SCRIPT}" \
    --queue_root "${QUEUE_ROOT}" \
    --worker_id "${worker_id}" \
    --preferred_judges "${order}" \
    --model_map_json "${MODEL_MAP_JSON}" \
    --max_model_len_json "${MAX_MODEL_LEN_JSON}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    > "${LOG_ROOT}/${worker_id}.log" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${worker_id}.pid"
}

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  rotated=("${JUDGES[@]}")
  for _ in $(seq 1 $((idx % ${#JUDGES[@]}))); do
    first="${rotated[0]}"
    rotated=("${rotated[@]:1}" "${first}")
  done
  order="$(IFS=','; echo "${rotated[*]}")"
  launch_worker "${gpu}" "${order}"
done

wait

failed_count="$(find "${QUEUE_ROOT}/failed" -type f -name '*.json' | wc -l | tr -d ' ')"
if [[ "${failed_count}" != "0" ]]; then
  echo "[vllm dynamic failed] failed=${failed_count}" >&2
  exit 2
fi

summarize_one() {
  local out_root="$1"
  local judge="$2"
  local prefix="$3"
  (
    cd "${ROOT}"
    python scripts/steering/summarize_intra_pause_full_steering_eval.py \
      --out_root "${out_root}" \
      --normalized_filename "${judge}_normalized.jsonl" \
      --joined_csv "${out_root}/${prefix}_${judge}_joined_rows.csv" \
      --summary_csv "${out_root}/${prefix}_${judge}_summary.csv" \
      --manifest_json "${out_root}/${prefix}_${judge}_summary_manifest.json"
  )
}

summary_prefix() {
  local out_root="$1"
  local base
  base="$(basename "${out_root}")"
  if [[ "${base}" == "all3" ]]; then
    local parent
    parent="$(basename "$(dirname "${out_root}")")"
    echo "${parent}_${base}"
  else
    echo "${base}"
  fi
}

IFS=',' read -r -a EVAL_ROOT_ARRAY <<< "${EVAL_ROOTS}"
for judge in "${JUDGES[@]}"; do
  for eval_root in "${EVAL_ROOT_ARRAY[@]}"; do
    [[ -n "${eval_root}" ]] || continue
    summarize_one "${eval_root}" "${judge}" "$(summary_prefix "${eval_root}")"
  done
done
echo "[vllm dynamic second judges done] ${LOG_ROOT}"
