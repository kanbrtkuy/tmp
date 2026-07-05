#!/usr/bin/env bash
set -euo pipefail

cd /workspace/cot-safety
source /workspace/venvs/stage1/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

BASE="${BASE:-runs/stage1_loso_surface/natural_8b_generated_generated_joined_260703}"
OUT="${OUT:-runs/stage1_loso_surface/natural_8b_generated_generated_surface_parallel_260703}"
LOG="${LOG:-/workspace/logs/stage1_surface_parallel_260703}"

mkdir -p "$OUT" "$LOG"

common=(
  --export-dir "$BASE"
  --max-features-word 300000
  --max-features-char 500000
  --bootstrap-pairs
  --bootstrap-samples 1000
  --cross-source-baselines word_tfidf,word_bow,char_tfidf
)

run_bg() {
  local name="$1"
  shift
  mkdir -p "$OUT/$name"
  echo "[$(date -Iseconds)] launch $name" | tee -a "$LOG/launcher.log"
  python scripts/data/run_stage1_surface_task.py "$@" --output-dir "$OUT/$name" > "$LOG/$name.log" 2>&1 &
}

run_bg feature --task feature "${common[@]}"
run_bg length --task length "${common[@]}"
run_bg trunc_words --task truncation "${common[@]}" --truncation-ks 4,8,16,32,64,128,256,full
run_bg token_16 --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks 16
run_bg token_32 --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks 32
run_bg token_64 --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks 64
run_bg token_128 --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks 128
run_bg token_256 --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks 256
run_bg token_full --task token "${common[@]}" --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B --token-truncation-ks full
run_bg embedding --task embedding "${common[@]}" --embedding-model sentence-transformers/all-MiniLM-L6-v2 --embedding-device cuda --embedding-batch-size 8192
run_bg cross_source --task cross_source "${common[@]}"

wait
status=$?
echo "[$(date -Iseconds)] all tasks finished status=$status" | tee -a "$LOG/launcher.log"

python - <<'PY'
from pathlib import Path
import json

out = Path("runs/stage1_loso_surface/natural_8b_generated_generated_surface_parallel_260703")
summary = {}
for p in sorted(out.glob("*/task_metrics.json")):
    obj = json.load(p.open())
    summary[p.parent.name] = {
        "task": obj["config"]["task"],
        "result_keys": sorted((obj.get("result") or {}).keys()),
    }
(out / "parallel_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
