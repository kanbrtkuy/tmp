#!/usr/bin/env bash
set -euo pipefail

cd /workspace/cot-safety
source /workspace/venvs/stage1/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-24}"

BASE="${BASE:-runs/stage1_loso_surface/natural_8b_generated_generated_joined_260703}"
OUT="${OUT:-runs/stage1_loso_surface/natural_8b_generated_generated_surface_parallel_260703}"
LOG="${LOG:-/workspace/logs/stage1_trunc_split_260703}"

mkdir -p "$OUT" "$LOG"

common=(
  --task truncation
  --export-dir "$BASE"
  --max-features-word 300000
  --max-features-char 500000
  --bootstrap-pairs
  --bootstrap-samples 1000
  --cross-source-baselines word_tfidf,word_bow,char_tfidf
)

for k in 4 8 16 32 64 128 256 full; do
  mkdir -p "$OUT/trunc_$k"
  python scripts/data/run_stage1_surface_task.py "${common[@]}" --truncation-ks "$k" --output-dir "$OUT/trunc_$k" > "$LOG/trunc_$k.log" 2>&1 &
done

wait

python - <<'PY'
from pathlib import Path
import json

out = Path("runs/stage1_loso_surface/natural_8b_generated_generated_surface_parallel_260703")
summary = {}
for p in sorted(out.glob("trunc_*/task_metrics.json")):
    obj = json.load(p.open())
    summary[p.parent.name] = obj["config"]["truncation_ks"]
(out / "trunc_split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
