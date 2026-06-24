#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 MODEL SAMPLE_DIR OUTPUT_DIR GPU_IDS [MAX_MODEL_LEN] [BATCH_SIZE] [GPU_MEMORY_UTILIZATION] [MAX_NUM_SEQS]"
  echo "Example: $0 prometheus-eval/prometheus-7b-v2.0 data/pause_sft/judge_sample_400 data/pause_sft/judge_prometheus_400 0,1,2,3 8192 16 0.92 16"
  exit 1
fi

MODEL="$1"
SAMPLE_DIR="$2"
OUTPUT_DIR="$3"
IFS=',' read -r -a GPU_IDS <<< "$4"
MAX_MODEL_LEN="${5:-8192}"
BATCH_SIZE="${6:-16}"
GPU_MEMORY_UTILIZATION="${7:-0.92}"
MAX_NUM_SEQS="${8:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_DIR"

if [ "${#GPU_IDS[@]}" -ne 4 ]; then
  echo "Expected exactly 4 GPU ids, got ${#GPU_IDS[@]}: $4" >&2
  exit 1
fi

declare -a PIDS=()
for shard_id in 0 1 2 3; do
  gpu="${GPU_IDS[$shard_id]}"
  shard="$(printf "shard_%02d" "$shard_id")"
  input_jsonl="$SAMPLE_DIR/${shard}.jsonl"
  output_jsonl="$OUTPUT_DIR/${shard}.judged.jsonl"
  log_file="$OUTPUT_DIR/${shard}.log"
  echo "Launching $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/judge_prometheus_vllm_shard.py" \
    --model "$MODEL" \
    --input_jsonl "$input_jsonl" \
    --output_jsonl "$output_jsonl" \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size "$BATCH_SIZE" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    > "$log_file" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "At least one judge shard failed. Check logs in $OUTPUT_DIR" >&2
  exit 1
fi

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for shard_path in sorted(out_dir.glob("shard_*.judged.jsonl")):
    with shard_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
if rows and all("index" in row for row in rows):
    rows.sort(key=lambda row: row["index"])

merged = out_dir / "all_judged.jsonl"
with merged.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

scores = Counter(str(row.get("judge", {}).get("score")) for row in rows)
summary = {
    "output_dir": str(out_dir),
    "merged_jsonl": str(merged),
    "rows": len(rows),
    "passed": sum(1 for row in rows if row.get("judge", {}).get("pass")),
    "scores": dict(scores),
}
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "All judge shards finished."
