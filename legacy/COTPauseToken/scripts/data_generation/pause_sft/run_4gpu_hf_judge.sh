#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 JUDGE_MODEL INPUT_JSONL OUTPUT_DIR GPU_IDS [MODEL_LABEL] [MAX_NEW_TOKENS] [BATCH_SIZE]"
  echo "Example: $0 prometheus-eval/prometheus-7b-v2.0 outputs/base/generations.jsonl outputs/base_judge 0,1,2,3 base 768 4"
  exit 1
fi

JUDGE_MODEL="$1"
INPUT_JSONL="$2"
OUTPUT_DIR="$3"
IFS=',' read -r -a GPU_IDS <<< "$4"
MODEL_LABEL="${5:-}"
MAX_NEW_TOKENS="${6:-768}"
BATCH_SIZE="${7:-4}"

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
  output_jsonl="$OUTPUT_DIR/${shard}.judged.jsonl"
  log_file="$OUTPUT_DIR/${shard}.log"
  echo "Launching judge $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/judge_prometheus_hf_shard.py" \
    --model "$JUDGE_MODEL" \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$output_jsonl" \
    --model_label "$MODEL_LABEL" \
    --shard_id "$shard_id" \
    --num_shards 4 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
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
    "parse_errors": sum(bool(row.get("judge", {}).get("parse_error")) for row in rows),
}
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "All HF judge shards finished."
