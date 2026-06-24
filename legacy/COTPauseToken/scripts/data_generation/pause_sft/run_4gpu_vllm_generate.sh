#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 MODEL MODEL_LABEL INPUT_JSON OUTPUT_DIR GPU_IDS [MAX_TOKENS] [BATCH_SIZE] [MAX_MODEL_LEN] [GPU_MEMORY_UTILIZATION] [MAX_NUM_SEQS]"
  echo "Example: $0 /path/to/model pause3 /path/to/test.json /path/to/out 0,1,2,3 1024 32 4096 0.90 32"
  exit 1
fi

MODEL="$1"
MODEL_LABEL="$2"
INPUT_JSON="$3"
OUTPUT_DIR="$4"
IFS=',' read -r -a GPU_IDS <<< "$5"
MAX_TOKENS="${6:-1024}"
BATCH_SIZE="${7:-32}"
MAX_MODEL_LEN="${8:-4096}"
GPU_MEMORY_UTILIZATION="${9:-0.90}"
MAX_NUM_SEQS="${10:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_DIR"

if [ "${#GPU_IDS[@]}" -ne 4 ]; then
  echo "Expected exactly 4 GPU ids, got ${#GPU_IDS[@]}: $5" >&2
  exit 1
fi

declare -a PIDS=()
for shard_id in 0 1 2 3; do
  gpu="${GPU_IDS[$shard_id]}"
  shard="$(printf "shard_%02d" "$shard_id")"
  output_jsonl="$OUTPUT_DIR/${shard}.jsonl"
  log_file="$OUTPUT_DIR/${shard}.log"
  echo "Launching $MODEL_LABEL $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/vllm_generate_sft_shard.py" \
    --model "$MODEL" \
    --model_label "$MODEL_LABEL" \
    --input_json "$INPUT_JSON" \
    --output_jsonl "$output_jsonl" \
    --shard_id "$shard_id" \
    --num_shards 4 \
    --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
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
  echo "At least one generation shard failed. Check logs in $OUTPUT_DIR" >&2
  exit 1
fi

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for shard_path in sorted(out_dir.glob("shard_*.jsonl")):
    with shard_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
rows.sort(key=lambda row: row["index"])
merged = out_dir / "generations.jsonl"
with merged.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
summary = {
    "output_dir": str(out_dir),
    "merged_jsonl": str(merged),
    "rows": len(rows),
    "prefix_counts": dict(Counter(row["prefix_bucket"] for row in rows)),
    "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
    "leading_pause_counts": dict(Counter(str(row["leading_pause_count"]) for row in rows)),
}
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "All generation shards finished."
