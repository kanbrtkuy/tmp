#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 INPUT_JSONL TOKENIZER_PATH OUTPUT_ROOT [TRAIN_SIZE] [VAL_SIZE] [TEST_SIZE]"
  echo "Example: $0 data/pause_sft/trusted_cot_18k/trusted_cot_raw.jsonl /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B data/pause_sft/trusted_cot_18k_intra_cot3 17000 500 500"
  exit 1
fi

INPUT_JSONL="$1"
TOKENIZER_PATH="$2"
OUTPUT_ROOT="$3"
TRAIN_SIZE="${4:-9000}"
VAL_SIZE="${5:-500}"
TEST_SIZE="${6:-500}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/build_intra_think_pause_sft_splits.py" \
  --input_jsonl "$INPUT_JSONL" \
  --output_root "$OUTPUT_ROOT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --cot_offset 3 \
  --n_pause_tokens 3

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/intra_pause_cot3" \
  --mode intra_pause_cot3 \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/intra_pause_cot3_format_validation.json"

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/no_pause_matched" \
  --mode no_pause \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/no_pause_matched_format_validation.json"

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/pre_think_pause3_matched" \
  --mode pre_think_pause \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/pre_think_pause3_matched_format_validation.json"

echo "Intra-think pause cot3 data prep finished: $OUTPUT_ROOT"
