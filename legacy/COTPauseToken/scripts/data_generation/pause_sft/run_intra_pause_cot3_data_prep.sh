#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 INPUT_JSONL TOKENIZER_PATH OUTPUT_ROOT [TRAIN_SIZE] [VAL_SIZE] [TEST_SIZE]"
  echo "Example: COT_OFFSET=4 INTRA_DIR_NAME=intra_pause_cot4 $0 data/pause_sft/trusted_cot_18k/trusted_cot_raw.jsonl /workspace/models/DeepSeek-R1-Distill-Llama-8B data/pause_sft/trusted_cot_18k_intra_cot4 17000 500 500"
  exit 1
fi

INPUT_JSONL="$1"
TOKENIZER_PATH="$2"
OUTPUT_ROOT="$3"
TRAIN_SIZE="${4:-9000}"
VAL_SIZE="${5:-500}"
TEST_SIZE="${6:-500}"
COT_OFFSET="${COT_OFFSET:-3}"
N_PAUSE_TOKENS="${N_PAUSE_TOKENS:-3}"
PAUSE_TOKEN="${PAUSE_TOKEN:-<|pause|>}"
INTRA_DIR_NAME="${INTRA_DIR_NAME:-intra_pause_cot${COT_OFFSET}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/build_intra_think_pause_sft_splits.py" \
  --input_jsonl "$INPUT_JSONL" \
  --output_root "$OUTPUT_ROOT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --cot_offset "$COT_OFFSET" \
  --n_pause_tokens "$N_PAUSE_TOKENS" \
  --pause_token "$PAUSE_TOKEN" \
  --intra_dir_name "$INTRA_DIR_NAME"

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/$INTRA_DIR_NAME" \
  --mode "$INTRA_DIR_NAME" \
  --cot_offset "$COT_OFFSET" \
  --expected_pause_tokens "$N_PAUSE_TOKENS" \
  --pause_token "$PAUSE_TOKEN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/${INTRA_DIR_NAME}_format_validation.json"

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/no_pause_matched" \
  --mode no_pause \
  --pause_token "$PAUSE_TOKEN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/no_pause_matched_format_validation.json"

python "$SCRIPT_DIR/validate_intra_think_pause_sft_format.py" \
  --dataset_dir "$OUTPUT_ROOT/pre_think_pause3_matched" \
  --mode pre_think_pause \
  --expected_pause_tokens "$N_PAUSE_TOKENS" \
  --pause_token "$PAUSE_TOKEN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_json "$OUTPUT_ROOT/pre_think_pause3_matched_format_validation.json"

echo "Intra-think pause cot${COT_OFFSET} data prep finished: $OUTPUT_ROOT/$INTRA_DIR_NAME"
