#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR MODEL_PATH [RUN_NAME] [ADD_PAUSE_TOKEN]"
  echo "Example: $0 /workspace/COTPauseToken/data/pause_sft/trusted_cot_18k_intra_cot3/intra_pause_cot3 /workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615 /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B deepseek_intra_pause_cot3_full_sft 1"
  echo "For matched no-pause control, pass ADD_PAUSE_TOKEN=0."
  exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
MODEL_PATH="$3"
RUN_NAME="${4:-deepseek_intra_pause_cot3_full_sft}"
ADD_PAUSE_TOKEN="${5:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
TF32="${TF32:-true}"
TAGS="${TAGS:-[deepseek,intra_pause_cot3,full_sft]}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DEEPSEEK_MODEL_PATH="$MODEL_PATH"
export PAUSE_SFT_DATA_DIR="$DATA_DIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SPECIAL_TOKEN_ARG='rl_algorithm.policy.model.special_tokens_to_add=["<|pause|>"]'
if [ "$ADD_PAUSE_TOKEN" = "0" ]; then
  SPECIAL_TOKEN_ARG='rl_algorithm.policy.model.special_tokens_to_add=[]'
fi

echo "Launching 4-GPU SFT"
echo "  DATA_DIR=$DATA_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  RUN_NAME=$RUN_NAME"
echo "  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "  EFFECTIVE_BATCH_SIZE=$((NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  PYTHON_BIN=$(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN")"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" "$PROJECT_ROOT/src/trl_train.py" \
  experiment=trl_train/deepseek_pause_full_sft \
  "$SPECIAL_TOKEN_ARG" \
  run_name="$RUN_NAME" \
  tags="$TAGS" \
  trainer.max_seq_length="$MAX_SEQ_LENGTH" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  trainer.args.per_device_eval_batch_size="$PER_DEVICE_EVAL_BATCH_SIZE" \
  trainer.args.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  trainer.args.learning_rate="$LEARNING_RATE" \
  trainer.args.num_train_epochs="$NUM_TRAIN_EPOCHS" \
  trainer.args.warmup_ratio="$WARMUP_RATIO" \
  trainer.args.eval_steps="$EVAL_STEPS" \
  trainer.args.save_steps="$SAVE_STEPS" \
  trainer.args.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  trainer.args.dataloader_pin_memory=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.bf16=true \
  trainer.args.fp16=false \
  +trainer.args.tf32="$TF32" \
  hydra.run.dir="$OUTPUT_DIR"

echo "Training finished. Final model should be under: $OUTPUT_DIR/final"
