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
OPTIM="${OPTIM:-adamw_torch}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-false}"
EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-false}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-2}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TF32="${TF32:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
TAGS="${TAGS:-[deepseek,intra_pause_cot3,full_sft]}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORMAT_ONLY="${FORMAT_ONLY:-false}"
FORMAT_ONLY_TRAINABLE_TOKENS="${FORMAT_ONLY_TRAINABLE_TOKENS:-[\"<|pause|>\"]}"
FORMAT_ONLY_INIT_TEXT="${FORMAT_ONLY_INIT_TEXT:-}"
PAUSE_KL_ENABLED="${PAUSE_KL_ENABLED:-false}"
PAUSE_KL_PAUSE_TOKEN="${PAUSE_KL_PAUSE_TOKEN:-<|pause|>}"
PAUSE_KL_PAUSE_TOKENS="${PAUSE_KL_PAUSE_TOKENS:-[\"$PAUSE_KL_PAUSE_TOKEN\"]}"
PAUSE_KL_CONTINUATION_WEIGHT="${PAUSE_KL_CONTINUATION_WEIGHT:-1.0}"
PAUSE_KL_PRE_WEIGHT="${PAUSE_KL_PRE_WEIGHT:-0.1}"
PAUSE_KL_SUPPRESSION_WEIGHT="${PAUSE_KL_SUPPRESSION_WEIGHT:-1.0}"
PAUSE_KL_EMIT_WEIGHT="${PAUSE_KL_EMIT_WEIGHT:-0.3}"
PAUSE_KL_EMIT_MARGIN_WEIGHT="${PAUSE_KL_EMIT_MARGIN_WEIGHT:-0.0}"
PAUSE_KL_STOP_WEIGHT="${PAUSE_KL_STOP_WEIGHT:-0.0}"
PAUSE_KL_N_PAUSE_TOKENS="${PAUSE_KL_N_PAUSE_TOKENS:-3}"
PAUSE_KL_SUPPRESSION_LOSS_TYPE="${PAUSE_KL_SUPPRESSION_LOSS_TYPE:-unlikelihood}"
PAUSE_KL_EMIT_MARGIN="${PAUSE_KL_EMIT_MARGIN:-3.0}"
PAUSE_KL_SUPPRESSION_MARGIN="${PAUSE_KL_SUPPRESSION_MARGIN:-5.0}"
PAUSE_KL_PAUSE_HEAD_ENABLED="${PAUSE_KL_PAUSE_HEAD_ENABLED:-false}"
PAUSE_KL_PAUSE_HEAD_HIDDEN_SIZE="${PAUSE_KL_PAUSE_HEAD_HIDDEN_SIZE:-64}"
PAUSE_KL_PAUSE_HEAD_DROPOUT="${PAUSE_KL_PAUSE_HEAD_DROPOUT:-0.0}"
PAUSE_KL_TEMPERATURE="${PAUSE_KL_TEMPERATURE:-1.0}"
PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE="${PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE:-256}"
PAUSE_KL_SUPPRESSION_CHUNK_SIZE="${PAUSE_KL_SUPPRESSION_CHUNK_SIZE:-1024}"
PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL="${PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL:-true}"
PAUSE_KL_ASSERT_ROWS_ONLY="${PAUSE_KL_ASSERT_ROWS_ONLY:-true}"
PAUSE_KL_POST_STEP_INVARIANT_CHECK="${PAUSE_KL_POST_STEP_INVARIANT_CHECK:-true}"
PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS="${PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS:-50}"
PAUSE_KL_TEACHER_EVAL_MODE="${PAUSE_KL_TEACHER_EVAL_MODE:-true}"
SAVE_BEFORE_TRAIN="${SAVE_BEFORE_TRAIN:-false}"
MAX_STEPS="${MAX_STEPS:-}"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DEEPSEEK_MODEL_PATH="$MODEL_PATH"
export PAUSE_SFT_DATA_DIR="$DATA_DIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SPECIAL_TOKEN_ARG='rl_algorithm.policy.model.special_tokens_to_add=["<|pause|>"]'
if [ "$ADD_PAUSE_TOKEN" = "0" ]; then
  SPECIAL_TOKEN_ARG='rl_algorithm.policy.model.special_tokens_to_add=[]'
elif [ "$FORMAT_ONLY_TRAINABLE_TOKENS" != "[\"<|pause|>\"]" ]; then
  SPECIAL_TOKEN_ARG="rl_algorithm.policy.model.special_tokens_to_add=$FORMAT_ONLY_TRAINABLE_TOKENS"
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
echo "  OPTIM=$OPTIM"
echo "  LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END"
echo "  EARLY_STOPPING=$EARLY_STOPPING_ENABLED patience=$EARLY_STOPPING_PATIENCE threshold=$EARLY_STOPPING_THRESHOLD"
echo "  FORMAT_ONLY=$FORMAT_ONLY trainable_tokens=$FORMAT_ONLY_TRAINABLE_TOKENS"
echo "  PAUSE_KL_ENABLED=$PAUSE_KL_ENABLED pause_tokens=$PAUSE_KL_PAUSE_TOKENS n_pause_tokens=$PAUSE_KL_N_PAUSE_TOKENS continuation=$PAUSE_KL_CONTINUATION_WEIGHT emit=$PAUSE_KL_EMIT_WEIGHT emit_margin_weight=$PAUSE_KL_EMIT_MARGIN_WEIGHT stop=$PAUSE_KL_STOP_WEIGHT suppression=$PAUSE_KL_SUPPRESSION_WEIGHT suppression_type=$PAUSE_KL_SUPPRESSION_LOSS_TYPE pause_head=$PAUSE_KL_PAUSE_HEAD_ENABLED suppression_chunk=$PAUSE_KL_SUPPRESSION_CHUNK_SIZE invariant_check=$PAUSE_KL_POST_STEP_INVARIANT_CHECK invariant_interval=$PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS"
echo "  SAVE_BEFORE_TRAIN=$SAVE_BEFORE_TRAIN"
echo "  WEIGHT_DECAY=$WEIGHT_DECAY"
echo "  GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
if [ -n "$MAX_STEPS" ]; then
  echo "  MAX_STEPS=$MAX_STEPS"
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  echo "  RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
fi
echo "  PYTHON_BIN=$(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN")"

EXTRA_ARGS=()
if [ -n "$MAX_STEPS" ]; then
  EXTRA_ARGS+=(+trainer.args.max_steps="$MAX_STEPS")
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  EXTRA_ARGS+=(+resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi
if [ "$PAUSE_KL_ENABLED" = "true" ]; then
  EXTRA_ARGS+=(
    trainer._target_=src.utils.pause_kl_trainer.PauseKLSFTTrainer
    +trainer.pause_kl.enabled=true
    +trainer.pause_kl.continuation_weight="$PAUSE_KL_CONTINUATION_WEIGHT"
    +trainer.pause_kl.pre_weight="$PAUSE_KL_PRE_WEIGHT"
    +trainer.pause_kl.suppression_weight="$PAUSE_KL_SUPPRESSION_WEIGHT"
    +trainer.pause_kl.emit_weight="$PAUSE_KL_EMIT_WEIGHT"
    +trainer.pause_kl.emit_margin_weight="$PAUSE_KL_EMIT_MARGIN_WEIGHT"
    +trainer.pause_kl.stop_weight="$PAUSE_KL_STOP_WEIGHT"
    +trainer.pause_kl.n_pause_tokens="$PAUSE_KL_N_PAUSE_TOKENS"
    +trainer.pause_kl.suppression_loss_type="$PAUSE_KL_SUPPRESSION_LOSS_TYPE"
    +trainer.pause_kl.emit_margin="$PAUSE_KL_EMIT_MARGIN"
    +trainer.pause_kl.suppression_margin="$PAUSE_KL_SUPPRESSION_MARGIN"
    +trainer.pause_kl.pause_head.enabled="$PAUSE_KL_PAUSE_HEAD_ENABLED"
    +trainer.pause_kl.pause_head.hidden_size="$PAUSE_KL_PAUSE_HEAD_HIDDEN_SIZE"
    +trainer.pause_kl.pause_head.dropout="$PAUSE_KL_PAUSE_HEAD_DROPOUT"
    +trainer.pause_kl.temperature="$PAUSE_KL_TEMPERATURE"
    +trainer.pause_kl.pause_tokens="$PAUSE_KL_PAUSE_TOKENS"
    +trainer.pause_kl.max_kl_tokens_per_example="$PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE"
    +trainer.pause_kl.suppression_chunk_size="$PAUSE_KL_SUPPRESSION_CHUNK_SIZE"
    +trainer.pause_kl.require_pause_before_continuation_kl="$PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL"
    +trainer.pause_kl.assert_rows_only="$PAUSE_KL_ASSERT_ROWS_ONLY"
    +trainer.pause_kl.post_step_invariant_check="$PAUSE_KL_POST_STEP_INVARIANT_CHECK"
    +trainer.pause_kl.invariant_check_interval_steps="$PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS"
    +trainer.pause_kl.teacher_eval_mode="$PAUSE_KL_TEACHER_EVAL_MODE"
  )
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" "$PROJECT_ROOT/src/trl_train.py" \
  experiment=trl_train/deepseek_pause_full_sft \
  "$SPECIAL_TOKEN_ARG" \
  run_name="$RUN_NAME" \
  tags="$TAGS" \
  trainer.max_seq_length="$MAX_SEQ_LENGTH" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  trainer.args.per_device_eval_batch_size="$PER_DEVICE_EVAL_BATCH_SIZE" \
  trainer.args.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  +trainer.args.optim="$OPTIM" \
  trainer.args.learning_rate="$LEARNING_RATE" \
  trainer.args.num_train_epochs="$NUM_TRAIN_EPOCHS" \
  trainer.args.warmup_ratio="$WARMUP_RATIO" \
  trainer.args.weight_decay="$WEIGHT_DECAY" \
  trainer.args.eval_steps="$EVAL_STEPS" \
  trainer.args.save_steps="$SAVE_STEPS" \
  trainer.args.save_total_limit="$SAVE_TOTAL_LIMIT" \
  trainer.args.load_best_model_at_end="$LOAD_BEST_MODEL_AT_END" \
  trainer.args.metric_for_best_model="$METRIC_FOR_BEST_MODEL" \
  trainer.args.greater_is_better="$GREATER_IS_BETTER" \
  +trainer.early_stopping.enabled="$EARLY_STOPPING_ENABLED" \
  +trainer.early_stopping.patience="$EARLY_STOPPING_PATIENCE" \
  +trainer.early_stopping.threshold="$EARLY_STOPPING_THRESHOLD" \
  trainer.args.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  trainer.args.dataloader_pin_memory=true \
  trainer.args.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  trainer.args.bf16=true \
  trainer.args.fp16=false \
  +trainer.args.tf32="$TF32" \
  +trainer.format_only.enabled="$FORMAT_ONLY" \
  +trainer.format_only.trainable_tokens="$FORMAT_ONLY_TRAINABLE_TOKENS" \
  +trainer.format_only.init_from_text="$FORMAT_ONLY_INIT_TEXT" \
  save_before_train="$SAVE_BEFORE_TRAIN" \
  hydra.run.dir="$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "Training finished. Final model should be under: $OUTPUT_DIR/final"
