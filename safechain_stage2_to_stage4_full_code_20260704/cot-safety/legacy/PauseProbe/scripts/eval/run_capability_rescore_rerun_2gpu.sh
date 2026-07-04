#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/PauseProbe}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/capability_safety_eval_260622}"
OUT_ROOT="${OUT_ROOT:-/workspace/PauseProbe/runs/eval/capability_rescore_rerun_260622}"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
SFT_MODEL="${SFT_MODEL:-/workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final}"
DELTA="${DELTA:-/workspace/PauseProbe/runs/steering/intra_pause_learned_delta_260618/zero_l14_steps80/learned_delta.pt}"
STEER_ALPHA="${STEER_ALPHA:-2}"
STEER_LAYER="${STEER_LAYER:-14}"
DEVICES="${DEVICES:-0,1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-48}"
CAP_MAX_NEW_TOKENS="${CAP_MAX_NEW_TOKENS:-2048}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
PYTHON="${PYTHON:-/workspace/venvs/probe/bin/python}"

mkdir -p "$OUT_ROOT"/{generations,logs,queue/done,queue/locks,queue/failed}
cd "$ROOT"
source /workspace/secrets/hf.env

CAP_IN="$DATA_ROOT/capability_prompts.jsonl"
QUEUE="$OUT_ROOT/queue/capability_generation_jobs.tsv"

cat > "$QUEUE" <<EOF
capability	base	base	0	$BASE_MODEL	$CAP_IN	$OUT_ROOT/generations/base_capability.jsonl	$CAP_MAX_NEW_TOKENS
capability	sft_alpha0	sft	0	$SFT_MODEL	$CAP_IN	$OUT_ROOT/generations/sft_alpha0_capability.jsonl	$CAP_MAX_NEW_TOKENS
capability	sft_steer_alpha${STEER_ALPHA}	steer	$STEER_ALPHA	$SFT_MODEL	$CAP_IN	$OUT_ROOT/generations/sft_steer_alpha${STEER_ALPHA}_capability.jsonl	$CAP_MAX_NEW_TOKENS
EOF

expected_rows() {
  wc -l < "$1" | tr -d " "
}

complete_jsonl() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || return 1
  local got
  got=$(wc -l < "$path" | tr -d " ")
  [[ "$got" == "$expected" && "$got" -gt 0 ]]
}

run_generation_worker() {
  local gpu="$1"
  local log="$OUT_ROOT/logs/generation_gpu${gpu}.log"
  echo "[$(date -Is)] capability worker gpu=$gpu start" >> "$log"
  while IFS=$'\t' read -r task model_label model_kind alpha model input output max_new; do
    local key
    key=$(printf "%s\n" "$task|$model_label|$output" | sha1sum | awk '{print $1}')
    local lock="$OUT_ROOT/queue/locks/gen_${key}.lock"
    local done="$OUT_ROOT/queue/done/gen_${key}.done"
    local failed="$OUT_ROOT/queue/failed/gen_${key}.failed"
    [[ -f "$done" ]] && continue
    mkdir "$lock" 2>/dev/null || continue
    local expected
    expected=$(expected_rows "$input")
    if complete_jsonl "$output" "$expected"; then
      echo "[$(date -Is)] gpu=$gpu skip complete $model_label rows=$expected" >> "$log"
      touch "$done"
      rmdir "$lock" || true
      continue
    fi
    rm -f "$output" "${output%.jsonl}.manifest.json"
    echo "[$(date -Is)] gpu=$gpu start $model_label rows=$expected max_new=$max_new batch=$GEN_BATCH_SIZE" >> "$log"
    local extra=()
    if [[ "$model_kind" == "steer" ]]; then
      extra=(--delta_checkpoint "$DELTA" --alpha "$alpha" --layer "$STEER_LAYER")
    fi
    if CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="${HF_HOME:-/workspace/hf_cache}" "$PYTHON" scripts/eval/run_model_comparison_generation.py \
      --input_jsonl "$input" \
      --output_jsonl "$output" \
      --model "$model" \
      --model_kind "$model_kind" \
      --model_label "$model_label" \
      --batch_size "$GEN_BATCH_SIZE" \
      --max_input_length "$MAX_INPUT_LENGTH" \
      --max_new_tokens "$max_new" \
      --torch_dtype bfloat16 \
      "${extra[@]}" >> "$log" 2>&1; then
      if complete_jsonl "$output" "$expected"; then
        echo "[$(date -Is)] gpu=$gpu done $model_label rows=$expected" >> "$log"
        touch "$done"
      else
        echo "[$(date -Is)] gpu=$gpu bad row count $model_label" >> "$log"
        touch "$failed"
      fi
    else
      echo "[$(date -Is)] gpu=$gpu failed $model_label" >> "$log"
      touch "$failed"
    fi
    rmdir "$lock" 2>/dev/null || true
  done < "$QUEUE"
  echo "[$(date -Is)] capability worker gpu=$gpu done" >> "$log"
}

IFS=',' read -r -a GPU_LIST <<< "$DEVICES"
for gpu in "${GPU_LIST[@]}"; do
  run_generation_worker "$gpu" > "$OUT_ROOT/logs/generation_worker_${gpu}.out" 2>&1 &
done
wait

"$PYTHON" scripts/eval/summarize_model_comparison_eval.py --root "$OUT_ROOT" > "$OUT_ROOT/logs/strict_summary.log" 2>&1 || true
"$PYTHON" scripts/eval/rescore_capability_outputs.py --root "$OUT_ROOT" --write_examples > "$OUT_ROOT/logs/rescore_summary.log" 2>&1

echo "[$(date -Is)] capability rescore rerun done: $OUT_ROOT"
