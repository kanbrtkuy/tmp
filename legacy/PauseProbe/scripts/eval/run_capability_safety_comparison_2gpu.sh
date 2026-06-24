#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/PauseProbe}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/capability_safety_eval_260622}"
OUT_ROOT="${OUT_ROOT:-/workspace/PauseProbe/runs/eval/capability_safety_260622}"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
SFT_MODEL="${SFT_MODEL:-/workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final}"
DELTA="${DELTA:-/workspace/PauseProbe/runs/steering/intra_pause_learned_delta_260618/zero_l14_steps80/learned_delta.pt}"
STEER_ALPHA="${STEER_ALPHA:-2}"
STEER_LAYER="${STEER_LAYER:-14}"
DEVICES="${DEVICES:-0,1}"
JUDGES="${JUDGES:-wildguard,llamaguard,harmbench}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-96}"
CAP_MAX_NEW_TOKENS="${CAP_MAX_NEW_TOKENS:-768}"
SAFETY_MAX_NEW_TOKENS="${SAFETY_MAX_NEW_TOKENS:-768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
PYTHON="${PYTHON:-/workspace/venvs/probe/bin/python}"

mkdir -p "$OUT_ROOT"/{generations,judges,logs,queue/done,queue/locks,queue/failed}
cd "$ROOT"
source /workspace/secrets/hf.env

CAP_IN="$DATA_ROOT/capability_prompts.jsonl"
SAFETY_IN="$DATA_ROOT/heldout_safety_prompts.jsonl"
GEN_QUEUE="$OUT_ROOT/queue/generation_jobs.tsv"
JUDGE_QUEUE="$OUT_ROOT/queue/judge_jobs.tsv"

cat > "$GEN_QUEUE" <<EOF
capability	base	base	0	$BASE_MODEL	$CAP_IN	$OUT_ROOT/generations/base_capability.jsonl	$CAP_MAX_NEW_TOKENS
capability	sft_alpha0	sft	0	$SFT_MODEL	$CAP_IN	$OUT_ROOT/generations/sft_alpha0_capability.jsonl	$CAP_MAX_NEW_TOKENS
capability	sft_steer_alpha${STEER_ALPHA}	steer	$STEER_ALPHA	$SFT_MODEL	$CAP_IN	$OUT_ROOT/generations/sft_steer_alpha${STEER_ALPHA}_capability.jsonl	$CAP_MAX_NEW_TOKENS
safety	base	base	0	$BASE_MODEL	$SAFETY_IN	$OUT_ROOT/generations/base_safety.jsonl	$SAFETY_MAX_NEW_TOKENS
safety	sft_alpha0	sft	0	$SFT_MODEL	$SAFETY_IN	$OUT_ROOT/generations/sft_alpha0_safety.jsonl	$SAFETY_MAX_NEW_TOKENS
safety	sft_steer_alpha${STEER_ALPHA}	steer	$STEER_ALPHA	$SFT_MODEL	$SAFETY_IN	$OUT_ROOT/generations/sft_steer_alpha${STEER_ALPHA}_safety.jsonl	$SAFETY_MAX_NEW_TOKENS
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
  echo "[$(date -Is)] generation worker gpu=$gpu start" >> "$log"
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
      echo "[$(date -Is)] gpu=$gpu skip complete generation $model_label $task rows=$expected" >> "$log"
      touch "$done"
      rmdir "$lock" || true
      continue
    fi
    rm -f "$output" "${output%.jsonl}.manifest.json"
    echo "[$(date -Is)] gpu=$gpu start generation $model_label $task rows=$expected max_new=$max_new" >> "$log"
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
        echo "[$(date -Is)] gpu=$gpu done generation $model_label $task rows=$expected" >> "$log"
        touch "$done"
      else
        echo "[$(date -Is)] gpu=$gpu bad row count generation $model_label $task" >> "$log"
        touch "$failed"
      fi
    else
      echo "[$(date -Is)] gpu=$gpu failed generation $model_label $task" >> "$log"
      touch "$failed"
    fi
    rmdir "$lock" 2>/dev/null || true
  done < "$GEN_QUEUE"
  echo "[$(date -Is)] generation worker gpu=$gpu done" >> "$log"
}

IFS=',' read -r -a GPU_LIST <<< "$DEVICES"
for gpu in "${GPU_LIST[@]}"; do
  run_generation_worker "$gpu" > "$OUT_ROOT/logs/generation_worker_${gpu}.out" 2>&1 &
done
wait

"$PYTHON" scripts/eval/summarize_model_comparison_eval.py --root "$OUT_ROOT" > "$OUT_ROOT/logs/summary_after_generation.log" 2>&1 || true

: > "$JUDGE_QUEUE"
for model_label in base sft_alpha0 "sft_steer_alpha${STEER_ALPHA}"; do
  input="$OUT_ROOT/generations/${model_label}_safety.jsonl"
  IFS=',' read -r -a JUDGE_LIST <<< "$JUDGES"
  for judge in "${JUDGE_LIST[@]}"; do
    judge_dir="$OUT_ROOT/judges/$judge"
    mkdir -p "$judge_dir"
    printf "%s\t%s\t%s\t%s\t%s\n" "$judge" "$model_label" "$input" "$judge_dir/${model_label}_raw.jsonl" "$judge_dir/${model_label}_normalized.jsonl" >> "$JUDGE_QUEUE"
  done
done

judge_batch_size() {
  case "$1" in
    harmbench) echo "${JUDGE_BATCH_SIZE_HARMBENCH:-1}" ;;
    llamaguard) echo "${JUDGE_BATCH_SIZE_LLAMAGUARD:-4}" ;;
    wildguard) echo "${JUDGE_BATCH_SIZE_WILDGUARD:-8}" ;;
    *) echo "${JUDGE_BATCH_SIZE:-2}" ;;
  esac
}

run_judge_worker() {
  local gpu="$1"
  local log="$OUT_ROOT/logs/judge_gpu${gpu}.log"
  echo "[$(date -Is)] judge worker gpu=$gpu start" >> "$log"
  while IFS=$'\t' read -r judge model_label input raw norm; do
    local key
    key=$(printf "%s\n" "$judge|$model_label|$input" | sha1sum | awk '{print $1}')
    local lock="$OUT_ROOT/queue/locks/judge_${key}.lock"
    local done="$OUT_ROOT/queue/done/judge_${key}.done"
    local failed="$OUT_ROOT/queue/failed/judge_${key}.failed"
    [[ -f "$done" ]] && continue
    mkdir "$lock" 2>/dev/null || continue
    local expected
    expected=$(expected_rows "$input")
    if complete_jsonl "$norm" "$expected"; then
      echo "[$(date -Is)] gpu=$gpu skip complete judge $judge $model_label rows=$expected" >> "$log"
      touch "$done"
      rmdir "$lock" || true
      continue
    fi
    rm -f "$raw" "${raw%.jsonl}.manifest.json" "$norm" "${norm%.jsonl}.manifest.json"
    local bs
    bs=$(judge_batch_size "$judge")
    echo "[$(date -Is)] gpu=$gpu start judge $judge $model_label rows=$expected batch=$bs" >> "$log"
    if CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="${HF_HOME:-/workspace/hf_cache}" "$PYTHON" scripts/judge/run_open_judges.py \
      --input_file "$input" \
      --output_jsonl "$raw" \
      --judges "$judge" \
      --batch_size "$bs" \
      --max_input_length "$MAX_INPUT_LENGTH" \
      --torch_dtype bfloat16 >> "$log" 2>&1 && \
      "$PYTHON" scripts/judge/normalize_judge_outputs.py \
        --input_file "$raw" \
        --output_jsonl "$norm" \
        --strategy conservative >> "$log" 2>&1; then
      if complete_jsonl "$norm" "$expected"; then
        echo "[$(date -Is)] gpu=$gpu done judge $judge $model_label rows=$expected" >> "$log"
        touch "$done"
      else
        echo "[$(date -Is)] gpu=$gpu bad row count judge $judge $model_label" >> "$log"
        touch "$failed"
      fi
    else
      echo "[$(date -Is)] gpu=$gpu failed judge $judge $model_label" >> "$log"
      touch "$failed"
    fi
    rmdir "$lock" 2>/dev/null || true
  done < "$JUDGE_QUEUE"
  echo "[$(date -Is)] judge worker gpu=$gpu done" >> "$log"
}

for gpu in "${GPU_LIST[@]}"; do
  run_judge_worker "$gpu" > "$OUT_ROOT/logs/judge_worker_${gpu}.out" 2>&1 &
done
wait

"$PYTHON" scripts/eval/summarize_model_comparison_eval.py --root "$OUT_ROOT" > "$OUT_ROOT/logs/final_summary.log" 2>&1
echo "[$(date -Is)] capability+safety comparison done: $OUT_ROOT"
