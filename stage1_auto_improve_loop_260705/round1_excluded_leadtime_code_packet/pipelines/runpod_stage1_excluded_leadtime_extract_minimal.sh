#!/usr/bin/env bash
set -euo pipefail

# Extract the minimal hidden arrays needed for the Fable-5 reviewed
# excluded-source lead-time confirmation. This is intentionally narrower than
# the full Stage1 extractor: SR/RS only, layer 28 only, and cot offsets
# 4,8,16,32,64 only.

REPO_DIR="${REPO_DIR:-/workspace/cot-safety}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/cot-natural}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
FREEZE_DIR="${FREEZE_DIR:-/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099}"
PREPARED_ROOT="${PREPARED_ROOT:-${FREEZE_DIR}/stage1_prepared}"
HIDDEN_ARCHIVE_ROOT="${HIDDEN_ARCHIVE_ROOT:-/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives}"
LOG_DIR="${LOG_DIR:-/workspace/logs/stage1_excluded_leadtime_260705}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
SOURCES="${SOURCES:-strongreject_full reasoningshield}"
SPLITS="${SPLITS:-train val test}"
LAYER="${LAYER:-28}"
COT_OFFSETS="${COT_OFFSETS:-4,8,16,32,64}"
BATCH_SIZE="${BATCH_SIZE:-18}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"

export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HIDDEN_ARCHIVE_ROOT}" "${LOG_DIR}"
cd "${REPO_DIR}"

for source in ${SOURCES}; do
  out_dir="${HIDDEN_ARCHIVE_ROOT}/stage1_natural_pairs_8b_a100_1x_loso_${source}"
  mkdir -p "${out_dir}"
  for split in ${SPLITS}; do
    input_file="${PREPARED_ROOT}/${source}/normalized/${split}.jsonl"
    stem="natural_pairs_8b_a100_1x_loso_${source}_${split}_dense_cot_layers_${LAYER}"
    output_npz="${out_dir}/${stem}.npz"
    metadata_jsonl="${out_dir}/${stem}.metadata.jsonl"
    manifest_json="${out_dir}/${stem}.manifest.json"
    progress_json="${out_dir}/${stem}.progress.json"
    log_file="${LOG_DIR}/extract_${source}_${split}.log"

    if [[ ! -s "${input_file}" ]]; then
      echo "missing prepared split: ${input_file}" >&2
      exit 2
    fi
    if [[ -s "${output_npz}" && -s "${metadata_jsonl}" && -s "${manifest_json}" ]]; then
      echo "skip existing ${source}/${split}: ${output_npz}"
      continue
    fi

    echo "===== START extract ${source}/${split} $(date -Is) ====="
    "${PYTHON}" legacy/PauseProbe/scripts/probe/extract_hidden_states.py \
      --model "${MODEL}" \
      --tokenizer "${TOKENIZER}" \
      --input_file "${input_file}" \
      --output_npz "${output_npz}" \
      --metadata_jsonl "${metadata_jsonl}" \
      --manifest_json "${manifest_json}" \
      --progress_json "${progress_json}" \
      --label_field trajectory_safety_label \
      --layers "${LAYER}" \
      --cot_offsets "${COT_OFFSETS}" \
      --prompt_positions "" \
      --pause_layout none \
      --n_pause_tokens 0 \
      --batch_size "${BATCH_SIZE}" \
      --max_length "${MAX_LENGTH}" \
      --device "${DEVICE}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --save_dtype "${SAVE_DTYPE}" \
      --trust_remote_code \
      2>&1 | tee "${log_file}"
    echo "===== END extract ${source}/${split} $(date -Is) ====="
  done
  touch "${out_dir}/.excluded_leadtime_minimal_hidden_archived.ok"
done

echo "===== EXCLUDED_LEADTIME_MINIMAL_EXTRACTION_DONE $(date -Is) root=${HIDDEN_ARCHIVE_ROOT} ====="
