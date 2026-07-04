#!/usr/bin/env bash
set -euo pipefail

# Hard-safe / over-refusal audit for the intra-pause steering delta.
#
# This launcher prepares XSTest / OR-Bench prompt-only rows, then delegates to
# run_intra_pause_full_steering_eval.sh.  The delegated generation script applies
# the steering delta only at pause_0/1/2.  It never modifies pre_pause_* or
# post_pause_* positions.

ROOT="${ROOT:-/workspace/PauseProbe}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/intra_pause_hardsafe_steering_eval_260621}"
OUT_ROOT="${OUT_ROOT:-/workspace/PauseProbe/runs/steering/intra_pause_hardsafe_steering_eval_260621}"

PREPARE_DATA="${PREPARE_DATA:-1}"
DEVICES="${DEVICES:-0,1}"
SEEDS="${SEEDS:-260621 260622 260623}"
ALPHAS="${ALPHAS:-0,1,2}"

# Keep generation parallel across GPUs.  WildGuard is memory-heavy, so judge
# defaults are conservative; raise these only after checking nvidia-smi.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
MAX_PARALLEL_GENERATION_JOBS="${MAX_PARALLEL_GENERATION_JOBS:-2}"
MAX_PARALLEL_JUDGE_JOBS="${MAX_PARALLEL_JUDGE_JOBS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

cd "${ROOT}"
mkdir -p "${DATA_ROOT}" "${OUT_ROOT}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON}" scripts/data/prepare_intra_pause_hardsafe_steering_data.py \
    --output_dir "${DATA_ROOT}"
fi

DATASET_SPECS_FILE="${DATA_ROOT}/dataset_specs.tsv"
export ROOT PYTHON DATASET_SPECS_FILE OUT_ROOT DEVICES SEEDS ALPHAS
export GEN_BATCH_SIZE JUDGE_BATCH_SIZE MAX_PARALLEL_GENERATION_JOBS MAX_PARALLEL_JUDGE_JOBS MAX_NEW_TOKENS

bash scripts/steering/run_intra_pause_full_steering_eval.sh
