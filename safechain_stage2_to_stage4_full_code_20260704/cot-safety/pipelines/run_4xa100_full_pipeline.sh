#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-configs/experiment/full_four_stage_8b_4xa100.yaml}"
RUN_STAGE1="${RUN_STAGE1:-0}"
RUN_STAGE2="${RUN_STAGE2:-0}"

cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/pipelines/runpod_base_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "${COT_SAFETY_RUN_ROOT}"
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python -m cot_safety.cli config show --config "$CONFIG" > "${COT_SAFETY_RUN_ROOT}/full_four_stage_resolved.yaml"
PYTHONPATH=src python -m cot_safety.cli pipeline plan --config configs/experiment/stage1_positionscan_8b_4xa100.yaml > "${COT_SAFETY_RUN_ROOT}/stage1_positionscan_plan.json"
PYTHONPATH=src python -m cot_safety.cli pipeline plan --config configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml > "${COT_SAFETY_RUN_ROOT}/stage2_intra_pause_sft_plan.json"
PYTHONPATH=src python -m cot_safety.cli pipeline plan --config configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot4_4xa100.yaml > "${COT_SAFETY_RUN_ROOT}/stage2_model_comparison_eval_plan.json"
PYTHONPATH=src python -m cot_safety.cli steer validate-scope --config configs/experiment/stage4_pause_gprs_8b_4xa100.yaml

echo "Config and safety-scope smoke checks passed for 4xA100."
echo "Set RUN_STAGE1=1 to launch Stage 1 PositionScan with pipelines/run_4xa100_stage1_positionscan.sh."
echo "Set RUN_STAGE2=1 to launch Stage 2 cot4 KL-transparent pause SFT with pipelines/run_4xa100_stage2_sft.sh."
echo "Stage 2 comparison eval defaults to the KL-transparent cot4 checkpoint chain."

if [[ "$RUN_STAGE1" == "1" ]]; then
  bash pipelines/run_4xa100_stage1_positionscan.sh
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  bash pipelines/run_4xa100_stage2_sft.sh
fi
