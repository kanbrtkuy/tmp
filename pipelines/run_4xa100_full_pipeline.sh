#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
CONFIG="${CONFIG:-configs/experiment/full_four_stage_8b_4xa100.yaml}"
RUN_STAGE1="${RUN_STAGE1:-0}"

cd "$ROOT"
if [[ -f /workspace/secrets/hf.env ]]; then
  source /workspace/secrets/hf.env
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p runs
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python -m cot_safety.cli config show --config "$CONFIG" > runs/full_four_stage_resolved.yaml
PYTHONPATH=src python -m cot_safety.cli pipeline plan --config configs/experiment/stage1_positionscan_8b_4xa100.yaml > runs/stage1_positionscan_plan.json
PYTHONPATH=src python -m cot_safety.cli steer validate-scope --config configs/experiment/stage4_pause_steering_8b_4xa100.yaml

echo "Config and safety-scope smoke checks passed for 4xA100."
echo "Set RUN_STAGE1=1 to launch Stage 1 PositionScan with pipelines/run_4xa100_stage1_positionscan.sh."

if [[ "$RUN_STAGE1" == "1" ]]; then
  bash pipelines/run_4xa100_stage1_positionscan.sh
fi
