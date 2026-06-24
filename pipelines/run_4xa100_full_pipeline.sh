#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cot-safety}"
CONFIG="${CONFIG:-configs/experiment/full_four_stage.yaml}"

cd "$ROOT"
if [[ -f /workspace/secrets/hf.env ]]; then
  source /workspace/secrets/hf.env
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p runs
PYTHONPATH=src python scripts/smoke_test.py
cot-safety config show --config "$CONFIG" > runs/full_four_stage_resolved.yaml
cot-safety steer validate-scope --config configs/experiment/stage4_pause_steering.yaml

echo "Config and safety-scope smoke checks passed. Deeper stage runners are migrated next."
