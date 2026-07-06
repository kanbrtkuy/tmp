#!/usr/bin/env bash
set -euo pipefail

cd /workspace/cot-safety
mkdir -p logs/stage3_remaining_folds
PYTHON=/workspace/venvs/stage2/bin/python

configs=(
  configs/experiment/stage3_intra_pause_probe_stage1_paired_harmbench_1p5b_cot5_2xa6000.yaml
  configs/experiment/stage3_intra_pause_probe_stage1_paired_reasoningshield_1p5b_cot5_2xa6000.yaml
  configs/experiment/stage3_intra_pause_probe_stage1_paired_strongreject_1p5b_cot5_2xa6000.yaml
)

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "==== $(date -Is) START ${name} ====" | tee -a logs/stage3_remaining_folds/stage3_remaining_folds.log
  "$PYTHON" scripts/run_stage3_intra_pause_probe.py --config "$cfg" --python "$PYTHON" --skip_pooled --skip_existing \
    2>&1 | tee -a "logs/stage3_remaining_folds/${name}.log"
  "$PYTHON" scripts/run_stage3_evidence_report.py --config "$cfg" \
    2>&1 | tee -a "logs/stage3_remaining_folds/${name}.evidence.log"
  echo "==== $(date -Is) DONE ${name} ====" | tee -a logs/stage3_remaining_folds/stage3_remaining_folds.log
done
