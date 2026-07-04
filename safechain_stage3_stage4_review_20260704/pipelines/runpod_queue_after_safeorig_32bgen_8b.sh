#!/usr/bin/env bash
set -euo pipefail

cd /workspace/cot-safety

QUEUE_LOG=/workspace/logs/queue_32bgen_8b_after_safeorig.log
RUN_LOG=/workspace/logs/stage1_32bgen_8b_a100_1x.log
CURRENT_SCREEN=${CURRENT_SCREEN:-stage1_32b_safeorig_32b}

mkdir -p /workspace/logs

{
  echo "[queue] waiting for ${CURRENT_SCREEN} to finish at $(date -Iseconds)"
  while screen -ls | grep -q "${CURRENT_SCREEN}"; do
    sleep 60
  done
  echo "[queue] ${CURRENT_SCREEN} finished at $(date -Iseconds)"
  echo "[queue] starting stage1_natural_pairs_32bgen_8b_a100_1x at $(date -Iseconds)"
} | tee -a "${QUEUE_LOG}"

bash pipelines/runpod_stage1_natural_pairs_32bgen_8b_a100_1x.sh 2>&1 | tee -a "${RUN_LOG}"

echo "[queue] stage1_natural_pairs_32bgen_8b_a100_1x finished at $(date -Iseconds)" | tee -a "${QUEUE_LOG}"
