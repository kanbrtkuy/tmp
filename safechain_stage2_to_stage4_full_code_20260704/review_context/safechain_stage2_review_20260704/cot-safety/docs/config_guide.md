# Config Guide

## Switching Models

Add a file under `configs/model/` and set:

```yaml
model:
  name: deepseek_r1_distill_llama_8b
  base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  local_base_model: ${COT_SAFETY_MODEL_ROOT:-/workspace/models}/DeepSeek-R1-Distill-Llama-8B
  sft_checkpoint: ${COT_SAFETY_OUTPUT_ROOT:-/workspace/outputs}/deepseek_8b_intra_pause_cot3/final
  max_length: 4096
  default_layers: [8, 16, 20, 24, 28, 32]
```

If the chat markers differ, either choose an existing template or add a new
`model_template` block.

## Switching Hardware

Use a runtime config:

```yaml
defaults:
  - ../runtime/a100_4x.yaml
```

The 4×A100 config controls SFT batch accumulation, hidden extraction jobs,
generation workers, and judge workers.

## Storage Roots

Deployment scripts should set these variables instead of editing Python code:

```bash
export COT_SAFETY_MODEL_ROOT=/dev/shm/cot-safety-hot/models
export COT_SAFETY_JUDGE_ROOT=/dev/shm/cot-safety-hot/models/judges
export COT_SAFETY_DATA_ROOT=/dev/shm/cot-safety-hot/data
export COT_SAFETY_OUTPUT_ROOT=/dev/shm/cot-safety-hot/outputs
export COT_SAFETY_RUN_ROOT=/dev/shm/cot-safety-hot/runs
```

Use `/workspace` only as persistent cold storage on RunPod network volumes.
Before GPU-heavy training or judging, stage the active model/checkpoint/judge
directories with `pipelines/runpod_stage_hot_storage.sh`.

## Steering Scope

For this paper, steering must remain pause-only:

```yaml
steering:
  target_positions: [pause_0, pause_1, pause_2]
```

Use:

```bash
cot-safety steer validate-scope --config configs/experiment/stage4_pause_steering.yaml
```

before launching a steering run.
