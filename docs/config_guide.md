# Config Guide

## Switching Models

Add a file under `configs/model/` and set:

```yaml
model:
  name: deepseek_r1_distill_llama_8b
  base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  local_base_model: /workspace/models/DeepSeek-R1-Distill-Llama-8B
  sft_checkpoint: /workspace/outputs/deepseek_8b_intra_pause_cot3/final
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
