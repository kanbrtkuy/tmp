# cot-safety

`cot-safety` is a config-driven refactor of the previous `PauseProbe` and
`COTPauseToken` experiment code.  The goal is to make the four-stage pause-token
safety pipeline reproducible across models without editing Python every time the
backbone changes.

## Four Stages

1. **PositionScan TrajProbe**: extract hidden states at CoT positions and test
   whether unsafe trajectory signal is linearly separable.
2. **Intra-Pause SFT**: train a model to emit pause tokens inside `<think>`
   before a configured CoT offset, currently before `cot_3`.
3. **Intra-Pause Probe**: extract `pause_0/pause_1/pause_2` hidden states and
   verify that pause positions are usable monitoring points.
4. **Pause-Only Steering**: learn/apply a steering vector only at pause token
   hidden states, then evaluate safety, hard-safe refusal, and capability.

## Design Principle

Core code describes reusable operations.  YAML config files describe the model,
data recipe, token template, pause placement, layers, judges, and hardware.  To
move from DeepSeek 1.5B to DeepSeek 8B, add or edit a model/runtime config rather
than changing the implementation.

## First-Stage Migration Status

This repository currently contains the new package skeleton and the critical
shared modules that are easiest to break during model migration:

- chat-template rendering
- intra-CoT pause insertion
- hidden-state position naming/location
- trajectory label/schema utilities
- probe feature matrix construction
- pause-only steering target validation

The legacy scripts in `PauseProbe` and `external/COTPauseToken` remain the source
of the already-validated full experiment logic while deeper migration continues.

## Quick Smoke Test

```bash
python -m pytest
```

## Example Commands

Dry-run config resolution:

```bash
cot-safety config show --config configs/experiment/stage3_intra_pause_probe.yaml
```

Validate that a steering config only targets pause tokens:

```bash
cot-safety steer validate-scope --config configs/experiment/stage4_pause_steering.yaml
```

## Security

Do not commit real Hugging Face tokens.  Keep tokens in a private file on the
machine and inject them into RunPod with SSH stdin as described in
`docs/runpod_setup.md`.
