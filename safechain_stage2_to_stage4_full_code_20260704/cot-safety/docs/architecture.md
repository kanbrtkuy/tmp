# Architecture

The pipeline separates three things that were mixed in the legacy scripts:

1. **Operations**: reusable Python modules for data prep, formatting, hidden
   extraction, probing, steering, generation, judging, and reporting.
2. **Recipes**: YAML configs describing the exact model, data source mix,
   position/layer scan, steering targets, alpha grid, judges, and runtime.
3. **Artifacts**: standardized run directories with resolved configs,
   manifests, logs, and outputs.

This lets the same code run DeepSeek 1.5B, DeepSeek 8B, or later models by
changing config files.

## Stage Mapping

| Stage | Old location | New module/config target |
| --- | --- | --- |
| PositionScan TrajProbe | `PauseProbe/scripts/probe` | `cot_safety.hidden`, `cot_safety.probes`, `configs/experiment/stage1_positionscan.yaml` |
| Intra-pause SFT | `external/COTPauseToken` | `cot_safety.sft`, `configs/experiment/stage2_intra_pause_sft.yaml` |
| Intra-pause Probe | `PauseProbe/scripts/probe/run_intra_pause_probe_full.py` | `cot_safety.hidden`, `cot_safety.probes`, `configs/experiment/stage3_intra_pause_probe.yaml` |
| Pause-only Steering | `PauseProbe/scripts/steering` | `cot_safety.steering`, `cot_safety.eval`, `configs/experiment/stage4_pause_steering.yaml` |

## Key Safety Constraint

The paper's steering intervention is pause-only.  The config must keep:

```yaml
steering:
  target_positions: [pause_0, pause_1, pause_2]
```

The code rejects `pre_pause_*`, `post_pause_*`, `cot_*`, and `control_cot_*`
targets for steering experiments.
