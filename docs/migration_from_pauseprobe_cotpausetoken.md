# Migration From Legacy Repos

## Legacy Repositories

- `PauseProbe`: trajectory probes, hidden extraction, steering, judges, evals.
- `external/COTPauseToken`: SFT data construction and TRL full-SFT training.

## Migration Strategy

The migration is intentionally staged:

1. Port shared semantics and tests first.
2. Keep legacy launchers callable while new runners are implemented.
3. Replace hard-coded defaults with YAML configs.
4. Preserve known 1.5B experiment configs as compatibility recipes.
5. Add DeepSeek 8B configs without changing core code.

## Legacy Compatibility Layer

The new repo keeps a code-only compatibility copy under:

```text
legacy/PauseProbe
legacy/COTPauseToken
```

This copy excludes data, logs, model checkpoints, run outputs, and caches.  It
exists so that the new config and CLI layer can launch already-validated
entrypoints while native runners are being ported.

Use:

```bash
cot-safety pipeline plan --config configs/experiment/stage4_pause_steering.yaml
```

to inspect the current planned commands for a stage.

## Critical Behaviors To Preserve

- Pause insertion is before `cot_3` after leading whitespace tokens are skipped.
- Stage 3 may diagnose pre/post/control positions, but Stage 4 steering only
  modifies `pause_0/pause_1/pause_2`.
- Partial labels are dropped from binary probe training unless a diagnostic
  config explicitly asks for them.
- SFT uses high-quality open CoT trajectories, not 1.5B self-generated
  trajectories.
