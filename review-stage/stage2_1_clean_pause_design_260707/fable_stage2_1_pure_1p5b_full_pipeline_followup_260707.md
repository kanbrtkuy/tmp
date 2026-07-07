# Fable Follow-up: Stage2.1-pure 1.5B Full Pipeline - 2026-07-07

## Verdict

`OK_TO_RUN`

Fable verified the post-review changes directly:

- `save_total_limit: 64`
- CLI batch overrides in `scripts/run_stage2_sft.py`
- full checkpoint path in the 1.5B eval config
- 1.5B full wrapper
- documentation amendments

No new blockers.

Advisory note: budget cold-storage disk for 64 checkpoints at `save_steps: 25`.
