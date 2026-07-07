# Fable Review: Stage2.1-pure 8B Formal Pipeline

Date: 2026-07-07

Scope:

- `configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_2xa6000.yaml`
- `configs/experiment/stage2_model_comparison_eval_8b_stage21_pure_cot5_2xa100.yaml`
- `pipelines/run_stage21_pure_8b_full.sh`

## Initial Verdict

Fable found one blocker:

- The initial 8B wrapper defaulted `STAGE21_PURE_8B_CHECKPOINT` to the hot
  output root. The Stage2 hot checkpoint watcher syncs the full output
  directory, including `final/`, to cold `/workspace/outputs` and then removes
  the hot copy. Therefore generation would fail after training because the
  exported SFT checkpoint path would no longer exist.

Required fix:

- Default `STAGE21_PURE_8B_CHECKPOINT` to
  `${COT_SAFETY_COLD_ROOT}/outputs/deepseek_8b_stage21_pause_pure_cot5_full_2xa100/final`.

## Confirmed Clean Points

- Explicit `expected_cot_offset: 5` and three pure `pause_tokens` in formal eval
  configs are correct and necessary for natural location metrics.
- Strict gate is wired to the natural outputs, not the forced outputs.
- Base condition receiving `pause_tokens` for text-level scanning only is safe:
  insertion is model-kind/forced-mode gated, and base metadata records no
  inserted pauses.
- Judge defaults on for 8B formal, and strict gate runs before judge.

## Follow-up Fixes

- Changed the 8B wrapper checkpoint default to the cold output path.
- Added `RUN_SYNC_COLD=1` default to sync hot run outputs to cold after summary.
- Added a checkpoint readiness gate before generation.
- Added an EXIT trap so strict-gate or judge failures still attempt to sync hot
  run outputs to cold for diagnosis.

## Final Fable Verdict

Fable follow-up verdict: no blocker, clear to commit/push.

Fable specifically verified:

- checkpoint gate is placed after training and returns cleanly under `set -e`;
- EXIT trap sync does not mask the original failure exit code;
- explicit end-of-script sync and trap cannot double-run because of the
  `SYNC_COLD_DONE` guard.
