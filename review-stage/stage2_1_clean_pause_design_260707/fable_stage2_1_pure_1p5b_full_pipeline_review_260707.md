# Fable Review: Stage2.1-pure 1.5B Full Pipeline - 2026-07-07

## Verdict

`OK_TO_RUN`

Fable reviewed:

- `configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml`
- `pipelines/run_stage21_pure_1p5b_full.sh`
- `scripts/run_stage2_sft.py`
- `configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_2xa6000.yaml`
- parity with `pipelines/run_stage21_pure_8b_full.sh`

## Main Findings

- No code blockers.
- `max_steps: null` clears the parent `max_steps: 400`; training is epoch-based.
- `num_train_epochs: 1.0`, `EARLY_STOPPING_ENABLED=false`, and
  `load_best_model_at_end=false` flow through correctly.
- CLI batch overrides work end-to-end through `runtime.sft` into the legacy
  launcher.
- Pause location and token identity match the research goal:
  `cot_offset: 5`, three repeated pure `<|pause|>` tokens, natural eval without
  forced insertion, and forced eval after cot_4 / before cot_5.
- Cold `/workspace` checkpoint paths are consistent between train and eval.
- `RUN_JUDGE=0` plus `RUN_SUMMARY=1` is safe because the summary phase tolerates
  absent judge files.
- `adamw_torch` means disabling the bitsandbytes preflight in the wrapper is
  harmless.

## Non-Blocking Risks

- If selecting the best checkpoint across the whole run, keep enough
  checkpoints; otherwise early checkpoints may be rotated away.
- Batch probing should either preserve the effective batch size or explicitly
  record deviations, because changing effective batch changes optimization.
- Add a short plan/result amendment for this 1.5B full run before spending GPU
  time.

## Response

- Set 1.5B full `save_total_limit: 64`, enough for the expected full 1-epoch
  run with `save_steps: 25`.
- Updated plan/result docs with the 1.5B full run settings and review status.
