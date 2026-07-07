# Fable Review: Stage2.1-pure 1.5B Smoke/Gate Automation

Date: 2026-07-07

Scope:

- `configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_smoke_2xa6000.yaml`
- `pipelines/run_stage21_pure_1p5b_smoke.sh`
- related plan/res documentation

## Verdict

Fable found one blocker before the smoke pipeline should be committed/run:

- `diag_stage2_checkpoint.py` wrote `gate.status: "fail"` to JSON but still exited
  `0`, so the pipeline would appear green even if the natural exact-3/location
  gate failed.

The required fix was to make the gate strict, either via a `--strict` flag or a
shell-side JSON check.

## Fable's Answers

1. The only blocker was the non-strict gate exit behavior.
2. The smoke gate does test the natural condition, not the forced condition:
   - the gate reads only `stage21_pure_cot5_natural_{capability,safety}.jsonl`;
   - that condition has `insert_pause_after_cot_tokens: -1`;
   - per-row `natural_pause_metrics` is computed on the natural generated text,
     excluding any forced inserted prefix.
3. The base / natural / forced condition design is clean:
   - base is the no-training reference;
   - natural is the actual Stage2.1 claim;
   - forced is a ceiling/control condition.
4. `RUN_JUDGE=0` by default is reasonable because the smoke gate is structural
   and judge scores at 25 steps x 32 prompts are noisy and GPU-expensive.
5. Minimal pre-run fixes/checks:
   - make the gate strict;
   - confirm the target smoke machine has two GPUs or override devices;
   - remember that 32-row source-wise gates are nearly zero-tolerance under
     `min_exact_chain=0.97` and `max_off_target=0.005`.

## Follow-up Fixes Applied

- Added `--strict` to `scripts/diag_stage2_checkpoint.py`.
- Updated `pipelines/run_stage21_pure_1p5b_smoke.sh` to call the gate with
  `--strict`.
- Added `expected_cot_offset` to the existing-metrics reuse check.
- Added tests covering:
  - no reuse of existing metrics when expected offset differs;
  - strict gate exits non-zero on fail.

## Remaining Suggestions

- Confirm GPU count on the actual 1.5B smoke node before running the inherited
  two-device eval config.
- Interpret 25-step smoke gate failures as strict structural sanity failures,
  not as final evidence that Stage2.1-pure cannot work.
