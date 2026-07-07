# Fable Review: Stage2.1 Checkpoint Selection Layer

Date: 2026-07-07

## Scope

This review covered the newly added checkpoint-selection layer for the running
1.5B Stage2.1-pure experiment:

- `scripts/prepare_model_comparison_eval_data.py`
- `configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml`
- `scripts/select_stage21_checkpoint.py`
- `plan/stage2_1_pure_plan_260707_zh.md`
- `res/stage2_1_pure_code_status_260707_zh.md`
- `review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_running_review_260707.md`

## Fable Verdict

Fable returned `EDITS_REQUIRED` for the selection layer only. It explicitly said
the running training needs no intervention.

## Blockers Found

1. `gate_score` used falsy `or` handling for numeric gate metrics.

   This meant a real `0.0` value for `max_off_target` or `max_malformed` was
   treated like a missing value and replaced by `1.0`, incorrectly penalizing a
   perfect checkpoint in the ranking.

2. The initial selection-dev config could silently create empty source groups.

   `harmbench_standard` and `jailbreakbench` were configured with
   `sample_offset: 300`, but those source files can have fewer than or only
   slightly more than 300 rows. The resulting selection-dev source could become
   empty without failing loudly.

## Fixes Applied

1. Fixed `gate_score` in `scripts/select_stage21_checkpoint.py`.

   The helper now treats only `None` as missing. Real numeric `0.0` values are
   preserved, so zero off-target and zero malformed rates rank correctly.

2. Added `min_rows` validation to `scripts/prepare_model_comparison_eval_data.py`.

   Each eval source can now specify a required minimum number of rows after
   `limit` and `sample_offset`. If the selection-dev subset is empty or too
   small, prepare fails immediately with a source-specific error.

3. Revised the 1.5B selection-dev config.

   The selection-dev config now uses:

   - capability: `gsm8k`, `math500`
   - safety: `strongreject`, `xstest`, `or_bench_hard`

   `harmbench_standard` and `jailbreakbench` remain in the full eval config, but
   are not used for checkpoint selection because the current raw files do not
   support the intended `sample_offset` without risking empty groups.

## Verification

Local:

- `python3 -m py_compile scripts/prepare_model_comparison_eval_data.py scripts/select_stage21_checkpoint.py`
- `gate_score` sanity check confirmed real `0.0` values remain `0.0`.
- `min_rows` sanity check confirmed too-small sources raise `ValueError`.

RunPod:

- Synced fixed files to `/workspace/cot-safety`.
- Remote `py_compile` passed.
- Restored raw eval data from R2:

  `cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/data/eval`

- Selection-dev prepare passed:

  - capability rows: `200`
  - safety rows: `173`
  - capability sources: `gsm8k`, `math500`
  - safety sources: `strongreject`, `xstest`, `or_bench_hard`
  - offsets: `gsm8k=500`, `math500=300`, safety sources `300`

- Full eval prepare passed:

  - capability rows: `800`
  - safety rows: `1300`
  - capability sources: `gsm8k`, `math500`
  - safety sources: `strongreject`, `harmbench_standard`, `jailbreakbench`,
    `xstest`, `or_bench_hard`

## Current Status

The two Fable blockers have been fixed and verified locally and on RunPod. The
fixed selection layer has been sent back to Fable for follow-up review.

## Fable Follow-Up Verdict

Fable follow-up returned `OK_TO_USE`.

It verified from the code diff that:

- `gate_score` now treats only `None` as missing, so real `0.0` values for
  `max_off_target` and `max_malformed` are ranked correctly.
- selection-dev disjointness holds because the same seed is used as full eval,
  full eval limits are `gsm8k=500`, `math500=300`, `strongreject=300`,
  `xstest=300`, `or_bench_hard=300`, and all selection-dev offsets equal or
  exceed those limits.
- `enforce_min_rows` prevents silent empty source groups.
- RunPod `min_rows` prepare passing proves there are enough rows past the offset.

Non-blocking risks noted by Fable:

1. `strongreject` selection-dev has only 13 rows after offset, so one row changes
   that group's rate by about 7.7%. Treat it as advisory when ties are close.
2. `write_summary` display formatting still uses `or 0.0`; this is display-only,
   not ranking logic.
3. Full-eval source counts should be checked with `wc -l`.
4. Do not run checkpoint selection while the cold watcher is still doing final
   sync/pruning. After training and final sync complete, use the current run's
   R2 root for sweep downloads.

Follow-up check for source counts:

```text
320  /workspace/data/eval/harmbench_standard/train.jsonl
100  /workspace/data/eval/jailbreakbench_behaviors/harmful.jsonl
313  /workspace/data/eval/strongreject/validation.jsonl
450  /workspace/data/eval/xstest/test.jsonl
1319 /workspace/data/eval/or_bench_hard_1k/train.jsonl
```

Correct sweep R2 root for the current run:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2
```
