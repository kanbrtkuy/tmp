# Fable Follow-up Review: Stage3 B1/B2 Fix Verification — 2026-07-06

**Verdict:** B1 and B2 are resolved. OK to launch the remaining Stage3 folds: extraction + scan + evidence report.

One new follow-up issue was introduced by the B2 fix: it blocks Stage4 gating, not the Stage3 fold runs.

## B1 — Bootstrap Cluster Unit: Resolved

- `_cluster_key` now prefers `pair_id` -> `prompt_key` -> prediction key.
- The new regression test with same `pair_id` and different `prompt_key`s genuinely discriminates the fix.
- The data path is end-to-end: extraction writes `pair_ids`, merge carries them, and `train_probe.py` emits `pair_id` in prediction rows.

## B2 — Top-Level Status: Resolved

- `_top_level_status` makes top-level `status` a composite.
- `independent_status` is promoted to top level.
- The previous bad case now yields `pass_pause_signal_only_independent_not_established`, not `fail_no_independent_pause_signal`.

## New Follow-Up Before Stage4

`src/cot_safety/steering/gprs.py` still expects `status == "pass"`, which the new reports no longer emit. This is fail-closed, not false-ready, but it must be fixed before Stage4 gating:

- gate on `independent_status == "pass"`, or
- gate on `status == "pass_independent"`.

## Remaining Process Blockers Before Final Fold Verdicts

1. Rerun WJB scan + evidence report under the new probe settings before cross-fold comparison.
2. Fix the GPRS gate string before Stage4.
3. Any adjudication that reads top-level `status` must use the new vocabulary.

Bottom line: launch harmbench/reasoningshield/strongreject extraction + scan now; land the gate fix and WJB rerun before writing cross-fold verdicts.
