# Fable Follow-up: Stage2.1-pure Smoke Strict Gate

Date: 2026-07-07

Follow-up after fixing the blocker from
`fable_stage2_1_pure_smoke_gate_review_260707.md`.

## Verdict

Fable verdict: no blocker. The strict gate fix resolves B1.

## Verified Fixes

- `scripts/diag_stage2_checkpoint.py` now has `--strict`, which raises
  `SystemExit(1)` when the natural pause gate fails.
- `pipelines/run_stage21_pure_1p5b_smoke.sh` passes `--strict` to the gate.
  With `set -euo pipefail`, a failed gate aborts the pipeline.
- Existing metrics reuse now requires both:
  - matching `pause_tokens`;
  - matching `expected_cot_offset`.
- Added tests cover:
  - stale expected-offset metrics are recomputed;
  - strict fail exits non-zero while still writing `gate.status == "fail"`.

## Optional Suggestions

- If a non-empty pause separator variant is ever introduced, add `separator` to
  emitted metrics and include it in the reuse guard.
- Run `tests/test_stage2_checkpoint_diag.py` once on the remote GPU venv, since
  local pytest is unavailable.

Conclusion: clear to commit/push.
