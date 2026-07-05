# Fable-5 Round 1 Response

Verdict: `BLOCKED`

Fable-5 reviewed the sanitized code packet and found two small blockers:

1. Module T hard blocker:
   - `platt_0p5` point estimates used calibrated test probabilities, but the
     bootstrap CI was computed on raw `test_rows` at threshold `0.5`.
   - Required fix: allow the bootstrap rows to be overridden and pass
     calibrated rows for the Platt test CI.

2. Module M conditional blocker:
   - The script enforced pair-complete censoring before hidden-score alignment,
     but hidden prediction alignment could drop one side of a pair afterward.
   - Required fix: after hidden/surface alignment, re-restrict both arms to
     shared ids whose pair still has both safe and unsafe labels; record
     dropped pairs.

Non-blocking suggestions:

- Return an actual maximizing threshold for `val_ba_max` rather than averaging
  non-contiguous maximizers.
- Add tests for Platt CI consistency, post-alignment pair completeness, and
  Holm adjustment.
- State that E3 is secondary because it reuses validation for selection and
  stacker training.
- State that across-k trends use changing censored populations.

Round 2 packet includes patches for all required fixes and the cheap test
additions.

