# Round 2 Patch Notes

Local validation:

```bash
python -m py_compile \
  cot-safety/scripts/data/run_stage1_threshold_reanalysis.py \
  cot-safety/scripts/data/run_stage1_matched_horizon_reanalysis.py

cot-safety/.venv-stage1-test/bin/python -m pytest \
  cot-safety/tests/test_stage1_threshold_reanalysis.py \
  cot-safety/tests/test_stage1_matched_horizon_reanalysis.py
```

Result:

```text
4 passed in 2.29s
```

## Fixes

Module T:

- `policy_rows_for_arm.add(...)` now accepts `ci_rows`.
- `platt_0p5` test CI passes `test_cal`, so the bootstrap matches the Platt
  point estimate.
- `best_ba_threshold` now returns a real maximizing threshold rather than the
  midpoint of possibly non-contiguous maximizers.
- Added `test_platt_bootstrap_ci_uses_calibrated_scores`.

Module M:

- Added `enforce_pair_complete_alignment(left, right)`.
- After hidden prediction alignment, hidden/surface val and test records are
  re-aligned by shared ids and restricted to pairs that still contain both
  labels.
- Summary rows now record `*_pairs_dropped_post_alignment`,
  `*_rows_dropped_post_alignment`, and rank-pair counts.
- Added a fixture with one missing hidden test prediction and assertions that
  its pair is dropped.
- Added a direct Holm adjustment test.
- Added limitation: across-k trends are descriptive because censoring changes
  retained populations.
