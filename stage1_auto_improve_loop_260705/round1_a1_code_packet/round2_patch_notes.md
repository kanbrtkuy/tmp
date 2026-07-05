# Round 2 Patch Notes

Local validation:

```bash
python -m py_compile cot-safety/scripts/data/run_stage1_score_pooling_reanalysis.py
cot-safety/.venv-stage1-test/bin/python -m pytest \
  cot-safety/tests/test_stage1_score_pooling_reanalysis.py
```

Result:

```text
3 passed in 0.92s
```

Fixes:

- Added `val_score_stats(...)` and `apply_z(...)`.
- Computed `hidden_val_pooled_stats[(source,k)]` from validation pooled hidden
  scores.
- Computed `surface_val_stats[(source,k)]` from validation surface scores.
- For `source="pooled"` summary and lead-time rows, both arms are now
  per-source val-z-scored before cross-source concatenation.
- Added preregistration fields:
  - `combined_source_normalization`
  - `monotone_tolerance`
  - `success_rule`
- Surface prediction files now also pass `expected_k=k`; absent metadata falls
  back to path assertion, explicit metadata is checked.
- Added `test_pooled_rows_are_cross_source_calibrated`.
