# Stage1 Auto-Improve Round 1 A1 Code Packet

Date: 2026-07-05

This packet contains the A1 CPU-only score-pooling implementation requested by
Fable-5 in the Round 1 result review. It contains code and synthetic tests only;
no raw prompts, raw CoTs, hidden activations, generated pairs, or production
prediction files.

## Files

- `scripts/data/run_stage1_score_pooling_reanalysis.py`
- `tests/test_stage1_score_pooling_reanalysis.py`

## Local Validation

Commands:

```bash
python -m py_compile cot-safety/scripts/data/run_stage1_score_pooling_reanalysis.py

cot-safety/.venv-stage1-test/bin/python -m pytest \
  cot-safety/tests/test_stage1_score_pooling_reanalysis.py
```

Result:

```text
2 passed in 0.64s
```

## Implementation Summary

Primary A1 rule:

```text
hidden_score_at_k = unweighted mean over j <= k of
  (hidden_score_j - mean_val_hidden_score_j) / std_val_hidden_score_j
```

Outputs:

- `stage1_score_pooling_preregistration.json`
- `stage1_score_pooling_summary.tsv/json`
- `stage1_score_pooling_lead_time_matrix.tsv`
- `stage1_score_pooling_position_diagnostics.tsv`
- `stage1_score_pooling_score_histograms.tsv`

## Fable-5 Requested Checks

1. No future positions:
   - `pool_ks_for(target_k, k_grid)` returns exactly `j <= k` and asserts no
     future position.

2. Identical retained pair IDs:
   - `align_records(...)` aligns by id and checks labels/pair ids.
   - `enforce_pair_complete(...)` restricts both arms to complete safe/unsafe
     pairs and asserts retained pair sets match.

3. Val-only z stats:
   - `z_stats(...)` reads only `data[source][k][("hidden", "val")]`.
   - Production test scores are never used for z normalization.

4. Fixed layer/family primary:
   - CLI default `--selected-layer 28`, `--surface-family char_tfidf`.
   - The preregistration JSON records these before results are written.

5. Preregistered pooling rule and Holm family:
   - `stage1_score_pooling_preregistration.json` is written before summary
     rows.
   - Output JSON echoes the same preregistration.

6. Position metadata assertion:
   - `read_predictions(..., expected_k=k)` calls
     `assert_position_metadata(...)` for hidden predictions.
   - If explicit metadata is absent, the script records
     `fallback_path_assertion`; the path itself must be under `k_{k}`.

7. Pair bootstrap:
   - `bootstrap_delta(...)` resamples `match_family` groups and uses a numpy
     RNG. B defaults to `500`.

## Review Request

Please verify whether this code is OK to run on RunPod against:

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/matched_horizon/predictions
```

Return:

- Verdict: `OK_TO_RUN` or `BLOCKED`.
- Any true leakage/cherry-picking blocker.
- Any correctness blocker in pooling, alignment, bootstrap, or lead-time matrix.
- If blocked, exact patch required.
