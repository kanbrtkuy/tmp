# Stage1 CPU Reanalysis Code Review Packet

Date: 2026-07-05

This packet is sanitized for external/model review. It contains code snapshots,
unit tests, and protocol descriptions only. It intentionally excludes raw
prompts, raw CoTs, hidden activations, generated pairs, and live experiment
status.

## Files

- `scripts/data/run_stage1_threshold_reanalysis.py`
  - Module T: CPU-only threshold/calibration reanalysis over frozen val/test
    scores.
- `scripts/data/run_stage1_matched_horizon_reanalysis.py`
  - Module M: CPU-only matched-horizon comparison of hidden `cot_k` probes
    against text baselines seeing `prompt + first k generated CoT tokens`.
- `tests/test_stage1_threshold_reanalysis.py`
- `tests/test_stage1_matched_horizon_reanalysis.py`

## Local Validation

Commands run locally:

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
2 passed in 2.11s
```

## Review Questions For Fable-5

Please review as a methods/code blocker check, not as a paper-writing pass.

1. Module T:
   - Are the threshold policies leakage-safe as implemented?
   - Is Platt-on-validation with threshold 0.5 acceptable as the primary
     reanalysis?
   - Are `val_ba_max`, unlabeled test-score median, and oracle test threshold
     labeled with appropriate diagnostic status?
   - Is it acceptable that the same threshold policies are applied to hidden,
     selected surface, and length-only arms?

2. Module M:
   - Does the implementation actually compare equal horizons by using
     `prompt + first k generated CoT tokens` for surface baselines?
   - Is pair-complete censoring at each k implemented in the right unit?
   - Is global selection of one hidden layer and one surface family at
     anchor `k=32` leakage-safe when based only on validation data?
   - Are E1 paired delta-AUROC, E2 within-pair ranking accuracy, and Holm
     correction implemented in a defensible way?

3. Residual/E3:
   - Hidden probe directories currently expose validation/test predictions,
     but not train or OOF hidden scores.
   - The script therefore labels E3 as
     `validation_stacker_not_oof_due_missing_hidden_train_predictions`.
   - Is this conservative labeling enough, or should E3 be omitted until
     OOF/train hidden scores are exported?

4. Missing tests or code risks:
   - Please list any concrete blockers before this is synced to RunPod and run
     as a CPU reanalysis.
   - If no blocker, please say explicitly that it is OK to run, and list any
     non-blocking caveats.
