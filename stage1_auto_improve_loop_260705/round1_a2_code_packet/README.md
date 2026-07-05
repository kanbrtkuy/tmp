# Stage1 Auto-Improve Round 1 A2 Code Packet

Purpose: code review before running Fable-5 A2.

This packet contains code only:

- `scripts/run_stage1_feature_pooling_reanalysis.py`
- `scripts/run_stage1_score_pooling_reanalysis.py` (imported helper module from A1)
- `tests/test_stage1_feature_pooling_reanalysis.py`
- `review/AUTO_REVIEW.md` (local review log with the A1 result-review decision)

It intentionally does not include raw prompts, raw CoTs, hidden activation
arrays, generated pair files, or prediction JSONL rows.

## A2 Protocol Implemented

- Hidden arm: refit linear probes on unweighted mean-pooled layer-28 hidden
  vectors over `cot_j` positions where `j <= k`.
- Surface arm: reuse unchanged matched-horizon `char_tfidf` text@k scores from
  Module M.
- k grid: `4,8,16,32,64`.
- Holm family: `8,16,32,64`; k=4 remains exploratory/diagnostic.
- Classifier: `StandardScaler + LogisticRegression(class_weight="balanced")`,
  fit on train split only.
- Validation split is used for reporting and for per-source z-normalization
  before pooled cross-source rows; test statistics are not used for fitting or
  normalization.
- Layer is fixed to 28; the script rejects other layers for this primary
  protocol.
- Remote provenance is fixed with `--code-commit`, because RunPod
  `/workspace/cot-safety` is not a git checkout.

## Review Requests

Please check for true blockers before run:

1. No future-position leakage: for target k, pooling set must be exactly
   `{j in k_grid: j <= k}` and position-name lookup must not be off by one.
2. No test leakage into fitting, validation z-normalization, or protocol
   choices.
3. Same evaluation population as A1/Module M: hidden/test rows are aligned to
   frozen surface rows, and the script raises if surface rows or pair-complete
   pairs would be dropped.
4. Cross-source pooled rows z-normalize both hidden and surface arms per source
   using validation aligned rows before concatenation.
5. Hidden loader touches only activation arrays and metadata ids/labels, not
   raw text fields.
6. Output JSON/TSV records preregistration, success preview, split diagnostics,
   and non-null code commit provenance.

Local validation:

```text
python3 -m py_compile scripts/data/run_stage1_feature_pooling_reanalysis.py tests/test_stage1_feature_pooling_reanalysis.py
pytest tests/test_stage1_feature_pooling_reanalysis.py
2 passed
```
