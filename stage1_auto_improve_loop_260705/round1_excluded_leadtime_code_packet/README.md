# Stage1 Excluded-Source Lead-Time Confirmation Code Packet

Date: 2026-07-05

Purpose: Fable-5 code review for the optional excluded-source lead-time
confirmation. This is not an equal-horizon rescue. Equal-horizon Stage1
iteration remains closed after A2.

## Files

- `res/stage1_excluded_source_leadtime_confirmation_prereg_plan_260705.md`
- `res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md`
- `scripts/data/run_stage1_excluded_leadtime_confirmation.py`
- `pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh`
- `tests/test_stage1_excluded_leadtime_confirmation.py`
- `review/AUTO_REVIEW.md`

## Local Validation

```text
python3 -m py_compile cot-safety/scripts/data/run_stage1_excluded_leadtime_confirmation.py
bash -n cot-safety/pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh
cot-safety/.venv-stage1-test/bin/python -m pytest \
  cot-safety/tests/test_stage1_score_pooling_reanalysis.py \
  cot-safety/tests/test_stage1_feature_pooling_reanalysis.py \
  cot-safety/tests/test_stage1_excluded_leadtime_confirmation.py
```

Result:

```text
6 passed
```

## Intended RunPod Flow If Code Review Passes

1. Sync code to RunPod.
2. Extract-minimal hidden arrays for `strongreject_full` and
   `reasoningshield` only:

```bash
cd /workspace/cot-safety
bash pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh
```

3. Run the confirmation analysis:

```bash
cd /workspace/cot-safety
/workspace/venvs/cot-natural/bin/python scripts/data/run_stage1_excluded_leadtime_confirmation.py \
  --folds-root /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/folds \
  --hidden-score-root /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20 \
  --hidden-archive-root /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives \
  --output-dir /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500 \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --code-commit <cot-safety commit> \
  --tmp-prereg-commit <tmp packet commit> \
  --n-bootstrap 500
```

Accept the run only if exit code is 0 and `n_errors=0`.

## Review Questions

Please review as Fable-5 and answer:

1. Does the code faithfully implement the reviewed excluded-source lead-time
   preregistration and config-pinning amendment?
2. Does it enforce the all-k frozen test population and minimum-power halt
   before scoring?
3. Does it avoid future-position leakage in both A1 score-pooling and A2
   feature-pooling paths?
4. Does it correctly pin Module M `char_tfidf` and the original Stage1
   single-position hidden-score recipe?
5. Are the preregistered gates implemented correctly:
   A1 pooled hidden@4-text@8 CI low >= 0; no source point estimate < -0.02;
   A2 pooled hidden@4-text@8 CI high >= 0 and point estimate >= -0.01?
6. Is this `OK_TO_RUN` on RunPod, or are there blockers?

No raw prompts, raw CoTs, generated pairs, hidden arrays, or row-level
prediction files are included in this packet.
