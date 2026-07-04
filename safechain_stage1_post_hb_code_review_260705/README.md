# SafeChain Stage 1 Post-HB Code Review

This directory is a content-quiet code review snapshot for the Stage 1
post-HarmBench freeze and LOSO orchestration work.

It intentionally excludes raw prompts, CoTs, model outputs, private bucket
paths, and unpublished experiment counts.

## Files

- `analysis_reports/stage1_post_hb_code_review_packet_260705.md`
- `STAGE1_OVERNIGHT_EXECUTION_PLAN.md`
- `scripts/data/build_stage1_loso_freeze.py`
- `scripts/data/export_normalized_pairs_for_stage1.py`
- `scripts/data/sample_stage1_human_qa.py`
- `scripts/data/summarize_stage1_human_qa.py`
- `scripts/data/run_stage1_bootstrap_ci.py`
- `scripts/data/build_stage1_safe_prompt_diagnostics.py`
- `scripts/data/quarantine_stage1_external_prompts.py`
- `pipelines/run_stage1_sequence.sh`
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
- `configs/experiment/stage1_natural_pairs_1p5b.yaml`
- `configs/experiment/stage1b_natural_pairs_1p5b.yaml`
- `configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml`
- `configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml`
- `tests/test_export_normalized_pairs_for_stage1.py`
- `tests/test_stage1_loso_freeze_build.py`
- `tests/test_stage1_aux_audits.py`

## Local Verification Already Run

The focused Stage1 suite passed locally in the `cot-safety` repo:

```bash
.venv-stage1-test/bin/python -m pytest \
  tests/test_fixed_budget_gen_gen_selection.py \
  tests/test_embedding_dedup_audit.py \
  tests/test_stage1_pair_freeze_audit.py \
  tests/test_stage1_text_baselines.py \
  tests/test_stage1_surface_audit.py \
  tests/test_stage1_loso_freeze_build.py \
  tests/test_stage1_aux_audits.py
```

Result: `19 passed`.

## Review Boundary

Please review whether the code and workflow are methodologically sound. Do not
assume access to private data. If an answer depends on data values, say which
audit output should be checked rather than guessing.
