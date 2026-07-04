# Stage 1 Post-HB Code Review Packet

Date: 2026-07-05

This packet is content-quiet.  It intentionally excludes raw prompts, CoTs,
model outputs, private bucket paths, and unpublished experiment counts.  Review
the code and workflow shape only.

## Scope

The patch fills the missing Stage 1 audit/orchestration gaps needed before a
post-HarmBench LOSO run:

- fixed-budget LOSO freeze builder
- human-QA sampling and pass/fail summarization
- grouped bootstrap AUROC confidence intervals
- safe-prompt S->S diagnostic manifest builder
- external-prompt quarantine for exact/near overlap with the freeze
- RunPod post-HB orchestrator with CPU gates before GPU Stage1

## New Files

- `scripts/data/build_stage1_loso_freeze.py`
- `scripts/data/sample_stage1_human_qa.py`
- `scripts/data/summarize_stage1_human_qa.py`
- `scripts/data/run_stage1_bootstrap_ci.py`
- `scripts/data/build_stage1_safe_prompt_diagnostics.py`
- `scripts/data/quarantine_stage1_external_prompts.py`
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
- `tests/test_stage1_loso_freeze_build.py`
- `tests/test_stage1_aux_audits.py`

## Intended Workflow

1. Wait for the HB-only generation wrapper to exit.
2. Run a final gen/gen selection and summary.
3. Build the primary first-budget freeze using samples `[0, 100)`.
4. Run CPU-only freeze and embedding-dedup audits.
5. Build 4-source LOSO folds from the fixed freeze.
6. Sample a blinded human-QA sheet, about 50 rows/source by default.
7. Optionally freeze safe-prompt S->S diagnostic inputs when safe-prompt inputs are provided.
8. Optionally quarantine external prompts against the freeze when external prompt manifests are provided.
9. Run CPU surface/text baselines and grouped bootstrap CIs.
10. Stop before GPU Stage1 unless a passing human-QA summary is present.
11. If gates pass and `RUN_GPU_STAGE1=1`, launch the existing Stage1 sequence.

## Review Questions

- Does the LOSO freeze correctly keep HB out of non-HB train/val and use HB only as its own held-out test fold?
- Is `[0, 100)` the right primary fixed-budget contract, with `[100, 300)` treated as sensitivity/diagnostic data rather than changing the primary freeze?
- Are the human-QA gates strict enough: per-source minimum labels and unsafe-label agreement bar?
- Is group-level bootstrap by `match_family,pair_id,id` the right default unit for paired rows?
- Are safe-prompt diagnostics and external quarantine explicit enough when their source manifests are not yet specified?
- Is the RunPod orchestrator conservative enough to avoid launching GPU Stage1 before human-QA passes?

## Local Verification

The new scripts compile:

```bash
python -m py_compile \
  scripts/data/build_stage1_loso_freeze.py \
  scripts/data/sample_stage1_human_qa.py \
  scripts/data/summarize_stage1_human_qa.py \
  scripts/data/run_stage1_bootstrap_ci.py \
  scripts/data/build_stage1_safe_prompt_diagnostics.py \
  scripts/data/quarantine_stage1_external_prompts.py
```

The focused Stage1 test set passes:

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

## Known Gate

Human QA cannot be completed automatically.  The orchestrator writes
`stage1_human_qa_sheet.tsv` and exits before GPU Stage1 until
`summarize_stage1_human_qa.py` produces a passing summary.
