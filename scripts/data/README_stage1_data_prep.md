# Stage 1 Data Preparation Code Guide

This directory contains the data-preparation scripts used for Stage 1. The
current primary output is a pair of completeness-clean frozen manifests for
same-prompt `U->U` vs `U->S` probing.

For the project-level plan, see:

- `plan/stage1_plan.md`

For the current prepared-data result snapshot, see:

- `res/stage1_data_preparation_status_260702.md`

## Pipeline Structure

```text
raw source data
  -> extract rewrite seeds
  -> OpenAI safe rewrite / controlled-clean polish
  -> OpenAI unsafe-preserving paraphrase
  -> paraphrase quality stratification
  -> full A/B audit
  -> frozen A-prime/B-prime manifests
  -> completeness audit and clean manifests
  -> Stage 1 split freeze
  -> Stage 1 teacher-forcing export
```

The primary Stage 1 data should use the completeness-clean manifests:

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl
```

## Script Index

### 1. Raw Source Extraction

#### `extract_harmthoughts_rewrite_seeds.py`

Purpose:

- Convert HarmThoughts raw sentence-level annotations into grouped rewrite
  seed rows.
- Preserve source metadata such as prompt id, category, model name, word counts,
  and trace labels.

Typical role:

```text
data/harmthoughts_raw/*
  -> data or runs rewrite seed JSONL
```

#### `extract_reasoningshield_rewrite_seeds.py`

Purpose:

- Convert ReasoningShield rows into rewrite seed rows with prompt, unsafe
  reasoning, unsafe final answer, source split, and category metadata.

Typical role:

```text
data/reasoningshield_raw/*
  -> data or runs rewrite seed JSONL
```

### 2. Safe Rewrite And Polish

#### `generate_safe_rewrites_openai.py`

Purpose:

- Prepare, submit, monitor, and collect OpenAI batch jobs for unsafe-to-safe
  rewrite.
- Run controlled-clean polish passes.
- Validate generated safe reasoning/final answer length and structure.

Key config files:

```text
configs/data/unsafe_to_safe_rewrite_harmthoughts_all1018_polish_v5_controlled_clean.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round2.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round3.yaml
```

Typical usage pattern:

```bash
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> prepare
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> submit
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> status
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> collect
```

Use the exact config and subcommands recorded in the corresponding run
directory when reproducing an existing run.

#### `validate_safe_rewrite_pairs.py`

Purpose:

- Local validation for generated safe rewrite pair files.
- Checks missing fields, word counts, and obvious structural issues.

### 3. Unsafe-Side OpenAI Paraphrase

#### `pilot_unsafe_preserving_paraphrase_openai.py`

Purpose:

- Small pilot utility for testing whether OpenAI can perform
  label-preserving unsafe-side paraphrase without adding operational detail or
  washing the label.

This was used as a feasibility check before the full unsafe-side paraphrase
pipeline.

#### `repair_openai_unsafe_paraphrases.py`

Purpose:

- Main batch/repair pipeline for unsafe-preserving paraphrase.
- Produces the OpenAI-processed unsafe side used to reduce provenance mismatch
  between `U->U` and `U->S`.

Important output family:

```text
runs/openai_unsafe_paraphrase_only_v1/
```

#### `stratify_openai_paraphrase_quality.py`

Purpose:

- Build A/B/holdout quality strata from unsafe paraphrase results.
- A-tier is stricter and used for the primary A-prime dataset.
- B-tier is larger and used for sensitivity.

Important output family:

```text
runs/openai_unsafe_paraphrase_only_v1/quality_strata_v1/
```

### 4. OpenAI Audit And Frozen Manifest Export

#### `audit_openai_control_samples.py`

Purpose:

- Sample-level audit helper for safe/unsafe control samples.
- Useful before committing to a full audit batch.

#### `audit_openai_full_ab.py`

Purpose:

- Full combined A/B row audit.
- Judges unsafe paraphrase quality, safe rewrite mode, and pair alignment.
- Exports original frozen manifests under:

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/
```

Important outputs:

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/B_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/dropped_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/manifest_hashes.json
```

Current original frozen counts:

- A-prime: `1097`
- B-prime: `1460`
- dropped: `60`

### 5. Completeness Audit And Clean Manifests

#### `audit_rewrite_completeness.py`

Purpose:

- Structural completeness audit over generated safe/unsafe fields.
- Checks empty fields, too-short fields, suspicious endings, unclosed fences,
  unbalanced `<think>` tags, and related hard incompleteness signals.
- Reports diagnostics without printing text excerpts.

Run:

```bash
python3 scripts/data/audit_rewrite_completeness.py
```

Default outputs:

```text
analysis_reports/rewrite_completeness_audit_260702.json
analysis_reports/rewrite_completeness_audit_260702.md
```

#### `filter_frozen_manifests_by_completeness.py`

Purpose:

- Creates a completeness-clean copy of frozen A-prime/B-prime manifests.
- Does not mutate the original frozen manifest directory.
- Fails if `--input-dir` and `--output-dir` are the same.
- Checks duplicate and missing `pair_id` rows.
- Checks cross-file A-prime/B-prime `pair_id` overlap.

Run:

```bash
python3 scripts/data/filter_frozen_manifests_by_completeness.py --force
```

Default outputs:

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/manifest_hashes.json
```

Current clean counts:

- A-prime: `1096`
- B-prime: `1460`
- completeness dropped: `1`

### 6. Stage 1 Split Freeze

#### `freeze_stage1_prompt_splits.py`

Purpose:

- Freeze prompt-group train/validation/test splits over one or more manifest
  files.
- Grouping uses the normalized prompt hash from the manifest, so same-prompt
  safe/unsafe pairs stay in the same split.
- Intended to be run over the clean A-prime and B-prime manifests before export.

Example:

```bash
python3 scripts/data/freeze_stage1_prompt_splits.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --output-jsonl runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-summary runs/stage1_clean_prompt_splits/stage1_prompt_splits_summary.json
```

Status:

- Pending unit tests and tiny synthetic dry-run.

### 7. Stage 1 Export

#### `export_safe_rewrite_pairs_for_stage1.py`

Purpose:

- Export teacher-forcing rows for hidden-state extraction/probe training.
- In `--input-manifest` mode, this script uses the frozen manifest fields
  directly:
  - unsafe side: `unsafe_reasoning`
  - safe side: `safe_reasoning`
  - prompt: `prompt`
  - hashes: verified by default.

Important default behavior in manifest mode:

- `render_mode = reasoning_only`
- `source = paired_openai_control_manifest`
- `require_manifest_hashes = true`
- `split_manifest` required unless `--allow-unfrozen-split` is explicitly set.

Example A-prime export:

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/A_prime_reasoning_only
```

Example B-prime sensitivity export:

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/B_prime_reasoning_only
```

Status:

- Pending unit tests and tiny synthetic dry-run.

### 8. External Review Bundle

#### `build_fable5_pipeline_review_bundle.py`

Purpose:

- Build compact, redacted summaries for Fable5/Claude review.
- Does not print full unsafe text in review bundles.

Relevant review outputs:

```text
analysis_reports/fable5_pipeline_integrity_review_redacted_260702.md
analysis_reports/fable5_completeness_filter_review_260702.md
analysis_reports/fable5_completeness_filter_fix_review_260702.md
analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md
```

## Safe-Prompt Diagnostic Data

The primary paired data is ready, but `S->S` diagnostic data is still pending.

Approved safe-prompt diagnostic sources:

- XSTest safe: hard-safe over-refusal diagnostic.
- WildJailbreak `adversarial_benign`: adversarial benign control.
- OR-Bench hard benign: hard-benign over-refusal diagnostic.
- GSM8K and/or Alpaca-style benign instructions: easy benign anchor.

Trajectory generation:

- Prefer R1-1.5B natural rollouts.
- Filter with open-source judges and local completeness checks.
- Add a 150-200 row OpenAI-paraphrased `S->S` subset only as a provenance
  diagnostic.

These rows must not be mixed into the primary `U->U` vs `U->S` train/test
split.

## Reproducibility Notes

- The primary data outputs under `runs/` are run artifacts. If they are not
  tracked by git, use the SHA-256 values in
  `res/stage1_data_preparation_status_260702.md` to verify local copies.
- Do not overwrite original frozen manifests. Use the completeness-clean
  directory for Stage 1.
- Do not start CPU baselines or GPU extraction until local unit tests and a
  tiny synthetic freeze/export dry-run pass.
