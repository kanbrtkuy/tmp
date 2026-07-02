# Stage 1 Data Preparation Status

Date: 2026-07-02.

This report records the current data-preparation state for Stage 1. It is a
result snapshot, not an experiment result.

## Summary

The primary paired unsafe-prompt dataset is ready for Stage 1 split freezing and
export.

Current primary data:

- A-prime primary: `1096` same-prompt pairs.
- B-prime sensitivity: `1460` same-prompt pairs.
- Pair cells:
  - `U->U`: unsafe prompt + OpenAI unsafe-preserving paraphrased reasoning.
  - `U->S`: the same unsafe prompt + OpenAI safe rewritten reasoning.

OpenAI API work for the primary paired data is complete:

- safe rewrite: complete.
- controlled-clean safe polish: complete.
- unsafe-preserving paraphrase: complete.
- full A/B row audit: complete.
- manifest freeze and completeness-clean pass: complete.

Safe-prompt diagnostic data (`S->S`) is not yet prepared.

## Primary Clean Manifests

The recommended Stage 1 inputs are the completeness-clean manifests:

| Manifest | Count | SHA-256 |
|---|---:|---|
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl` | 1096 | `abcd42b47e61511306dc207dfc05fe4333496ced745dead3d27445d1b9af5fd8` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl` | 1460 | `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0` |

Completeness-clean provenance:

- Summary:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json`
- Compatibility hash summary:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/manifest_hashes.json`
- Completeness filter script:
  `scripts/data/filter_frozen_manifests_by_completeness.py`
- Completeness rule source:
  `scripts/data/audit_rewrite_completeness.py:any_strong_incomplete`

## Original Frozen Manifests

The original full A/B audit exported these manifests before completeness
filtering:

| Manifest | Count | SHA-256 |
|---|---:|---|
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/A_prime_manifest.jsonl` | 1097 | `86b654c9b6bc8e7bb899071ff8ea522637759c3f99878dc51d69f2fd66753e47` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/B_prime_manifest.jsonl` | 1460 | `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/dropped_manifest.jsonl` | 60 | `4732b840250c0e5cc48c61d39ea64186ac4bbab977119ec3a55ebcef5b581d74` |

Original A/B audit provenance is stored in:

- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/manifest_hashes.json`

## Completeness Filtering Result

Completeness filtering removed exactly one A-prime row:

- Dropped count: `1`.
- Dropped field: `unsafe_reasoning`.
- Reason: `strong_completeness_incomplete`, specifically `ends_with_ellipsis`.
- Dropped manifest:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl`
- Dropped manifest SHA-256:
  `471c3c5828effa922ddd26e5d03260f27231ce4144b27a18c1528eb5f6224ed3`

Post-filter verification:

- clean A-prime hard incomplete counts:
  - `safe_reasoning`: `0`
  - `safe_final_answer`: `0`
  - `unsafe_reasoning`: `0`
- clean B-prime hard incomplete counts:
  - `safe_reasoning`: `0`
  - `safe_final_answer`: `0`
  - `unsafe_reasoning`: `0`
- Idempotency check:
  - Running the completeness filter on the clean manifests produced `0` drops.
  - A-prime and B-prime input/output hashes were unchanged in the idempotency
    check.

Fable5 reviewed the completeness filter and approved proceeding to unit tests
and tiny synthetic dry-run:

- `analysis_reports/fable5_completeness_filter_review_260702.md`
- `analysis_reports/fable5_completeness_filter_fix_review_260702.md`

## Data Design Decision

Fable5 approved the following Stage 1 composition:

| Cell | Source | Size | Role | Primary train/test |
|---|---:|---:|---|---|
| `U->U` | A-prime HarmThoughts + ReasoningShield | 1096 | Primary positive | yes |
| `U->S` | A-prime same prompts | 1096 | Primary negative | yes |
| `U->U/U->S` | B-prime | 1460 pairs | Sensitivity only | no headline |
| `S->S` | XSTest safe | about 250 | Hard-safe diagnostic | no |
| `S->S` | WildJailbreak `adversarial_benign` | about 500 | Adversarial-benign diagnostic | no |
| `S->S` | OR-Bench hard benign | about 500 | Hard-benign diagnostic | no |
| `S->S` | GSM8K/Alpaca-style benign | about 200 | Easy benign anchor | no |
| `S->S` paraphrased | 150-200 sampled safe-prompt rows | 150-200 | Provenance diagnostic | no |

Detailed review:

- `analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md`

Important constraints:

- `S->S` rows must not enter the primary train/test set.
- Stage 1 headline should be A-prime.
- B-prime should be reported as sensitivity.
- Thresholds for `S->S` false-positive checks should be frozen on paired
  validation data, not calibrated on `S->S`.
- `S->S` natural trajectories should be generated by R1-1.5B where possible.
- Use a small OpenAI-paraphrased `S->S` subset only as a provenance diagnostic.

## Current Next Steps

Completed local test work:

- Added `tests/test_stage1_manifest_freeze_export.py`.
- `python3 -m py_compile` passed for the new test and the two freeze/export
  scripts.
- A temporary synthetic CLI dry-run passed: 3 prompt groups, 4 pairs, 4
  exported rows, and prompt-hash mismatch rejection.
- `python3 -m pytest cot-safety/tests/test_stage1_manifest_freeze_export.py`
  could not run on this machine because `pytest` is not installed.

No CPU baselines or GPU runs should start until these remaining local checks
and data exports pass:

1. Run the pytest target in an environment with dev dependencies installed.
2. Prompt-group split freeze over the clean A-prime/B-prime manifests.
3. Stage 1 export using `reasoning_only` manifest mode.
4. Text/surface baselines and prompt-only controls.

After those pass, the first GPU-facing task is hidden-state extraction/probe
training on clean A-prime.

## Code Index

See the code-level guide:

- `scripts/data/README_stage1_data_prep.md`

The main scripts used for the prepared primary data are:

- `scripts/data/extract_harmthoughts_rewrite_seeds.py`
- `scripts/data/extract_reasoningshield_rewrite_seeds.py`
- `scripts/data/generate_safe_rewrites_openai.py`
- `scripts/data/validate_safe_rewrite_pairs.py`
- `scripts/data/pilot_unsafe_preserving_paraphrase_openai.py`
- `scripts/data/repair_openai_unsafe_paraphrases.py`
- `scripts/data/stratify_openai_paraphrase_quality.py`
- `scripts/data/audit_openai_control_samples.py`
- `scripts/data/audit_openai_full_ab.py`
- `scripts/data/audit_rewrite_completeness.py`
- `scripts/data/filter_frozen_manifests_by_completeness.py`
- `scripts/data/freeze_stage1_prompt_splits.py`
- `scripts/data/export_safe_rewrite_pairs_for_stage1.py`
