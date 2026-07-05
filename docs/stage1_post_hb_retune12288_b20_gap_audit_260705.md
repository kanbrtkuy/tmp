# Stage1 Post-HB Gap Audit: retune12288_b20

Date: 2026-07-05

This audit is a sanitized status packet. It records aggregate results and
artifact paths only; it does not include raw prompts, CoTs, or completions.

## Roots

- RunPod repo: `/workspace/cot-safety`
- Primary Stage1 run root: `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso`
- Frozen LOSO dir: `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099`
- GPU archive root: `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20`
- R2 hidden prefix: `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/hidden_archives/stage1_post_hb_260705_retune12288_b20/`

## Completed

- HB raw snapshot exists under `hb_raw_snapshot/`.
- Fixed-budget N=100 selection exists under `fixed_budget_samples_000_099/`.
- Pair freeze audit exists under `freeze_audit_fixed_budget_samples_000_099/`.
- Embedding/TF-IDF dedup audit exists under `embedding_dedup_fixed_budget_samples_000_099/`.
- Human QA sampling sheet exists under `human_qa_fixed_budget_samples_000_099/`.
- Frozen LOSO manifests exist under `loso_freeze_fixed_budget_samples_000_099/folds/`.
- Surface audit outputs exist under `surface_audit/<source>/`.
- Text baseline predictions were added under `text_baseline_predictions_retune12288_b20/<source>/`.
- GPU Stage1/Stage1b completed for RS, SR, WJB, and HB.
- Validation-selected probe report was added under `val_fixed_probe_report_retune12288_b20/`.
- Post-GPU prediction row audit was added under `post_gpu_prediction_row_audit_retune12288_b20/`.
- Hidden-minus-surface bootstrap delta CI was added under `hidden_surface_delta_ci_retune12288_b20/`.

## Machine Audit Results

### GPU Completion

Run log ended with:

`ALL_STAGE1_SEQUENCE_DONE 2026-07-05T04:21:51+00:00`

### Prediction Row Audit

Artifact:

- `post_gpu_prediction_row_audit_retune12288_b20/stage1_prediction_row_audit_summary.json`
- `post_gpu_prediction_row_audit_retune12288_b20/stage1_prediction_row_audit_files.tsv`

Summary:

- Prediction files audited: 4464
- Mismatch files: 133
- All mismatches are in Stage1 linear high-offset positions (`cot_96`/`cot_128`).
- Stage1b and multilayer selected runs do not show this mismatch pattern.
- This is consistent with high-CoT-offset coverage gaps rather than extractor-level full-row drops.

Groups with mismatches:

| Run group | Split | Mismatch files | Expected rows | Prediction row counts |
|---|---:|---:|---:|---|
| HB Stage1 linear | val | 38 | 264 | 262, 263, 264 |
| RS Stage1 linear | test | 19 | 670 | 669, 670 |
| SR Stage1 linear | test | 38 | 554 | 553, 554 |
| WJB Stage1 linear | val | 19 | 124 | 123, 124 |
| WJB Stage1 linear | test | 19 | 4038 | 4034, 4038 |

### Validation-Selected Probe Report

Artifact:

- `val_fixed_probe_report_retune12288_b20/val_fixed_probe_report.tsv`
- `val_fixed_probe_report_retune12288_b20/val_fixed_probe_report.json`
- `val_fixed_probe_report_retune12288_b20/val_fixed_probe_report_ranked.tsv`

The report selects by validation AUROC only. The reporting script was updated
to preserve `layer_combine` and `layers` for multilayer rows.

### Hidden-Minus-Surface Delta CI

Artifact:

- `hidden_surface_delta_ci_retune12288_b20/hidden_surface_delta_ci_summary.json`
- `hidden_surface_delta_ci_retune12288_b20/hidden_surface_delta_ci_summary.tsv`

The delta CI script compares validation-selected hidden probes against the
validation-selected surface baseline for each held-out source. Bootstrap
resampling is grouped by `match_family,pair_id,id`, with group-internal
alignment by example id.

Result summary:

- Items: 16
- Errors: 0
- All 16 hidden-minus-surface AUROC deltas are negative.
- Surface baselines selected by validation AUROC are `word_bow` for HB/RS/SR
  and `char_tfidf` for WJB.

Examples:

| Run | Kind | Surface | Hidden minus surface AUROC | 95% CI |
|---|---|---|---:|---|
| HB Stage1 | linear | word_bow | -0.1259 | [-0.1677, -0.0853] |
| RS Stage1 | linear | word_bow | -0.2141 | [-0.2492, -0.1814] |
| SR Stage1 | linear | word_bow | -0.1229 | [-0.1552, -0.0944] |
| WJB Stage1 | linear | char_tfidf | -0.0927 | [-0.1031, -0.0816] |

This is a major claim risk: the current hidden probe is not outperforming the
strongest validation-selected surface baseline.

## Remaining Blockers

1. Human QA is still not annotated.
   - Existing sheet: `human_qa_fixed_budget_samples_000_099/stage1_human_qa_sheet.tsv`
   - A bypass marker exists because the overnight GPU run was explicitly
     allowed before QA.

2. Formal Stage1 claim is currently weak.
   - Hidden probes lose to validation-selected surface baselines in the new
     paired delta CI.
   - Any paper claim should either be revised or supported by additional
     controls/alternative probe settings.

3. S-to-S safe-prompt diagnostics are not run.
   - No XSTest safe / WJB benign / OR-Bench benign / GSM8K-Alpaca benign
     diagnostic outputs were found in this run root.

4. HT external quarantine/test is not run.
   - No HT quarantine output was found in this run root.
   - HT should remain external confirmatory until quarantine is done.

5. Some earlier hidden arrays were not preserved.
   - Lost cached hidden arrays: RS Stage1, RS Stage1b, SR Stage1.
   - They can be regenerated from the frozen splits/configs if needed.
   - Later hidden arrays were backed up to R2 before local cleanup.

## Immediate Next Order

1. Have the user complete the human QA sheet, then run `summarize_stage1_human_qa.py`.
2. Prepare a Fable/tmp review packet including this audit, the row audit, and
   the hidden-minus-surface delta CI.
3. Decide whether Stage1 is now a negative/control result or whether a new
   probe/control design is needed before formal claims.
4. Run S-to-S safe-prompt diagnostics once input prompt files are fixed.
5. Run HT quarantine and only then HT external test.
