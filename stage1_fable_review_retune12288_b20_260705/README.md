# Stage1 Retune12288/B20 Fable Review Packet

Date: 2026-07-05

This packet is sanitized. It contains aggregate metrics, paths, hashes, and
row-count audits only. It intentionally excludes raw prompts, raw CoTs,
completions, and row-level prediction JSONL files.

## Review Request

Please review the post-HB Stage1 status after the provisional GPU run and the
new hidden-vs-surface delta CI audit.

Questions:

1. Does the post-GPU row audit support the interpretation that the remaining
   mismatches are high-CoT-offset coverage gaps, not extractor-level full-row
   drops?
2. Is the new hidden-minus-surface bootstrap CI procedure acceptable for
   formal reporting? It uses validation-selected hidden probes and
   validation-selected surface baselines, with pair/group resampling and
   group-internal example-id alignment.
3. Given all hidden-minus-surface deltas are negative, should Stage1 now be
   treated as a negative/control result rather than evidence for hidden-state
   superiority?
4. Are the remaining formal blockers exactly human QA, S-to-S diagnostics, and
   HT quarantine/external testing, or is any GPU rerun still required?

## Key Files

- `decision_addendum_260705.md`
- `fable_response.md`
- `docs/stage1_post_hb_retune12288_b20_gap_audit_260705.md`
- `val_fixed/val_fixed_probe_report.tsv`
- `row_audit/stage1_prediction_row_audit_summary.json`
- `row_audit/stage1_prediction_row_audit_files.tsv`
- `delta_ci/hidden_surface_delta_ci_summary.tsv`
- `delta_ci/hidden_surface_delta_ci_summary.json`
- `text_baselines/<source>/summary.tsv`
- `text_baselines/<source>/metrics.json`

## Code Commits

- cot-safety main: `0d1935a`
- Added hidden-surface delta CI orchestration and stricter paired bootstrap
  alignment.
- Added retune12288_b20 gap audit doc.

## RunPod Artifact Roots

- Main run root:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso`
- GPU archive root:
  `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20`
- R2 backup prefix:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/`

## Current Interpretation

The GPU sequence completed, and the new row audit does not show a broad
extractor-level row-drop pattern. However, the new hidden-minus-surface CI
audit is strongly unfavorable to the hidden probes: every validation-selected
hidden probe has a negative AUROC delta against the strongest
validation-selected surface baseline.

Human QA remains unannotated, so all GPU results remain provisional.
