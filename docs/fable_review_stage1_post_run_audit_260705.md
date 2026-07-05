# Stage1 Post-Run Audit Packet for Fable Review

Date: 2026-07-05

This packet contains aggregate counts, artifact paths, and process notes only. It intentionally excludes raw prompts, raw completions, and unreleased experimental examples.

Handoff rule: if a direct Fable review channel is blocked by platform policy, follow `docs/fable_review_handoff_protocol_260705.md`: push the sanitized packet to the GitHub tmp review repo and ask Fable to review that repo link/path.

## Review Request

We need a strict external review of the current Stage1 status after an overnight automatic run. Please identify which items are formal blockers before Stage1 results can be claimed, which items can be treated as diagnostic/provisional, and the minimal remediation order.

The core concern is that GPU Stage1 ran automatically before human QA was completed, per user override. We need to know whether the current artifacts are enough to proceed after a human QA pass, or whether any data/code/probe reruns are required.

## Artifact Roots

- RunPod repo/workdir: `/workspace/cot-safety`
- Main run root: `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso`
- Archive root: `/workspace/stage1-results/stage1_post_hb_260705_after_hb_n100_loso`
- Freeze dir: `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099`
- Local/GitHub code commit for batch tuning: `a8cdf34`
- Local/GitHub code commit for prediction row audit: `5432326`
- Local/GitHub code commit for Fable tmp handoff protocol: `a82163f`
- Local/GitHub code commit for LOSO word-budget gate: `21cada5`

## Data Status

### HB Generation Snapshot

Artifact:

- `hb_raw_snapshot/hb_raw_snapshot_summary.json`
- `hb_raw_snapshot/run_files/selection_gen_gen_summary.json`

Selected pairs at final raw snapshot:

| Source | selected pairs |
|---|---:|
| HarmBench | 167 |
| StrongReject | 277 |
| WildJailbreak | 2043 |

The HB generator had run up to the fixed budget cap. 33 HB prompts still had no pair at 300 attempts.

### Fixed-Budget N=100 Re-Selection

Artifact:

- `fixed_budget_samples_000_099/selection_gen_gen_summary.json`

Selected pairs after fixed-budget re-selection:

| Source | selected pairs | dropped prompts | yield |
|---|---:|---:|---:|
| HarmBench | 152 | 48 | 0.760 |
| StrongReject | 277 | 36 | 0.885 |
| WildJailbreak | 2019 | 981 | 0.673 |

ReasoningShield pairs were restored from Cloudflare/R2 and included later in freeze:

| Source | selected pairs |
|---|---:|
| ReasoningShield | 335 |

## Freeze Audit

Artifact:

- `freeze_audit_fixed_budget_samples_000_099/audit_summary.json`

Freeze audit summary:

| Field | Value |
|---|---:|
| input pairs | 2783 |
| kept main pairs | 2783 |
| dropped pairs | 0 |
| duplicate edges | 0 |

Kept pairs by source:

| Source | kept pairs | readiness |
|---|---:|---|
| HarmBench | 152 | min met, below ideal |
| ReasoningShield | 335 | ideal met |
| StrongReject | 277 | min met, slightly below ideal |
| WildJailBreak | 2019 | ideal met |
| HarmThoughts | 0 | external/quarantine only, not included |

Token-window availability:

| k | available pairs |
|---:|---:|
| 4 | 2783 |
| 8 | 2783 |
| 16 | 2783 |
| 32 | 2783 |
| 64 | 2783 |
| 128 | 2778 |
| 256 | 2691 |
| 512 | 962 |
| 1024 | 26 |

Length caliper counts by source:

| Caliper | HarmBench | ReasoningShield | StrongReject | WildJailbreak |
|---:|---:|---:|---:|---:|
| 0.8 | 22 | 91 | 61 | 373 |
| 0.9 | 7 | 41 | 33 | 179 |

## Embedding Dedup Audit

Artifacts:

- `embedding_dedup_fixed_budget_samples_000_099/embedding_dedup_summary.json`
- `embedding_dedup_fixed_budget_samples_000_099/top_cross_source_neighbors.jsonl`
- `embedding_dedup_fixed_budget_samples_000_099/top_near_band_cross_source_neighbors.jsonl`

Pairs by source:

| Source | pairs |
|---|---:|
| HarmBench | 152 |
| ReasoningShield | 335 |
| StrongReject | 277 |
| WildJailbreak | 2019 |

The cross-source near-band count for cosine similarity in `[0.8, 0.9)` is 0. Please verify whether this is a sufficient dedup/no-leakage sanity check or if additional thresholds/reporting are needed.

## Human QA Gate

Artifacts:

- `human_qa_fixed_budget_samples_000_099/stage1_human_qa_manifest.jsonl`
- `human_qa_fixed_budget_samples_000_099/stage1_human_qa_sample_summary.json`
- `human_qa_fixed_budget_samples_000_099/stage1_human_qa_sheet.tsv`
- `human_qa_fixed_budget_samples_000_099/human_qa_gate_bypassed.json`

Sampling packet exists:

| Source | safe rows | unsafe rows | total rows |
|---|---:|---:|---:|
| HarmBench | 30 | 30 | 60 |
| ReasoningShield | 30 | 30 | 60 |
| StrongReject | 30 | 30 | 60 |
| WildJailbreak | 30 | 30 | 60 |

Important deviation:

- The QA sheet has not been human-annotated.
- The run contains a bypass marker with reason: `user_requested_overnight_automatic_provisional_stage1_before_human_QA`.
- Current GPU Stage1 should therefore be treated as provisional unless Fable says otherwise.

## LOSO Freeze

Artifact:

- `loso_freeze_fixed_budget_samples_000_099/stage1_loso_freeze_summary.json`

Kept pairs by source after freeze:

| Source | keep pairs |
|---|---:|
| HarmBench | 152 |
| ReasoningShield | 335 |
| StrongReject | 277 |
| WildJailbreak | 2019 |

The freeze summary enforces post-freeze source counts on `keep_pairs_by_source`, not only on raw inputs.

Fold layout:

| Held-out source | Train pairs | Val pairs | Test pairs | Notes |
|---|---:|---:|---:|---|
| ReasoningShield | 879 | 98 | 335 | train/val from StrongReject + capped WildJailbreak |
| StrongReject | 931 | 104 | 277 | train/val from ReasoningShield + capped WildJailbreak |
| WildJailbreak | 550 | 62 | 2019 | train/val from ReasoningShield + StrongReject; full WJB as test |
| HarmBench | 1180 | 132 | 152 | train/val from ReasoningShield + StrongReject + capped WildJailbreak |

WildJailbreak is capped at 700 pairs for train/val folds where it appears in train/val. Hashes are recorded for normalized train/val/test/all and fold manifests.

Please judge whether this LOSO layout is acceptable given HarmBench only has 152 kept pairs and appears only as a held-out test family, not a train/val family.

## CPU / Surface Baselines

Artifacts exist for each of the four held-out sources under:

- `surface_audit/<source>/metrics.json`
- `surface_audit/<source>/feature_audit_word_tfidf.tsv`
- `surface_audit/<source>/feature_audit_char_tfidf.tsv`
- `surface_audit/<source>/length_analysis.json`
- `surface_audit/<source>/length_matched_baselines.tsv`
- `surface_audit/<source>/truncation_curves.tsv`
- `surface_audit/<source>/truncation_bootstrap_ci.tsv`
- `surface_audit/<source>/cross_source_transfer.tsv`

Please verify whether this baseline package satisfies the requested Stage1 controls:

- length-only baselines
- word/char TF-IDF or BoW baselines
- token-window truncation curves
- length-matched controls
- validation-selected reporting instead of test-max reporting
- probe-minus-surface delta confidence intervals

## Bootstrap CIs

Artifacts exist for each held-out source under:

- `bootstrap_ci/<source>/stage1_bootstrap_ci_summary.json`
- `bootstrap_ci/<source>/stage1_bootstrap_ci_summary.tsv`

Please verify whether these CIs are sufficient for formal reporting, especially whether they compare hidden-state probes against the appropriate surface baselines.

## GPU Hidden Probe Runs

The overnight GPU sequence completed all four LOSO held-out sources for both Stage1 and Stage1b. Archive counts were verified after a manual archive repair.

Verified archive counts:

| Run | linear metrics | multilayer metrics |
|---|---:|---:|
| stage1 / ReasoningShield | 361 / 361 | 12 / 12 |
| stage1b / ReasoningShield | 171 / 171 | 14 / 14 |
| stage1 / StrongReject | 361 / 361 | 12 / 12 |
| stage1b / StrongReject | 171 / 171 | 14 / 14 |
| stage1 / WildJailbreak | 361 / 361 | 12 / 12 |
| stage1b / WildJailbreak | 171 / 171 | 14 / 14 |
| stage1 / HarmBench | 361 / 361 | 12 / 12 |
| stage1b / HarmBench | 171 / 171 | 14 / 14 |

Important caveats:

- The GPU sequence was launched after the human QA bypass marker, so results are provisional.
- The final sequence log did not end with `ALL_STAGE1_SEQUENCE_DONE`; it ended after final HarmBench Stage1b metrics with an archive-side error because `rsync` was unavailable and the old running script hit a quoting EOF. The final HarmBench Stage1b artifacts were manually copied into the archive and then counts were verified.
- A durable prediction-row audit was added and run after this packet was first drafted:
  - Script: `scripts/data/audit_stage1_prediction_rows.py`
  - RunPod output: `prediction_row_audit/stage1_prediction_row_audit_summary.json`
  - Result: `passes=false`, `n_prediction_files=4464`, `n_mismatch_files=1230`.
- Most Stage1-only mismatches are high-CoT-offset coverage differences, e.g. `cot_96`/`cot_128`, which may be expected if a row does not have enough available CoT tokens at that offset and should be reported with per-position n.
- However, one extraction-level row issue is confirmed and should be treated as a formal blocker:
  - The same WildJailbreak unsafe row is missing from StrongReject-fold validation predictions and WildJailbreak-fold test predictions.
  - Missing row hash: `1bedd82f59c0f070`.
  - The prepared row has an extreme unsafe reasoning length: 6305 words.
  - The extractor drops rows whose rendered sequence exceeds `extract_max_length=4096` as `too_long`.
  - Because the run used `--skip_data_prep`, the configured `max_reasoning_words=1600` did not truncate the frozen normalized row before extraction.

## Batch / Resource Tuning

The user correctly objected that GPU VRAM was underutilized. Before the final HarmBench runs, the 8B A100 extraction batch was increased from 16 to 24 in:

- `configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml`
- `configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml`

This change was committed and pushed as `a8cdf34`. HarmBench Stage1/Stage1b ran with `--extract_batch_size 24` / `--batch_size 24` and completed without OOM.

## Not Yet Run / Not Yet Certified

The following are not yet formalized as completed:

- Human QA annotation and pass/fail summary.
- Formal remediation for the confirmed too-long WildJailbreak unsafe row, followed by rerun of affected Stage1/Stage1b folds or a full frozen rerun.
- S-to-S safe-prompt diagnostic run.
- HarmThoughts quarantine/external held-out diagnostic after Stage1.
- Fable review of this final post-run state.
- Certification that CPU/surface baseline artifacts satisfy all planned controls.
- Certification that bootstrap CIs are validation-selected and compare the intended quantities.

## Post-Packet Remediation Update

After the first post-run packet, we added a fail-closed LOSO freeze word-budget gate:

- Code: `scripts/data/build_stage1_loso_freeze.py`
- Pipeline call: `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
- Test: `tests/test_stage1_loso_freeze_build.py`
- Commit: `21cada5`

The gate drops an entire pair if any row exceeds the configured Stage1 text budgets:

- `STAGE1_MAX_PROMPT_WORDS=800`
- `STAGE1_MAX_REASONING_WORDS=1600`
- `STAGE1_MAX_FINAL_WORDS=800`

A non-destructive rebuild probe on RunPod wrote:

- `loso_freeze_fixed_budget_samples_000_099_wordcap_probe/stage1_loso_freeze_summary.json`

Probe counts:

| Source | keep pairs after word-budget gate |
|---|---:|
| HarmBench | 151 |
| ReasoningShield | 304 |
| StrongReject | 271 |
| WildJailbreak | 1953 |

Drop reason counts:

| reason | pairs |
|---|---:|
| `final_answer_words_gt_cap` | 97 |
| `reasoning_words_gt_cap` | 7 |

This should eliminate the confirmed too-long WildJailbreak unsafe row from the next formal freeze. It also reduces WJB below its previous ideal count of 2000, though it remains above the minimum floor. Please review whether this fail-closed word-budget gate is preferable to extractor-side truncation, and whether these post-gate source counts are still acceptable for formal Stage1.

## Questions for Fable

1. Is the current Stage1 GPU run usable only as provisional until human QA passes, or can it be conditionally accepted after a retrospective human QA pass on the frozen packet?
2. Is the post-freeze source gate sufficient now that it checks `keep_pairs_by_source` after freeze, with HarmBench at 152?
3. Is the LOSO layout acceptable when HarmBench is held out only as test and has 152 pairs, while WildJailbreak is capped only in train/val?
4. Does the existing CPU/surface audit artifact list satisfy the Stage1 baseline/control plan, or what exact missing baseline(s) should be added?
5. Are the bootstrap CI artifacts sufficient for probe-minus-surface claims, or do we need a new delta CI pass?
6. Should the hidden-extraction row-count discrepancy block formal claims until rerun, even if metrics files are complete?
7. What is the recommended minimal remediation order before formal Stage1 claims?
