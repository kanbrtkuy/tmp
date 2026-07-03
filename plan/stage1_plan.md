# Stage 1 Plan

Last updated: 2026-07-02.

This plan tracks Stage 1 of the SafeChain project: testing whether hidden
states of `DeepSeek-R1-Distill-Qwen-1.5B` contain a separable signal between
unsafe-continuation and safe-refusal/safe-redirection trajectories when the
prompt is held fixed.

## Scope

Stage 1 is not yet a steering experiment. It is a measurement gate for later
pause-token training and steering.

Primary claim wording should stay narrow:

> Given a fixed unsafe prompt, R1-1.5B hidden states under teacher forcing
> linearly separate externally generated unsafe-continuation trajectories from
> externally generated safe-refusal trajectories.

Do not claim a general on-policy unsafe-reasoning detector until natural rollout
transfer and benign-prompt specificity are shown.

## Dataset Composition

### Primary Paired Data

| Cell | Source | Count | Trajectory provenance | Role | Status |
|---|---:|---:|---|---|---|
| `U->U` | A-prime HarmThoughts + ReasoningShield | 1096 pairs | OpenAI unsafe-preserving paraphrase | Primary positive | Done |
| `U->S` | Same A-prime prompts | 1096 pairs | OpenAI safe rewrite / controlled-clean | Primary negative | Done |
| `U->U/U->S` | B-prime HarmThoughts + ReasoningShield | 1460 pairs | Same as above | Sensitivity only | Done |

Primary training and primary testing must use only paired `U->U` vs `U->S`.
`S->S` rows must not enter the primary train/test split, because that would
reintroduce prompt-type confounding.

### Safe-Prompt Diagnostic Data

These rows are required before committing Stage 2 budget, but are diagnostic
only for Stage 1.

| Cell | Recommended source | Target size | Trajectory provenance | Role | Status |
|---|---:|---:|---|---|---|
| `S->S hard-safe` | XSTest safe | all / about 250 | R1-1.5B natural rollout, judge-filtered | Over-refusal specificity | Pending |
| `S->S adversarial-benign` | WildJailbreak `adversarial_benign` | about 500 | R1-1.5B natural rollout, judge-filtered | Jailbreak-style benign control | Pending |
| `S->S hard-benign` | OR-Bench hard benign | about 500 | R1-1.5B natural rollout, judge-filtered | Hard benign control | Pending |
| `S->S easy anchor` | GSM8K and/or Alpaca-style benign instructions | about 200 | R1-1.5B natural rollout, judge-filtered | Ordinary benign floor | Pending |
| `S->S provenance matched` | 150-200 sampled from the rows above | about 150-200 | R1 rollout plus OpenAI paraphrase | Provenance diagnostic | Pending |

Fable5 recommended dropping SQuAD as a Stage 1 `S->S` source because its
context-passage QA format is too distributionally different from the chat-style
prompts used elsewhere.

## Current Status Checklist

### Primary Data Preparation

- [x] Select primary unsafe trajectory sources: HarmThoughts and ReasoningShield.
- [x] Extract rewrite seed rows from raw source formats.
- [x] Generate same-prompt safe rewrites (`U->S`) with OpenAI.
- [x] Run safe polish / controlled-clean passes.
- [x] Generate unsafe-preserving OpenAI paraphrases for the unsafe side (`U->U`).
- [x] Stratify unsafe paraphrase quality into A/B/holdout tiers.
- [x] Run full A/B quality audit.
- [x] Freeze original A-prime and B-prime manifests.
- [x] Audit structural completeness.
- [x] Produce completeness-clean manifests.
- [x] Add manifest freeze/export unit tests and pass a tiny synthetic dry-run.
- [x] Run the manifest freeze/export pytest target on the RunPod CPU node.
- [x] Freeze Stage 1 prompt-group splits on the clean manifests.
- [x] Export Stage 1 teacher-forcing JSONL from the clean manifests.
- [x] Add CPU text-baseline readiness script and tiny fixture tests.
- [x] Run CPU/text baselines.
- [x] Run CPU-only surface audits: feature audit, length matching, truncation
  curves, and cross-source transfer.
- [ ] Run GPU hidden-state extraction and probe training.

### Safe-Prompt Diagnostic Data

- [ ] Build safe-prompt source manifest: XSTest safe, WildJailbreak
  adversarial-benign, OR-Bench hard benign, GSM8K/Alpaca easy benign.
- [ ] Generate R1-1.5B natural `S->S` rollouts.
- [ ] Judge/filter `S->S` rollouts for malformed or unsafe outputs.
- [ ] Run completeness audit on `S->S` trajectories.
- [ ] Build 150-200 row OpenAI-paraphrased `S->S` provenance-matched subset.
- [ ] Add `S->S` false-positive/specificity evaluation to Stage 1 reports.

## Frozen Data Artifacts

Primary clean manifests:

- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl`
  - count: `1096`
  - sha256: `abcd42b47e61511306dc207dfc05fe4333496ced745dead3d27445d1b9af5fd8`
- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl`
  - count: `1460`
  - sha256: `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0`
- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json`
  - full provenance for the clean filter.

One A-prime row was dropped by completeness filtering because
`unsafe_reasoning` ended with an ellipsis:

- dropped count: `1`
- drop manifest:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl`
- dropped manifest sha256:
  `471c3c5828effa922ddd26e5d03260f27231ce4144b27a18c1528eb5f6224ed3`

## Related Natural-Pair Archive

The later 2026-07-03 natural-pair Stage 1 workspace snapshot is archived in
Cloudflare R2 under:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

See `docs/stage1_paired_r2_archive_260703.md` for archive structure and
restore commands. Natural-pair planning and results are tracked in
`plan/stage1_natural_pair_experiment_plan_260703.md` and
`res/stage1_natural_pair_experiment_results_260703.md`.

## CPU/Text Baseline Snapshot

Run artifacts:

- `runs/stage1_text_baselines/A_prime/`
- `runs/stage1_text_baselines/B_prime/`

Headline result:

- Prompt-only TF-IDF is at chance on A-prime and B-prime test splits
  (`balanced_accuracy = 0.5`).
- Length-only is already high:
  - A-prime test balanced accuracy: `0.953846`
  - B-prime test balanced accuracy: `0.869565`
- Reasoning-text baselines are near-perfect or perfect:
  - word TF-IDF / BoW test balanced accuracy: `1.0` on both A-prime and B-prime.
  - char TF-IDF test balanced accuracy: `1.0` on both A-prime and B-prime.
  - first-sentence-removed TF-IDF test balanced accuracy: `1.0` on both
    A-prime and B-prime.

Interpretation: prompt confounding is controlled in the paired split, but the
safe/unsafe reasoning texts remain separable by shallow surface features. Any
hidden-state probe result must therefore be compared against these baselines and
should not be claimed as evidence of latent safety monitoring without additional
controls.

## Surface Audit Snapshot

Run artifacts:

- `runs/stage1_surface_audit/A_prime/`
- `runs/stage1_surface_audit/B_prime/`

Summary report:

- `analysis_reports/stage1_surface_audit_summary_260702.md`

Key result:

- The first 4 reasoning words are already highly separable by surface baselines:
  - A-prime test BA: word TF-IDF `0.907692`, word BoW `0.953846`, char TF-IDF
    `0.923077`.
  - B-prime test BA: word TF-IDF `0.963768`, word BoW `0.971014`, char TF-IDF
    `0.978261`.
- Strict +/-10% pairwise length matching retained very few pairs:
  - A-prime train/val/test retained pairs: `82/4/3`.
  - B-prime train/val/test retained pairs: `155/2/7`.
- Cross-source word TF-IDF transfer remained near-perfect:
  - A-prime HarmThoughts -> ReasoningShield BA: `0.996111`; reverse: `1.0`.
  - B-prime HarmThoughts -> ReasoningShield BA: `0.997927`; reverse: `1.0`.

Decision: do not run GPU hidden-state extraction on A-prime/B-prime as-is. The
current data fails the informative-window gate and should be treated as a
diagnostic for rewrite/provenance confounds. Next step is A-double-prime data
design with symmetric processing, length targeting, and format/style
harmonization.

## Code Layout For Data Preparation

All Stage 1 data-preparation code lives under `scripts/data/`.

### Source Extraction

- `scripts/data/extract_harmthoughts_rewrite_seeds.py`
  - Converts HarmThoughts raw grouped sentence annotations into rewrite seed
    rows.
- `scripts/data/extract_reasoningshield_rewrite_seeds.py`
  - Converts ReasoningShield raw rows into rewrite seed rows.

### Safe Rewrite Pipeline

- `scripts/data/generate_safe_rewrites_openai.py`
  - Main OpenAI batch pipeline for unsafe-to-safe rewrite and controlled-clean
    polish.
  - Uses configs under `configs/data/unsafe_to_safe_rewrite_*.yaml`.
- `scripts/data/validate_safe_rewrite_pairs.py`
  - Local validation of generated safe rewrite pairs.

### Unsafe-Side Paraphrase And Quality

- `scripts/data/pilot_unsafe_preserving_paraphrase_openai.py`
  - Pilot utility for unsafe-preserving paraphrase feasibility.
- `scripts/data/repair_openai_unsafe_paraphrases.py`
  - Full unsafe-preserving paraphrase batch/repair pipeline.
- `scripts/data/stratify_openai_paraphrase_quality.py`
  - Builds A/B/holdout quality tiers from paraphrase results.
- `scripts/data/audit_openai_control_samples.py`
  - Sample-level audit utility for safe/unsafe control rows.
- `scripts/data/audit_openai_full_ab.py`
  - Full A/B combined quality audit and original frozen manifest export.

### Manifest Integrity And Stage 1 Export

- `scripts/data/audit_rewrite_completeness.py`
  - Structural completeness audit for generated reasoning/final fields.
- `scripts/data/filter_frozen_manifests_by_completeness.py`
  - Writes completeness-clean A-prime/B-prime manifests without mutating the
    original frozen manifests.
- `scripts/data/freeze_stage1_prompt_splits.py`
  - Freezes prompt-group train/validation/test splits over one or more clean
    manifests.
- `scripts/data/export_safe_rewrite_pairs_for_stage1.py`
  - Exports teacher-forcing rows for Stage 1 hidden-state extraction.
  - In manifest mode it uses:
    - unsafe side: `manifest.unsafe_reasoning`
    - safe side: `manifest.safe_reasoning`
    - default render mode: `reasoning_only`
- `scripts/data/run_stage1_text_baselines.py`
  - Runs Stage 1 CPU surface baselines from exported
    `normalized/{train,val,test}.jsonl` rows.
  - Supported by default: length-only, prompt-only TF-IDF, word TF-IDF,
    word BoW, char n-gram TF-IDF, and first-sentence-removed TF-IDF.
  - The original-unsafe vs OpenAI-paraphrased provenance classifier is skipped
    unless a reviewed pair-id-aligned original unsafe source is provided.

### Review / Audit Bundle

- `scripts/data/build_fable5_pipeline_review_bundle.py`
  - Builds compact bundles for external Fable5/Claude review.
- Fable5 reviews used for the current plan:
  - `analysis_reports/fable5_pipeline_integrity_review_redacted_260702.md`
  - `analysis_reports/fable5_completeness_filter_review_260702.md`
  - `analysis_reports/fable5_completeness_filter_fix_review_260702.md`
  - `analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md`

## Basic Usage

From the repository root `cot-safety/`.

Run completeness audit:

```bash
python3 scripts/data/audit_rewrite_completeness.py
```

Rebuild completeness-clean manifests:

```bash
python3 scripts/data/filter_frozen_manifests_by_completeness.py --force
```

Freeze prompt-group splits after the clean manifests are accepted:

```bash
python3 scripts/data/freeze_stage1_prompt_splits.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --output-jsonl runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --summary-json runs/stage1_clean_prompt_splits/stage1_prompt_splits_summary.json
```

Export Stage 1 rows for A-prime:

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/A_prime_reasoning_only
```

Export Stage 1 rows for B-prime sensitivity:

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/B_prime_reasoning_only
```

Execution order is fixed: first run the freeze/export tests and tiny synthetic
dry-run, then freeze real prompt-group splits on the clean manifests, then
export real Stage 1 `reasoning_only` rows. Do not start CPU baselines or GPU
extraction until the real split/export artifacts exist.

## Stage 1 Gate Criteria

Before proceeding to Stage 2:

- A-prime hidden-state probe beats text/surface baselines by the pre-registered
  margin.
- Result is stable across at least three split seeds.
- Prompt-only probe is at or near chance.
- Shuffled-label probe is at or near chance.
- B-prime sensitivity agrees directionally with A-prime.
- Source-model/provenance controls do not explain the effect.
- `S->S` diagnostic false-positive rate is acceptable at a threshold frozen on
  paired validation data.
- Natural rollout transfer is checked before committing pause-token training
  budget.
