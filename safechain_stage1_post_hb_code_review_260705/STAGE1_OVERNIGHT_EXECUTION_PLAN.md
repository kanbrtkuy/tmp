# Stage 1 Overnight Execution Plan

This is a content-quiet plan/code review note. It intentionally omits raw
prompts, CoTs, private bucket paths, and unpublished count tables.

## Intended Primary Policy

- Use fixed-budget first-N generated/generated rollouts as the primary Stage 1
  dataset.
- Primary budget: `N=100`, implemented as `sample_idx in [0, 100)`.
- Earlier `N=50` remains an original-preregistration/sensitivity artifact, not
  the primary LOSO claim dataset.
- Later HB-only samples beyond the first-N window are diagnostic/sensitivity
  only and must not be mixed into the primary LOSO freeze.

## Post-HB Sequence

1. Wait for the HB generation wrapper to finish.
2. Snapshot the raw post-HB state before any new fixed-budget selection:
   - current `selection_gen_gen_summary.json`
   - candidates, judged candidates, selected pair files, normalized pair files
   - active manifest summaries for final HB round / remaining prompt state
   - source-expansion and HB-only logs
   - hashes, sizes, and line counts in `hb_raw_snapshot_summary.json`
3. Run final fixed-budget re-selection at `N=100`.
4. Combine the selected SR/HB/WJB primary file with an explicit RS extra file.
5. Fail closed on LOSO source floors both pre-freeze and post-freeze:
   - pre-freeze raw pair counts
   - post-freeze `keep_pairs_by_source` in `stage1_loso_freeze_summary.json`
6. Run CPU freeze/audit steps:
   - pair freeze audit
   - TF-IDF embedding dedup audit and top-k nearest neighbor packet
   - frozen LOSO fold builder
   - human QA sheet sampling
   - safe-prompt diagnostics only if explicit safe-prompt inputs are supplied
   - HT/external quarantine only if explicit external inputs are supplied
   - text/surface baselines and bootstrap CI on the same frozen folds
7. Formal GPU Stage 1 should normally require a passing human QA summary.

## Overnight Exception

The user requested that the machine should not sit idle overnight, even though
human QA will not be completed until later.

The orchestrator therefore has an explicit opt-in:

- `ALLOW_UNREVIEWED_GPU_STAGE1=1`
- writes `human_qa_gate_bypassed.json`
- labels the GPU outputs as provisional/debug-only
- does not mark human QA as passed

The intended interpretation is:

- These overnight GPU results may help debug throughput, memory, launchers, and
  early signal quality.
- They are not formal LOSO evidence until human QA is completed and passes.

## Frozen LOSO Folds

The freeze builder is expected to create these heldout folds:

- `holdout_reasoningshield`
  - train/val: SR + capped WJB
  - test: RS
- `holdout_strongreject_full`
  - train/val: RS + capped WJB
  - test: SR
- `holdout_wildjailbreak_vanilla_harmful`
  - train/val: RS + SR
  - test: full WJB
- `holdout_harmbench_standard`
  - train/val: RS + SR + capped WJB
  - test: HB

HB is excluded from train/val for F1-F3 and appears as a test-only family
except in its own holdout fold.

## GPU Launcher

`pipelines/run_stage1_sequence.sh` now expects `STAGE1_FREEZE_DIR`.
For each fold under `${STAGE1_FREEZE_DIR}/folds/<source>`:

1. Export normalized split files to Stage 1 `cotpause/{train,val,test}.json`
   using `scripts/data/export_normalized_pairs_for_stage1.py`.
2. Generate fold-specific Stage1 and Stage1b configs from the one-A100
   natural-pair templates.
3. Run:
   - `scripts/run_stage1_positionscan.py`
   - `scripts/run_stage1b_prompt_baseline.py`
4. Archive logs/configs/results under `COT_SAFETY_STAGE1_ARCHIVE_ROOT`.

The A100 templates use high per-run parallelism to keep GPU/CPU utilization
high during the tuning/debug phase.

## Known Non-Final Items

- S->S safe-prompt diagnostic datasets still require explicit input files
  before final scoring.
- HT is still external/quarantine-only and should not enter LOSO.
- Human QA remains the formal hard gate before claiming Stage 1 LOSO results.

## Verification Already Run Locally

```bash
bash -n pipelines/runpod_stage1_post_hb_freeze_then_loso.sh
bash -n pipelines/run_stage1_sequence.sh

.venv-stage1-test/bin/python -m pytest \
  tests/test_export_normalized_pairs_for_stage1.py \
  tests/test_fixed_budget_gen_gen_selection.py \
  tests/test_embedding_dedup_audit.py \
  tests/test_stage1_pair_freeze_audit.py \
  tests/test_stage1_text_baselines.py \
  tests/test_stage1_surface_audit.py \
  tests/test_stage1_loso_freeze_build.py \
  tests/test_stage1_aux_audits.py
```

Result: `26 passed`.
