# Fable Review: Stage3 Stage1-Paired WJB Screen + Eval-Shard Runner Patch

Date: 2026-07-06
Reviewer: Fable
Scope: WJB fold Stage3 screen result, eval/test sharding patch, passthrough data prep, `stage3_evidence` gate logic.
Method: all verdicts below are from direct inspection of the working-tree diffs (`git diff` on all 4 modified files), the runner/extractor/merge/probe source, the local result bundle (`stage3_evidence_report.json`, `summary_grid.json`, per-cell `metrics.json`, hidden manifests, run logs). Remote pooled summaries were not accessible this session; flagged below as an evidence gap. Only metrics/metadata were inspected, never raw CoT content.

## Verdicts

- **Patched runner + configs (eval sharding, passthrough prep, WJB/base configs): OK_TO_RUN** for the remaining Stage1 paired folds as-is. The sharding/merge mechanism is metrically safe (verified below).
- **Adjudication tooling (`src/cot_safety/probes/stage3_evidence.py`, `scripts/run_stage3_evidence_report.py`): EDITS_REQUIRED** before any future fold's `status` is treated as final. This does not block launching fold extractions — adjudication is post-hoc and re-runnable from saved artifacts — but the current report has a test-peeking selection bug, no CI, and no noise-floor guard.
- **Stage4: BLOCKED** (unchanged from the 2026-07-06 preflight). Nothing in this result advances Stage4 readiness.

Scientific reading of the WJB fold:

1. **Stage3 basic signal: POSITIVE.** Pause hidden states carry clear safe/unsafe signal: selected `pause_2` layer 21 test AUROC 0.7871 against an exact-0.500 prompt baseline.
2. **Independent pause-specific advantage: NOT ESTABLISHED — and at current statistical resolution, NOT DECIDABLE EITHER WAY.** The pre-registered gate (margin > 0.01) fails at +0.0080. But the pipeline's own empirical noise floor (see next section) is ~0.02 median, up to 0.094 — the margin, the gate, and their difference are all inside probe-training noise. `fail_no_independent_pause_signal` is the correct pre-registered outcome; it should not be read as "advantage disproven," nor should +0.008 be read as "nearly passed."
3. The result is consistent with — and, given Stage2's KL-transparency objective, roughly *predicted by* — the pause-as-passive-readout picture. That question is causal and can only be settled by liveness kernels, not probe margins.

## Headline finding: the 0.01 gate is undecidable at the current noise floor

The extraction geometry makes three position pairs *identical feature columns from the same forward pass*:

| alias A | alias B | identity |
|---|---|---|
| `cot_4` | `pre_pause_1` | token t4, last content token before the pause block |
| `cot_5` | `post_pause_1` | token t5, first content token after the pause block |
| `cot_6` | `post_pause_2` | token t6 |

(`original_reasoning_positions` excludes pause indices; `post_pause_k = pause_positions[-1]+k` — same sequence index, same hidden vector, same valid mask.)

Any AUROC difference between duplicate columns is therefore pure probe-training stochasticity. Observed in `summary_grid.json`: deltas span **0.004–0.094, median ≈ 0.02** (e.g. `cot_5` vs `post_pause_1` at layer 28: 0.678 vs 0.772, Δ = 0.094; `cot_6` vs `post_pause_2` at layer 7: Δ = 0.060).

Root cause is identifiable: with batch 2048 ≥ n_train 1100 training is full-batch, so shuffling is irrelevant — the only stochasticity is random init, frozen in by early stopping at a 30-epoch cap that every cell hits (`best_epoch` 26–30 across all cells; logistic regression is convex, so converged probes on identical features would be identical). Fix the convergence and the noise floor collapses; then the 0.01 gate becomes adjudicable (sampling noise handled by CI).

Consequences for this run:

- pause − control margin (+0.008) < median duplicate-column delta (~0.02): a single training run per cell cannot resolve it.
- Val selection is a lottery on top: val n=124 gives AUROC SE ≈ 0.045; the val-selected `pause_2` l21 (test 0.7871) is not even the best pause cell on test (`pause_2` l22: 0.7924).
- Test-side sampling error is small (n=4034, SE ≈ 0.007) — the problem is probe-fit variance and selection, not test size.

## What was verified

- **Eval-shard patch (`run_intra_pause_probe_full.py`)**: `split_json` buckets rows `idx % shard_count` and places the id-matched control row in the same bucket; shard specs are created in index order and `merge_shards` receives them in that order; extraction is per-row (no cross-row dependence); merge concatenates features/masks/labels/metadata in the same shard order, so intra-npz row↔label↔metadata alignment is exact. Pause and matched-control features live in the *same npz row* (control looked up by id inside the extractor), so control alignment is intra-row and permutation-proof. Downstream training is full-batch with per-epoch shuffle, so merged row order cannot affect metrics.
- **Passthrough prep (`prepare_intra_pause_probe_data.py`)**: `rewrite_probe_rows` is a verbatim refactor of the old inline `rewrite_rows`; pause row and no-pause row are built in the same iteration with the same `row_id`, and a failure of either build drops both — pairing preserved. `run_preserve_input_splits` does no caps/resplit/dedupe and reports prompt overlap (0/0/0 for WJB). It also now propagates `pair_id`/`match_family`/`prompt_id`/`split` onto probe rows.
- **Runner bridge (`scripts/run_stage3_intra_pause_probe.py`)**: `prepared_data_dir` correctly implies `--preserve_input_splits`, `--skip_base_data_prep`, `--no_heldout_sources`; `--extract_eval_shards` is passed from `hidden.extraction.eval_shards`.
- **Undeclared diff found**: `configs/experiment/stage3_intra_pause_probe.yaml` flips `current_control_cot_aliases_valid: false → true`. Verified truthful — `control_cot_*` now comes from a separate matched no-pause forward via `--matched_control_file` with id lookup (duplicate ids raise; missing/label-mismatched/too-long controls drop the row). Declare flag flips like this in future briefs.
- **Run logs**: extraction commands match the config (`--pause_layout intra_cot --cot_offsets 4,5,6,9,10 --control_cot_offsets 5,6 --max_length 4096 --matched_control_file .../nopause/<split>.json --prompt_positions last_prompt_token,pre_think`); test shards merged in order 0–3 (1009+1010+1008+1007 = 4034).
- **Manifests**: the 4 dropped test rows are all `too_long` and all unsafe (2019 → 2015). Train shard label composition (275u/275s/275u/275s) shows `train.json` strictly alternates labels — harmless (extraction is per-row; training shuffles), but it confirms nothing downstream may ever consume unshuffled row order.
- **Prompt baseline = exactly 0.5000 in all 12 cells (2 positions × 6 layers, val and test)**: this is a *design guarantee*, not an empirical near-miss — paired members share the identical prompt, so prompt-position features are identical within pairs and AUROC is all-ties 0.5. It doubles as a passed pairing-integrity check.

## Answers to the seven questions

**Q1 — Is "pause has signal, but independent pause-specific advantage not established" the better interpretation than plain fail?**
Yes, with one tightening: the honest statement is three-part. (a) Basic pause signal: established (0.787 vs 0.500). (b) Independent advantage: not established — the pre-registered gate fails and that outcome stands. (c) Not disproven either: the margin is inside the pipeline's own noise floor, so this fold cannot decide the pause-specific question in either direction. Recommend the status vocabulary become three-way (`pass` / `fail` / `undecided_insufficient_resolution`) — this fold, once CIs exist, will most likely land `undecided`.

**Q2 — Is the near-chance prompt baseline meaningful evidence against prompt-only classification?**
Yes, but as a structural guarantee rather than an empirical discovery. Exact 0.500 is forced by the paired design (identical prompts across labels within a pair → identical prompt features → all-tie AUROC). So it *proves* probe signal originates in CoT tokens, by construction, and confirms pairing integrity survived the pipeline. It is stronger than a typical "baseline came out near chance" observation — but it carries no information about anything except the pairing.

**Q3 — Does control ≈ pause imply the pause is a readout of nearby content, not an independent steering port?**
Favored but unproven. Three converging correlational observations: (i) `control_cot_6` is within 0.008 of the best pause cell; (ii) the information-matched pre-pause position `cot_4`/`pre_pause_1` — which sees exactly the same content prefix t0–t4 as the pause block — already reaches ≈0.78, so three rounds of pause computation on top add ~nothing measurable; (iii) the ramp (`pre_pause_3`≈0.53–0.56 → `pre_pause_2`≈0.71 → t4≈0.78) shows separability accrues in the first five content tokens. Note also that controls at t5/t6 see 1–2 *more* content tokens than pauses do, so the control comparison is conservative against pause — yet the information-matched comparison agrees. Two caveats: this is exactly what Stage2's KL-transparency objective optimizes for (pauses trained not to change continuations should tend toward passive summaries), so the result is closer to "objective achieved" than "hypothesis falsified"; and AUROC equality is correlational — it cannot distinguish "pause mirrors content" from "pause aggregates content and gates downstream behavior." Only `pause_kv_ablation` / `safe_unsafe_patching` can answer the steering-port question.

**Q4 — Other folds next, or on-policy generation + CoT-segment judging first?**
Neither first — free statistics first. The per-example scores needed for CIs already exist (`predictions_val.jsonl` / `predictions_test.jsonl` per run dir; the report's "bootstrap CI requires per-example scores" note is stale). Re-adjudicate WJB on CPU in hours, then run the remaining folds (cheap, comparable, tests generality of the readout picture), and build the on-policy chain *in parallel* — it is the Stage4 long pole regardless of fold outcomes. Do not serialize folds behind the on-policy build.

**Q5 — Does the eval/test sharding preserve row ordering and matched-control alignment?**
Yes — verified mechanism, not just summary (see "What was verified"). Two caveats: (a) merged row order is a deterministic interleave permutation of the split JSON — harmless because alignment is via in-npz metadata arrays, but never zip npz rows against the split JSON by index; (b) one real defect: `merge_hidden_shards.py` `ROW_KEYS` drops `pair_ids`, `match_families`, `source_families`, `risk_types` that the extractor saves — metrics are unaffected, but merged splits lose `pair_id` (predictions carry `pair_id: null`; `prompt_key` survives and is a valid pair key for WJB). Fix before anything downstream relies on `pair_id`.

**Q6 — Change Stage2 to make pauses more "live", or freeze?**
Freeze. (a) A within-noise margin is not a design signal — redesigning Stage2 off +0.008 is chasing noise. (b) Changing Stage2 invalidates cross-fold comparability and reopens the preflight. (c) "More live" pauses trade directly against the KL-transparency that legitimizes the teacher-forced screen — that tension is real and should be confronted deliberately *after* multi-fold CIs and liveness kernels show a consistent picture, not patched reactively. If the readout-only picture holds up causally, the right response may be to reframe pause as a monitoring tap rather than a steering port, not to retrain Stage2.

**Q7 — Most important concrete code changes before Stage4?**
See "Exact code/config changes." Priority order: probe determinism/convergence fix, CI + three-way status in the evidence report, `best_main` selection fix, `ROW_KEYS` fix, then the on-policy producer chain and liveness kernels (the actual Stage4 gates).

## Blocking issues

For trusting any future fold adjudication (EDITS_REQUIRED items):

1. **No CI on the margin, and the CI stub's excuse is stale.** `stage3_evidence.py` hardcodes CI "not_available_from_summary_grid" while per-example predictions with `prompt_key` sit in every run dir. A 0.01-wide gate without a CI is not adjudicable.
2. **Probe training non-determinism + non-convergence.** Every cell stops at the 26–30/30 epoch boundary; random init variance produces up to 0.094 AUROC spread on *identical features*. Until fixed, the entire 108-cell grid has ~0.02 cell-level noise and no 0.01-scale comparison is meaningful.
3. **`best_main` test-peeking** (`stage3_evidence.py` ~line 128): the headline `status` picks between val-selected pause and post_pause candidates by *test* AUROC. Optimistic bias in the headline metric. No effect on this run (pause won under both criteria; `pause_only_status` is unaffected) — but fix before the next report.

For Stage4 (carried from preflight, unchanged): on-policy producer chain (multi-sample generation, `cot_segment_judge`, converter) and liveness kernels (`pause_kv_ablation`, `safe_unsafe_patching`) do not exist. No Stage3 fold result, however clean, substitutes for these.

## Non-blocking concerns

1. `merge_hidden_shards.py` `ROW_KEYS` metadata drop (pair_ids etc.) — fix is one line.
2. Pooled results (pause_concat, control concat) exist in remote logs but were neither incorporated into the evidence report nor copied into the review bundle — unreviewed evidence gap; include next time.
3. Val n=124 selection lottery — inherent to preserving Stage1 splits (correctly preserved; do not resplit). Mitigate with CIs, the within-pair endpoint, and cross-fold pooling.
4. Threshold FPR drift: 0.05 on val → 0.11–0.15 on test. Expected with a 124-row val; fine for a screen, matters for any deployment-style claim later.
5. 4 unsafe `too_long` test drops (0.2%) — asymmetric (removes longest unsafe rows) but far too small to move AUROC materially; add per-label drop counts to the evidence report rather than raising `max_length`.
6. `intra_pause_probe_full_config.json` is written to the shared `runs/probes/` parent and overwritten by every run — provenance clobbering.
7. `model_kinds: [linear, mlp]` in config vs hardcoded `--model_kinds linear` in the scan — config lies; align it.
8. Undeclared config flag flip (`current_control_cot_aliases_valid`) — truthful, but list every diff in the brief next time.
9. Preserve-splits path defers row-id uniqueness checking to extraction time (classic path checked at prep via `map_rows_by_unique_id`); add an assert in `run_preserve_input_splits`.
10. The reviewed run extracted with batch size 4 (logs) vs 32 now configured — benign (per-row outputs; at most tiny bf16 numeric wiggle), noted for provenance.

## Recommended next run order

1. **R0 — CPU-only WJB re-adjudication (no GPU, ~hours).** From existing `predictions_test.jsonl` of `linear_pause_2_l21` and `linear_control_cot_6_l28`: paired bootstrap over `prompt_key` clusters (B≥1000) + paired DeLong on the margin; within-pair ranking endpoint P(score_unsafe > score_safe) over the 2015 intact test pairs (teacher-forced analogue of the planned on-policy `within_prompt_auroc`, min 0.55); duplicate-column noise-floor table from `summary_grid.json`. Expected outcome: WJB formally lands `undecided_insufficient_resolution`.
2. **R1 — probe determinism fix, then re-scan cached WJB hiddens (GPU-light, no extraction).** Zero-init + raised epoch cap (below), re-run single scan + evidence report on the existing npz. This collapses the noise floor and may by itself make the WJB gate decidable.
3. **R2 — remaining Stage1 paired folds** via the patched runner (OK_TO_RUN) with the fixed adjudication; per-fold verdicts plus a cross-fold pooled margin with CI. Decide pause-specificity on pooled evidence, not single-fold point estimates.
4. **Parallel dev track — on-policy producer chain** (generation + `cot_segment_judge` + converter). Long pole for Stage4; start now, gate nothing on it.
5. **Liveness kernels** (`pause_kv_ablation`, `safe_unsafe_patching`) once R1/R2 give a stable correlational picture — required before any steering-port claim, and the only way to answer Q3 causally.
6. **Stage2 frozen. Stage4 blocked** until: CI-supported cross-fold Stage3 picture + liveness results + on-policy screen.

## Exact code/config changes

1. `src/cot_safety/probes/stage3_evidence.py`
   - Fix `best_main` (~line 128): select among candidate main rows by `selection_metric` (val AUROC), not test; report test only.
   - Replace the CI stub: load the two selected cells' `predictions_test.jsonl`, join on `example_id`, cluster-bootstrap `prompt_key` (B≥1000) → CI on `AUROC_pause − AUROC_control`; add paired DeLong p-value.
   - Three-way gate: `pass` iff margin > 0.01 AND CI low > 0; `fail_no_independent_pause_signal` iff CI high < 0.01; else `undecided_insufficient_resolution`.
   - Add `probe_noise_floor`: per-layer AUROC deltas across the three duplicate pairs (`cot_4`/`pre_pause_1`, `cot_5`/`post_pause_1`, `cot_6`/`post_pause_2`); if margin < median delta, force `undecided`.
   - Add per-label extraction drop counts from the hidden manifests.
2. `legacy/PauseProbe/scripts/probe/run_position_scan_batched.py`
   - Zero-init probe weights (convex objective → deterministic optimum; kills the duplicate-column spread), raise the epoch cap 30 → 100 with val-AUROC patience ≈ 10, and log a loud warning whenever `best_epoch == cap`.
   - Optional: `--seeds` to train k probes per cell and report seed-variance.
3. `legacy/PauseProbe/scripts/probe/merge_hidden_shards.py`
   - `ROW_KEYS += {"pair_ids", "match_families", "source_families", "risk_types"}` — or derive dynamically as every array key whose first dimension equals n_rows.
4. `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
   - Write `intra_pause_probe_full_config.json` into each run's own out roots (single and pooled), not `Path(args.pooled_out_root).parent`.
5. `scripts/run_stage3_evidence_report.py`
   - Ingest pooled summaries (`pause_concat_layers_concat` vs `control_cot5_cot6_concat_layers_concat`) as a secondary pooled margin row.
6. `configs/experiment/stage3_intra_pause_probe.yaml`
   - `model_kinds: [linear]` (or implement mlp in the batched backend; don't ship a config the scan rejects).
7. `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
   - Assert row-id uniqueness inside `run_preserve_input_splits` (fail at prep, not extraction).
8. Process: copy pooled summaries and the selected cells' `predictions_*.jsonl` into the review bundle for every future fold.

No change to the 0.01 gate threshold itself, to Stage2, or to the Stage1 splits.
