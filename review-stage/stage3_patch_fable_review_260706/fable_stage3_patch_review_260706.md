I couldn't get approval to re-run the tests, so I'm relying on your reported result (4 passed) plus my own reading of the test code. Here is the complete review.

---

# Fable Review: Stage3 Patch (stage3_evidence + batched scan + fold configs) — 2026-07-06

**Scope reviewed:** all 12 listed files, plus their load-bearing dependencies (`run_stage3_intra_pause_probe.py`, `cot_safety/config.py`, `train_probe.py` — `predictions_rows`/`make_sample_weights`, `run_position_scan_pilot.py` — `build_summary`, `extract_hidden_states.py` — position/metadata semantics, `configs/runtime/a6000_2x.yaml`).

**Verdict in one line:** The patch is scientifically aligned and the leakage fix is real; the extraction/scan side is safe to launch on the remaining three folds now, but two report-side issues (bootstrap cluster unit, top-level `status` semantics) must be fixed before any fold status is treated as final, and WJB's scan must be rerun under the new probe settings before cross-fold comparison.

---

## Blocking Issues

### B1. Cluster bootstrap resamples by `prompt_key`, not `pair_id`

`stage3_evidence.py` prefers `prompt_key` and only falls back to `pair_id`. For the Stage1 natural-pair folds, the sampling/dependency unit is the matched pair, and pair-mates are different prompts. With `prompt_key` clustering, the bootstrap degenerates to an approximately i.i.d. per-example bootstrap, ignoring within-pair dependence induced by matching.

Fix: check `pair_id` first, then `prompt_key`, then the prediction key.

### B2. Top-level `report["status"]` still means the stronger endpoint

A fold where `pause_signal.status == "pass"` but the independent margin is undecided/failed will show `"status": "fail_no_independent_pause_signal"` at the top of the JSON. Any adjudication step that greps top-level `status` will record "fail" for a fold that satisfies the stated minimum useful Stage3 claim.

Fix: make top-level `status` a composite, or rename the current field to `independent_status` and promote `pause_signal.status` alongside it.

### B3. Rerun WJB scan under the new probe settings before cross-fold comparison

Zero init plus epochs=100/patience=10 change the probe estimator. Existing WJB hidden states are fine, but WJB's scan and evidence report should be rerun before comparing against remaining folds.

---

## Verified Correct

- Leakage fix is real: main selection is validation-based, test is only reported.
- Bootstrap AUROC/tie handling and prediction-root layout are broadly correct.
- Noise-floor duplicate pairs are correct for `insert_cot_offset=5`.
- Metadata merge carries `source_families`, `risk_types`, `pair_ids`, and `match_families`.
- Config snapshots are no longer written to a shared parent.
- Preserve-splits ID uniqueness checks are correct.
- New fold configs are consistent and fold-specific.
- `probe.model_kinds: [linear]` now matches the actual scan implementation.
- True-content controls are genuine matched no-pause forwards.

---

## Non-Blocking Issues

- Noise-floor pairs are hardcoded to `insert_cot_offset=5`; derive from config or assert offset is 5.
- Noise floor is diagnostic and may underestimate probe-training noise under zero-init/shared worker batching.
- `pause_signal` has no uncertainty treatment yet; ideally add bootstrap CI for pause vs prompt too.
- `pause_only_status` can still pass without CI when best main is post-pause; either compute a second bootstrap or force undecided.
- Merge silently fills optional missing metadata keys with empty strings on mixed old/new shards; record key presence or warn.
- With small folds and batch_size=2048, probe training is full-batch AdamW; inspect first fold histories to confirm plateau.
- Preserve mode counts prompt overlap but does not assert it.
- CI margin is computed on joined examples while summary margin comes from per-probe metrics; usually small but should be documented.
- Add synthetic tests for bootstrap, noise floor, and undecided branches.

---

## Answers

1. Scientifically aligned: yes.
2. `pause_signal` vs `independent_pause_signal`: conceptually right, but top-level status must not contradict it.
3. Correctness bugs: substantive one is bootstrap cluster unit; noise floor is correct only for offset 5.
4. OK to run remaining folds: yes for extraction + scan, but fix B1/B2 before treating status as final.
5. Rerun WJB: yes, scan + report only, reusing extracted hidden states.
6. Before Stage4: on-policy producer chain, CoT judge labels, within-prompt AUROC, liveness checks, B1/B2, WJB rerun, and preferably CI on the primary pause-signal screen.

Bottom line: launch harmbench/reasoningshield/strongreject extraction+scan after landing the two small report fixes; rerun WJB scan before writing cross-fold verdicts.
