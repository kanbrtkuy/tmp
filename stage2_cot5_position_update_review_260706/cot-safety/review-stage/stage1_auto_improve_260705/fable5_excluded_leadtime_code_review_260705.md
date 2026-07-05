# Fable-5 Code Review: Stage1 Excluded-Source Lead-Time Confirmation

Date: 2026-07-05

Packet:
`tmp/stage1_auto_improve_loop_260705/round1_excluded_leadtime_code_packet`

tmp commit: `c7a427e`

```text
Review complete. All seven checklist items verified against the pinned plan and amendment.

Verdict: OK_TO_RUN

Blockers: none.

Checklist verification:

- All-k frozen test population — build_prediction_files intersects pair-complete test pairs across the full k grid (run_stage1_excluded_leadtime_confirmation.py:288), filters every k's written test files to that set, and assert_same_pairs_across_k (:156-177) hard-fails on any missing/extra pair or non-{0,1} pair at any k, both arms. A1 consumes only the frozen files; A2 consumes the same frozen surface files with fail_on_error=True, so any archive-side pair loss is a hard error, not silent drift.
- Min-power halt — fires per source on the all-k frozen count before any A1/A2 metric computation (:289-293), default 150, recorded in the prereg JSON; SystemExit gives nonzero exit, so a halted run can't be interpreted.
- No future leakage — surface models fit per k on train-split retained rows only; A1 reuses archived single-position scores with val-stat z-scoring (rule="zmean"); hidden@4 uses only j<=4 positions; prereg JSON with estimands/gates is written before build_prediction_files (:419-438), satisfying the write-before-fit/score requirement.
- Recipe pinning — layer hard-pinned to 28 and surface family hard-pinned to char_tfidf via SystemExit checks (:410-413); vectorizer/estimator defaults match the amendment (min_df=1, char_wb 3-5, max_features 200k, max_iter 2000, seed 260705, B=500); --code-commit and --tmp-prereg-commit both required; whitespace-tokenizer fallback off by default.
- Gates — exact match to the amendment's pins: A1 CI-low >= 0, per-source min >= -0.02, A2 CI-high >= 0 AND point >= -0.01, with the four-way decision mapping including the low-power-pass floor (:359-370). find_row fails closed on missing/duplicate cells.
- Extract-minimal helper — SR/RS only, layer 28 only, cot_offsets 4,8,16,32,64, empty prompt positions, pause_layout=none, n_pause_tokens=0, label_field=trajectory_safety_label, max_length=12288, frozen prepared splits, set -euo pipefail so extraction failure propagates. Matches the approved extract-minimal scope exactly.
- Tests — the e2e test exercises the load-bearing property: a pair censored only at k=8 is removed from the k=4 files too (frozen_test_pairs == 5 asserted at every k), plus n_errors=0, prereg propagation, and gate decision validity through the real A1/A2 payloads.

Non-blocking notes (no action required for OK_TO_RUN): the R2 backup required by the amendment is not in the helper — do it as the operational post-step; and the min-power halt is CLI-overridable (like sources/k-grid) but recorded in the prereg JSON, so the results-packet review must confirm minimum_pairs_per_source=150 and the default grid/sources were used.

Run conditions stand as preregistered: non-null --code-commit, --tmp-prereg-commit = the registry commit hash, exit 0, n_errors=0; anything else is INVALID and returns to review.
```
