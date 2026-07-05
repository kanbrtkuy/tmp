# Fable-5 Review - Round 3: Is Any Non-Fishing Stage1 Improvement Path Left?

The packet is complete (5 files, all read, all aggregate-level). Here is the
review.

## Verdict: `FRESH_PREREG_ONLY`

This entails `STOP_CURRENT_STAGE1` for the frozen test set.
`CODE_AND_RUN_ALLOWED` is rejected: no improvement-seeking code or run on the
current frozen Stage1 test set is scientifically valid. The only admissible
positive path is a fresh, separately preregistered Stage2/on-policy experiment.
On the frozen set, the only permitted remaining work is claim-neutral or
claim-strengthening-of-the-negative (reporting, integrity gates, write-up) -
none of it may be described as "making Stage1 better."

## Direct Answers

1. Any legitimate improvement path left on the frozen set? No. Every analysis
   aimed at producing a positive cell on this test set is now post-hoc fishing.
   The only legitimate remaining work is:
   - operating-point reporting;
   - completing formal blockers for the negative claim (human QA summary,
     S-to-S safe-prompt diagnostics, HT quarantine, G1-G8);
   - a sensitivity/power statement quantifying what the CIs exclude;
   - documenting the 133-file coverage-gap mismatch.

   One narrow exception exists in principle: if an integrity audit uncovers a
   genuine implementation bug in the hidden arm, a single declared bug-fix rerun
   of the full preregistered pipeline is error correction, not improvement. No
   such bug is in evidence.

2. Stopping rule and phrasing: the preregistration already contains it and it
   has fired twice: `a1_fail -> drop_leadtime_claim`, and the M1 kill criterion
   ("if hidden@k - text@k is not CI-separated positive at any k <= 32,
   permanently stop Stage1 probing"). A2 failed at every k and the
   excluded-source confirmation failed all three gates.

   User-facing phrasing:

   > Stage1 is closed. The preregistered confirmation tests were run, they came
   > back null-to-negative, and the prereg rules say that ends it. The frozen
   > test set's multiplicity budget is spent; any further search on it would
   > produce results we could not trust or publish. The negative result is the
   > result.

3. Minimum valid next experiment: Stage2/on-policy, fresh data, single
   preregistered early-horizon endpoint.

4. Threshold/balanced accuracy: allowed strictly as operating-point reporting.
   It may never be phrased as "Stage1 improved." AUROC is threshold-invariant
   and unchanged; even retuned hidden BA (~0.71) remains below surface (~0.865)
   and length-only (~0.801).

5. Loop status: Fable recommends marking the improvement loop complete, not
   blocked and not continued. Any continuation is a new loop under a new
   preregistered Stage2/on-policy objective.

## Rationale

- The confirmatory budget is spent. The frozen test set has now supported the
  main Stage1/Stage1b runs, threshold reanalysis, matched-horizon reanalysis,
  A1, A2, and the excluded-source confirmation. SR/RS were the last
  quasi-held-out data within the Stage1 pool, and they were consumed by the
  final confirmation, which failed decisively.
- The failure is not marginal. All 16 hidden-minus-surface deltas are negative;
  length-only beats the hidden probe on all four sources; matched-horizon
  deltas are negative for every k >= 8; the k=4 hint shrank under A2 and to
  null in excluded-source confirmation.
- The conditional GPU permission is void. The `ONLY AFTER GATES` decision
  allowed GPU regeneration of RS/SR hidden arrays only if Phase-1 continue
  criteria passed. They failed.
- The negative is still valuable: hidden probes are above chance
  (0.68-0.84 AUROC), prompt/pre-CoT controls are near random, and pipeline
  integrity checks passed. This is a well-executed, interpretable negative:
  linear prefix-hidden probes do not beat matched or full-trajectory surface
  baselines in this teacher-forced natural-pair setting.

## Minimum Next Experiment

- Stage: Stage2/on-policy, not Stage1 again.
- Data: fresh prompts never used in any Stage1 split; new on-policy rollouts
  from the target model; new pair/example freeze with hashes committed before
  any hidden extraction.
- Primary endpoint: paired/clustered delta AUROC, hidden@k* minus
  matched-capacity surface@k* at a single preregistered early horizon k* in
  `{4, 8}`. Gate: bootstrap CI-low > 0.
- Feasibility gate before GPU: preregistered power check.
- Stop gate: primary CI includes 0 -> stop hidden-vs-surface early-warning line
  program-wide. No secondary-endpoint rescue, no horizon shopping.

## Code To Write, If Any

Before any Stage2 data is unblinded: the full Stage2 pipeline -
rollout generation/labeling harness, freeze manifest with hashes, matched
surface-feature spec at k*, probe training, and a single-shot gate evaluator -
committed, Fable-reviewed, and dry-run on Stage1-train-only or synthetic data.

On the frozen Stage1 set: report-generation code only.

## What Not To Do

- No new probe families, layers, positions, classifier heads, pooling schemes,
  calibration variants, or nonlinear rescues on the frozen test set.
- No subgroup, per-source, or per-k slicing to locate a positive cell.
- No re-splitting, re-freezing, or re-weighting of the existing 2,783 pairs to
  manufacture a "new" test set.
- No GPU regeneration of RS/SR hidden arrays.
- No phrasing of threshold/BA correction as an improvement or partial success.
- Do not start Stage2 informally.

## User-Facing Summary

The improvement loop ran to completion and did its job. We tested the main
design, then every reasonable fair-comparison and early-warning variant, each
under rules written down in advance. Every confirmation came back negative or
null - the hidden-state probe reads real signal, but it never beats simply
reading the text, not even a length counter. The rules say this ends Stage1:
the test set has been examined so many times that any further "win" found on it
would be a statistical artifact, not a discovery. Stage1 is now a solid,
honest negative result. Nothing more should be run on this data except
finishing audit paperwork and writing it up. The one legitimate route forward is
a new, preregistered on-policy experiment (Stage2) on fresh data with a single
pass/fail test decided before looking at any results.
