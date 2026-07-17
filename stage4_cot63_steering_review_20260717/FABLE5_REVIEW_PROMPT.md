# Fable5 Review Prompt

Please review this Stage4 cot63 steering bundle as a code/results/method audit.

Read these files first:

- `README.md`
- `code/diagnose_stage4_cot63_target309.py`
- `code/run_stage4_judge.py`
- `results/score_gate_existing_a2_summary.json`
- `results/projection_scores.jsonl`
- `results/gprs_replay_diagnostic_analysis_summary.json`
- `results/gprs_wildguard_normalized/*.jsonl`

Questions:

1. Algorithm correctness:
   - Does the current fixed-relative steering path preserve strict A1-replay counterfactual semantics?
   - Does any part of the code accidentally use A2 outcome, judge label, or post-intervention information in generation-time gating?
   - Are hidden-state target positions/layers handled consistently?

2. Analysis correctness:
   - Does the code mishandle WildGuard labels (`unsafe_valid`, `safe_refusal`, `ambiguous_partial`) anywhere besides the known generic analyzer caveat?
   - Which reported metrics are safe to use in the paper, and which should be marked diagnostic only?

3. Result patterns:
   - Why does fixed-relative alpha=0.1 flip 30/99 baseline-unsafe rows, while legacy GPRS replay only flips 9/99 at alpha=0.1 and gets worse at alpha=2?
   - What patterns do you see in the score-gated result table?
   - Does score-gating look promising enough to run a fresh generation grid, or is current evidence weak?

4. Upgrade plan:
   - Which small algorithm upgrades should be tried first on the same 309 rows?
   - Please prioritize methods that do not break the strict A1-replay counterfactual claim.
   - Specifically evaluate:
     - score-gated fixed-relative steering
     - adaptive-alpha fixed-relative steering
     - two-layer small-norm steering, e.g. `{14,15}`
     - multi-layer score ensemble for gate/alpha

5. Reviewer-risk assessment:
   - What would a skeptical reviewer object to?
   - What experiments or wording would reduce that risk?
   - Is the current Stage4 bottleneck more likely steering algorithm, pause SFT contract quality, position/layer choice, label noise, or sample composition?

Please give a concrete minimal next experiment plan with:

- exact variants to run;
- exact alphas/gates/layers;
- expected diagnostic outputs;
- pass/fail criteria;
- what should be considered main result vs ablation.
