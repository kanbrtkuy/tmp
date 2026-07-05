# Fable Decision Review: Stage1 retune12288_b20

Date: 2026-07-05

Source packet:

- tmp repo commit: `1c8b6a5`
- packet path: `stage1_fable_review_retune12288_b20_260705/`
- decision request: `decision_addendum_260705.md`
- response archived in tmp: `fable_decision_response_260705.md`

## Decision

Fable recommends accepting Stage1 as a negative/control result for the current
hidden-probe design.

Final decision label:

> Accept negative/control; no more GPU for current Stage1 design.

## Cause Attribution

Fable's rough attribution:

- Data problem: 10-15%
- Probe/method problem: 70-75%
- Claims/expectation problem: 10-15%

Fable's interpretation is that the weak Stage1 result is mainly a method/probe
architecture issue, not a data-quality failure.

## Evidence

Fable emphasizes:

- All 16 hidden-minus-surface AUROC deltas are negative.
- Confidence intervals are tight and bounded away from zero.
- The result holds across HarmBench, ReasoningShield, StrongReject, and
  WildJailbreak.
- The result holds across Stage1, Stage1b, and multilayer settings.
- Surface baselines on the same frozen splits reach roughly 0.91-0.97 AUROC,
  showing that the data contain strong discriminative signal.
- Row-audit mismatches are localized to Stage1 linear high-CoT-offset positions
  (`cot_96`, `cot_128`) rather than broad extractor-level row drops.

## Data Interpretation

Fable does not see evidence that the data are too compromised to interpret the
current negative result.

Evidence against a data-failure explanation:

- Splits are balanced by label.
- Duplicate IDs are not present in the audited prepared splits.
- Row-count mismatches are localized and do not affect Stage1b/multilayer in
  the same way.
- Surface baselines perform strongly on the same data.
- Fixed-budget selection and frozen LOSO splits provide controlled provenance.

## If Redesign Is Pursued Later

Fable does not recommend another GPU run for this design. If a new hidden-state
probe is pursued, Fable recommends that it be a scientifically motivated
redesign, not hyperparameter fishing.

Possible redesign directions:

- Early reasoning positions rather than late CoT offsets only.
- Nonlinear classifier with an explicit separability rationale.
- Principled multi-layer or multi-position aggregation.
- Experiments that separate layer-selection effects from position-selection
  effects.

Non-GPU gates before any redesigned GPU run:

- Coverage validation for the selected positions/layers.
- Written redesign rationale explaining why the new design addresses the
  current failure mode.
- Baseline stability check on the same frozen splits.
- Pre-GPU data completeness audit.
- Validation-only hyperparameter protocol fixed before running.

## Allowed Claims After Remaining Checks

After human QA, S-to-S safe-prompt diagnostics, and HT quarantine/external
testing, Fable says the allowed claim is narrow:

- The tested linear hidden-state probes on the tested CoT positions/layers did
  not outperform validation-selected surface text baselines on held-out safety
  classification.
- This negative result is consistent across the four tested sources.
- Alternative hidden-probe architectures remain untested.

Not allowed:

- Hidden states lack safety signal in general.
- Surface text is sufficient for safety detection in general.
- The tested layers are generally poor for safety across all positions and
  architectures.
