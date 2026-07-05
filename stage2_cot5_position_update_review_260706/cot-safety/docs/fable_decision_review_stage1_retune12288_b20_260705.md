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

## Fable-5 Probe Redesign Review

Follow-up review:

- tmp repo commit: `0ea86e7`
- packet path: `stage1_fable_review_retune12288_b20_260705/`
- response: `fable_probe_redesign_response_260705.md`
- reviewer mode: Fable-5 pro/high-rigor

Final decision label:

> ONLY AFTER GATES

Fable-5 keeps the current Stage1 run as a negative/control result for the
current linear hidden-probe design. It does not approve another GPU sweep of
the same design.

The methodological diagnosis is sharper than the first decision review:

- The dominant failure is the evaluation contrast, not simply linear probe
  capacity.
- Hidden probes use a prefix-limited hidden snapshot, while surface baselines
  use full-trajectory hindsight text.
- `length_only` beats the selected hidden probe on all four sources, suggesting
  that much of the surface advantage is outcome-correlated hindsight structure.
- The early-position variant has effectively already been run through Stage1b;
  trying more early positions alone is not a scientific fix.

The approved redesign is CPU-only matched-horizon reanalysis on frozen data:

- Compare hidden@k against surface features from the same emitted prefix
  at k, not against full-completion text.
- Use k in `{4, 8, 16, 32, 64}`.
- Report paired delta AUROC, within-pair ranking accuracy, and residual
  delta log-loss / delta AUROC.
- Treat full-text surface baselines and `length_only` as hindsight reference
  lines, not matched competitors.

GPU is only conditionally allowed after non-GPU gates pass and Phase-1 HB+WJB
results satisfy continue criteria. Its only approved use is regeneration of
missing RS/SR hidden arrays from frozen configs; it is not approval for another
position/layer/classifier sweep.

The kill criterion is also explicit: if hidden@k does not beat matched text@k
with CI-separated deltas at early horizons on HB and WJB, Stage1 probing should
stop permanently and be reported as a well-posed negative/control result.

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
