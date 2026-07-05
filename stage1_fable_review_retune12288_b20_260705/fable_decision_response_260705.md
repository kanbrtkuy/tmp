# Fable Decision Review: Stage1 retune12288_b20

Date: 2026-07-05

Reviewed sanitized packet:

- `README.md`
- `decision_addendum_260705.md`
- `fable_response.md`
- `delta_ci/hidden_surface_delta_ci_summary.tsv`
- `val_fixed/val_fixed_probe_report.tsv`
- `row_audit/stage1_prediction_row_audit_summary.json`
- `docs/stage1_post_hb_retune12288_b20_gap_audit_260705.md`
- `text_baselines/<source>/summary.tsv`

The packet excludes raw prompts, raw CoTs, completions, and row-level
prediction JSONL files.

## Verdict

Fable's decision:

**Accept Stage1 as a negative/control result for the current hidden-probe
design. Do not spend more GPU on this exact design.**

The weak Stage1 performance is more likely a probe/method architecture problem
than a data quality problem.

Approximate attribution from Fable:

- Data problem: 10-15%
- Probe/method problem: 70-75%
- Claims/expectation problem: 10-15%

## Rationale

Fable says the evidence is decisive:

- All 16 hidden-minus-surface AUROC deltas are negative.
- Confidence intervals are tight and bounded away from zero.
- Negative deltas hold across all four sources: HarmBench, ReasoningShield,
  StrongReject, and WildJailbreak.
- Negative deltas hold across Stage1, Stage1b, and multilayer settings.
- Hidden probes reach roughly 0.70-0.84 test AUROC, while validation-selected
  surface baselines reach roughly 0.91-0.97.

Fable does not interpret this as marginal ambiguity. It is systematic
underperformance of the current hidden-probe design relative to simple
word/character surface baselines.

## Why Fable Does Not Blame The Data

Fable points to these aggregate checks as evidence against "the data are simply
bad":

- Splits are label-balanced within each source and split.
- Row audit reports zero duplicate example IDs.
- Remaining row-count mismatches are localized to Stage1 linear high-CoT-offset
  positions (`cot_96`, `cot_128`), not broad extractor-level full-row drops.
- Stage1b and multilayer outputs do not show the same mismatch pattern.
- Surface baselines achieve high AUROC on the same frozen splits, proving that
  strong discriminative signal exists in the data.
- Fixed-budget sampling and LOSO freeze provide controlled provenance.

Fable says the 133 mismatching prediction files do not indicate dataset
corruption. They are more consistent with incomplete hidden-state coverage at
specific high offsets.

## If A Redesign Is Forced

Fable does not recommend another GPU run for the current design. If the project
nevertheless wants to explore hidden-state signal, the redesign must be
scientifically motivated rather than hyperparameter fishing.

Minimum credible redesign directions:

- Test early reasoning positions rather than late CoT positions only.
- Try a nonlinear classifier if there is a specific rationale for nonlinear
  separability.
- Use a principled multi-layer or multi-position aggregation scheme rather than
  ad hoc concatenation.
- Separate layer selection effects from position selection effects.

Required non-GPU gates before any redesigned GPU run:

- Coverage validation for the new positions/layers.
- A short redesign rationale explaining why the new architecture addresses the
  failure mode of the current linear probe.
- Baseline stability check on the same frozen splits.
- Pre-GPU data completeness audit stricter than the current high-offset
  mismatch tolerance.
- Validation-only hyperparameter protocol fixed before GPU execution.

## Claims Allowed After Remaining Checks

After human QA, S-to-S safe-prompt diagnostics, and HT quarantine/external
testing, Fable says the project can claim:

- A linear hidden-state probe on the tested CoT positions/layers did not
  outperform validation-selected surface text baselines on held-out safety
  classification.
- The negative result is consistent across the four tested sources with
  confidence intervals not overlapping zero.
- The limitation is specific to the current probe design; alternative hidden
  probe architectures remain untested.

Fable says not to claim:

- Hidden states lack safety signal in general.
- Surface text is sufficient for safety detection in general.
- The tested layers are generally poor for safety across all positions and
  architectures.

## Final Decision Label

Accept negative/control; no more GPU for current Stage1 design.
