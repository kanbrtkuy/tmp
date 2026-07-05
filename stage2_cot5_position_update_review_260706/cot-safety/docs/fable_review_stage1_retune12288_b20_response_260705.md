# Fable Review Response: Stage1 retune12288_b20

Date: 2026-07-05

Reviewed packet:

- GitHub tmp repo: `kanbrtkuy/tmp`
- Tmp commit: `ec5af66`
- Packet path: `stage1_fable_review_retune12288_b20_260705/`

## Verdict

Fable verdict: **acceptable execution, major claim architecture issue**.

The post-GPU audit and hidden-minus-surface CI are acceptable, but the current
Stage1 result should be treated as a robust negative/control result unless the
probe design is changed.

## Row Audit

Fable confirms that the post-GPU row audit supports the interpretation that
remaining mismatches are high-CoT-offset coverage gaps, not extractor-level
full-row drops.

Evidence accepted by Fable:

- 4464 prediction files audited.
- 133 mismatches.
- Mismatches are localized to Stage1 linear high-CoT-offset positions
  (`cot_96`, `cot_128`).
- Stage1b and multilayer runs do not show this mismatch pattern.

## Hidden-Minus-Surface Delta CI

Fable judges the delta CI procedure methodologically acceptable:

- Hidden probes are validation-selected.
- Surface baselines are validation-selected.
- Bootstrap resampling is paired/grouped.
- Group-internal example-id alignment is implemented.
- All 16 items completed with `n_bootstrap_valid=2000`.

## Main Result Interpretation

Fable says all 16 hidden-minus-surface AUROC deltas being negative is a robust
negative result, not a marginal or ambiguous result.

Representative deltas:

| Source | Hidden | Surface | AUROC delta | 95% CI |
|---|---:|---:|---:|---|
| HB Stage1 linear | 0.8396 | 0.9655 | -0.1259 | [-0.1677, -0.0853] |
| RS Stage1 linear | 0.7025 | 0.9166 | -0.2141 | [-0.2492, -0.1814] |
| SR Stage1 linear | 0.8147 | 0.9371 | -0.1229 | [-0.1552, -0.0944] |
| WJB Stage1 linear | 0.8252 | 0.9181 | -0.0927 | [-0.1031, -0.0816] |

Interpretation from Fable:

- Current hidden probes do not extract signal superior to simple word/character
  n-gram surface baselines.
- This is a negative finding for the current probe design rather than a data
  quality failure.

## Remaining Blockers

Formal non-GPU blockers:

1. Human QA must be completed.
2. S-to-S safe-prompt diagnostics must be run.
3. HT quarantine/external testing must be completed before external claims.

## GPU Rerun Guidance

Fable says no GPU rerun is required to confirm the current finding.

GPU rerun is only needed if the probe design changes, for example different
layers, positions, architecture, or combination method.

## Recommended Decision

Fable says the next decision is:

1. Accept Stage1 as a negative/control result and revise claims accordingly.
2. Or redesign the probe and rerun GPU Stage1 for that new design.

Fable's final blocker summary:

> Human QA + S-to-S + HT quarantine. No GPU work remains unless design changes.
