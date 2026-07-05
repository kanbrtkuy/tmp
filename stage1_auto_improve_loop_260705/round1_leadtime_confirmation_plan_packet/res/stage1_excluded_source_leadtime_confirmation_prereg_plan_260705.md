# Stage1 Excluded-Source Lead-Time Confirmation Prereg Plan

Date: 2026-07-05

Status: draft for Fable-5 review before any code or run. This is not an
equal-horizon rescue variant. Equal-horizon iteration remains closed.

## Motivation

Fable-5 closed the teacher-forced equal-horizon thread after A2:

- A2 valid run.
- k=8 failure branch fired.
- No A2b or further equal-horizon probe variants.

The only Stage1 experiment Fable-5 left open was an optional, separately
preregistered confirmation of the exploratory k=4 lead-time diagnostic on the
excluded sources `strongreject_full` and `reasoningshield`, running both the A1
score-pooling recipe and the A2 feature-pooling recipe and accepting either
outcome.

This plan specifies that optional confirmation. It should be run only if
Fable-5 agrees that the plan is not a forbidden rescue path and if the paper
needs lead-time as more than an exploratory diagnostic.

## Current Data Readiness

Checked on RunPod, without printing raw prompts or CoTs:

| Source | train rows | val rows | test rows | prepared splits | matched-horizon predictions | dense hidden archive |
|---|---:|---:|---:|---|---|---|
| `strongreject_full` | 1862 | 208 | 554 | present | not present for A1/A2 k-grid | not present |
| `reasoningshield` | 1758 | 196 | 670 | present | not present for A1/A2 k-grid | not present |

Existing full-text baseline outputs exist under
`text_baseline_predictions_retune12288_b20`, but these are not the
matched-horizon text@k scores needed for the confirmation. Existing
hidden-vs-surface delta summaries also include these sources, but they are
validation-selected full Stage1 summaries and are not the lead-time
confirmation.

Therefore, if this plan is approved, the first executable step is a readiness
audit and artifact generation plan, not interpretation.

## Claims And Anti-Claims

Primary confirmatory question:

> Does the k=4 lead-time pattern seen in the A1 diagnostic replicate on the two
> excluded sources under a preregistered, fixed recipe?

Supporting question:

> Is any replicated lead-time pattern robust to the A2 feature-level refit
> recipe, or is it still recipe-sensitive?

Anti-claim:

> This is not an attempt to resurrect the closed equal-horizon hidden-over-text
> claim. A negative or mixed result must be accepted and reported.

## Fixed Protocol

Sources:

- `strongreject_full`
- `reasoningshield`

Splits:

- Use the existing frozen splits from
  `loso_freeze_fixed_budget_samples_000_099/stage1_prepared/{source}/cotpause/`.
- Preserve pair completeness at each k.
- Do not re-split, resample, or change source budgets.

k grid:

- `4,8,16,32,64`

Surface arm:

- Matched-horizon `char_tfidf` text@k only.
- Use the same tokenizer and pair-complete truncation semantics as Module M.
- No surface-family reselection.

Hidden recipes:

1. **A1-compatible score-pooling recipe**
   - layer fixed to 28.
   - train single-position linear probes at each `cot_k` using the original
     Stage1 probe pipeline recipe where possible.
   - convert per-position validation/test hidden scores to A1-style cumulative
     scores: unweighted mean over validation-z-scored hidden scores for
     positions `j <= k`.
   - no learned weights, max pooling, new layer search, or hyperparameter
     tuning.

2. **A2 feature-pooling recipe**
   - layer fixed to 28.
   - mean-pool hidden vectors over `cot_j` where `j <= k`.
   - refit `StandardScaler + LogisticRegression(class_weight=balanced)` per k
     on the training split only.
   - compare against the same frozen matched-horizon text@k scores.

Hidden extraction:

- If dense hidden archives for these sources do not exist, extract only the
  fixed layer/position set required for this plan.
- Required positions: `cot_4,cot_8,cot_16,cot_32,cot_64`.
- Required layer: 28.
- Store manifests and metadata ids/labels only in review packets; do not send
  raw text, prompts, CoTs, hidden arrays, or generated pairs to Fable-5.

Bootstrap and statistics:

- Paired bootstrap over pair/match-family groups, `B=500`, seed `260705`.
- Report p=0.0 as `p < 0.002`.
- Report per-source rows and pooled rows.
- Publish full hidden@k vs text@k' lead-time matrices for both recipes.

## Preregistered Estimands

Primary A1 diagnostic estimand:

- pooled A1-compatible hidden@4 minus text@8 delta AUROC.

Secondary A1 diagnostic estimands:

- pooled A1-compatible hidden@4 minus text@16 delta AUROC.
- pooled A1-compatible hidden@4 minus text@32 delta AUROC.
- absolute AUROC comparison among hidden@4, text@16, and text@32.

A2 robustness estimand:

- pooled A2 hidden@4 minus text@8 delta AUROC.

Descriptive-only matrices:

- all hidden@k minus text@k' cells for k,k' in `4,8,16,32,64`.

## Decision Rules

Positive lead-time confirmation:

- A1 primary gate: pooled A1 hidden@4 minus text@8 CI low >= 0.
- Directional per-source sanity: neither source has A1 hidden@4 minus text@8
  point estimate < -0.02.
- A2 robustness gate: pooled A2 hidden@4 minus text@8 CI high >= 0.

Interpretation:

- If all three gates pass: report a recipe-aware lead-time diagnostic
  confirmation. Still do not reopen the equal-horizon claim.
- If A1 passes but A2 robustness fails: report A1 lead-time as replicated but
  recipe-sensitive; keep it exploratory.
- If A1 primary fails: drop the lead-time claim from the main story.
- If per-source sanity fails: report heterogeneity; no pooled-only headline.

These rules intentionally allow a negative outcome and forbid searching for a
different cell after seeing results.

## Disallowed Actions

- No new equal-horizon variants.
- No layer, classifier, k-grid, pooling-rule, normalization, or
  surface-family search.
- No A2b.
- No switching the headline to pair-rank accuracy if AUROC fails.
- No cell shopping in the lead-time matrix.
- No reporting pooled-only success if one source is directionally negative by
  more than the preregistered sanity margin.

## Required Fable-5 Review Before Execution

Before code is written or GPU hidden extraction is launched, ask Fable-5 to
review this plan and decide:

- Is this an allowed excluded-source lead-time confirmation, or a disguised
  equal-horizon rescue?
- Are the estimands and decision rules sufficiently fixed?
- Should the A2 robustness gate be required, weakened, or treated purely as a
  reporting constraint?
- Is the required hidden extraction acceptable, given dense archives are not
  currently present for these sources?

Only after `OK_TO_IMPLEMENT` should code be written. Only after code review
should RunPod execution start.
