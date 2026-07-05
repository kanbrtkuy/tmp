# Stage1 Excluded-Source Lead-Time Confirmation Prereg Plan

Date: 2026-07-05

Status: reviewed by Fable-5 on 2026-07-05 (EDITS_REQUIRED, applied verbatim
below) -> OK_TO_IMPLEMENT_PLAN_ONLY. This is not an equal-horizon rescue
variant. Equal-horizon iteration remains closed.

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

## Prior Exposure And Power Disclosure

- These sources are excluded from the A1/A2 lead-time analyses, not pristine:
  layer 28 and the `char_tfidf` surface family were originally selected using
  validation data that included `strongreject_full` and `reasoningshield` val
  splits, and full-trajectory (non-matched-horizon) hidden-vs-surface summary
  numbers for these sources' test splits have already been computed and seen.
  This confirmation therefore tests generalization of the k=4 lead-time
  pattern to sources unseen by the lead-time analyses, not to never-touched
  data. This disclosure must be repeated in any writeup of the result.
- Power: the frozen splits imply roughly ~612 pooled test pairs (~277
  strongreject_full + ~335 reasoningshield), vs 2171 in the original
  diagnostic. Expected delta-AUROC CI half-width is ~±0.02 (vs ~±0.01 at
  n=2171). The A1 point estimate (+0.056) is detectable at this n; effects
  below ~0.02 are not. A null here is evidence against a +0.05-scale effect,
  not against any effect.

## Fixed Protocol

Sources:

- `strongreject_full`
- `reasoningshield`

Splits:

- Use the existing frozen splits from
  `loso_freeze_fixed_budget_samples_000_099/stage1_prepared/{source}/cotpause/`.
- Frozen evaluation population: the set of test pairs that are pair-complete
  at ALL k in the grid (4..64), fixed once before any scoring and identical
  across both recipes and both arms at every k. No per-k population changes.
- Fail-closed alignment guards: any hidden/surface row misalignment or pair
  loss raises a hard error; silent drops are disallowed.
- Minimum-power halt: if the frozen population has fewer than 150 test pairs
  for either source, halt before scoring and report data insufficiency; do
  not run a reduced-n confirmation.
- Do not re-split, resample, or change source budgets.

k grid:

- `4,8,16,32,64`

Surface arm:

- Matched-horizon `char_tfidf` text@k only.
- Use the exact Module M configuration: same vectorizer hyperparameters,
  tokenizer, truncation semantics, and estimator settings, pinned by
  reference to the archived Module M config JSON in
  `res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md`
  (see Hidden recipes below). No "same as Module M where possible" latitude.
- No surface-family reselection.

Hidden recipes:

1. **A1-compatible score-pooling recipe**
   - layer fixed to 28.
   - train single-position linear probes at each `cot_k` using the exact
     original Stage1 probe pipeline recipe (same estimator class,
     hyperparameters, preprocessing, and score convention as the run that
     produced the A1 inputs). "Where possible" language is disallowed: before
     code review, a written amendment to this plan must pin both configs by
     reference to the archived config JSONs of
     `stage1_post_hb_260705_after_hb_n100_loso` (probe pipeline) and Module M
     (`char_tfidf`). This amendment is
     `res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md`.
     Any deviation forced by the codebase must be listed in the amendment
     before execution, not discovered after.
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
- Percentile bootstrap CIs (95%).
- Report p=0.0 as `p < 0.002`.
- Report per-source rows and pooled rows.
- Pooled rows: z-score BOTH the hidden and surface score arms per source
  using validation-split statistics before cross-source concatenation (same
  rule as the A1 round-1 blocker fix and the A2 implementation). Raw-score
  cross-source pooling is disallowed.
- No Holm/multiplicity correction is applied or needed: there is exactly one
  primary confirmatory cell (pooled A1 hidden@4 minus text@8). All other
  estimands are secondary or descriptive and must be labeled as such wherever
  reported.
- Publish full hidden@k vs text@k' lead-time matrices for both recipes,
  descriptive-only.

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
- A2 robustness gate: pooled A2 hidden@4 minus text@8 CI high >= 0 AND point
  estimate >= -0.01. (CI-high-only is a non-inferiority criterion that an
  underpowered run passes by default; the point-estimate floor prevents a
  low-power pass.)

Interpretation:

- If all three gates pass: this is the only outcome permitted to use the word
  "confirmed" (as "preregistered lead-time confirmation"). Still do not
  reopen the equal-horizon claim.
- If A1 passes but A2 robustness fails: the result is "replicated but
  recipe-sensitive"; it stays labeled exploratory in the paper, and the word
  "confirmed" may not be used anywhere for it.
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
- No pooling the excluded sources with `harmbench_standard` /
  `wildjailbreak_vanilla_harmful` into a 4-source pooled row to rescue a weak
  excluded-source result; the confirmation population is the two excluded
  sources only.
- No sample-size escalation after seeing results (no added prompts, pairs,
  rollouts, or seeds to move a CI across a gate boundary).
- No promoting any secondary estimand (text@16/text@32 deltas, absolute-AUROC
  comparisons) to primary after results are seen.
- Same-horizon diagonal cells (hidden@k minus text@k) are descriptive-only;
  the equal-horizon thread stays closed regardless of what they show.
- No file-drawer: whichever branch fires, the outcome is committed to the
  registry and reported wherever the k=4 lead-time diagnostic is mentioned.

## Execution Order And Provenance

- This plan (with the required edits and the config-pinning amendment) must
  be committed to the tmp registry BEFORE any hidden extraction or scoring;
  that commit hash is the preregistration timestamp and must be quoted in the
  results packet.
- The run script must write the preregistered estimands, gates, and decision
  rules into its config/output JSON before the fit/score loop executes, so
  the preregistration is auditable from the artifact itself.
- Run conditions (same as A2): non-null `--code-commit`, `--fail-on-error`
  set, and the run is accepted only with exit code 0 and `n_errors=0`. A run
  violating any of these is INVALID and may not be interpreted; fixes go
  through code re-review, not quiet reruns.

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

Fable-5 review outcome (2026-07-05): `OK_TO_IMPLEMENT_PLAN_ONLY` after
verbatim application of the required edits. Ruled an allowed excluded-source
confirmation (pre-flagged before A2 ran; off-diagonal estimands; both
recipes; accepts negative outcomes), not an equal-horizon rescue. The
requested hidden extraction is acceptable in extract-minimal form only:
teacher-forced replay of the frozen splits, layer 28 and positions
`cot_4,cot_8,cot_16,cot_32,cot_64` — no additional layers or positions. No
plan re-review is needed after these edits; the config-pinning amendment
(Edit 1) is verified at code review. Code must still pass a separate Fable-5
code review before any RunPod execution.
