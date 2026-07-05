# Fable-5 Review Request: Stage1-Only Pause Position Choice

Please act as a strict senior ML methods reviewer.

Read `STAGE1_ONLY_EVIDENCE_SUMMARY.md` first. The packet is sanitized and
contains only aggregate information.

## Critical Scope Constraint

For this review, please **ignore all Stage2, Stage3, and Stage4 outcome
evidence**.

Do not use any of the following as evidence:

- pause-SFT checkpoint behavior,
- format-only vs full-SFT comparisons,
- capability or safety metrics from pause-token models,
- Stage3 pause-probe heatmaps,
- Stage4 steering outcomes.

We want a clean answer to:

> Based only on the latest Stage1 results, where should pause tokens be
> inserted, if anywhere?

## Core Local Interpretation To Review

Our current Stage1-only interpretation is:

```text
If we must choose a Stage2 pause candidate from Stage1 only, choose k=4:
insert 3 pause tokens before cot_4, i.e. after four non-whitespace CoT tokens
and before the fifth.

But this should be framed as a weak/exploratory candidate, not a confirmed
optimal position.
```

Reason:

- matched-horizon hidden-minus-text delta is positive only at `k=4`
  (`pooled +0.0584`);
- `k=8` is approximately zero/negative and `k>=16` is strongly negative;
- A1 score pooling gives an early story, but A2 feature pooling largely removes
  it;
- excluded-source lead-time confirmation fails;
- high-offset positions have coverage/censoring issues and are more like
  hindsight readouts than early pause ports.

## Questions

1. Under the strict Stage1-only scope, is `before cot_4` the right main
   exploratory pause-position candidate, or should we instead choose:
   - before `cot_3`,
   - after `cot_4` / before `cot_5`,
   - `cot_8`,
   - a later position such as `cot_16`, `cot_64`, or `cot_120`,
   - multiple positions,
   - or no Stage1-derived position at all?

2. How should Stage1 `hidden@cot_4` map to an actual pause insertion point?
   In particular, does a signal measured at the natural `cot_4` hidden state
   justify inserting pause tokens before `cot_4`, or would a neighboring
   position be more causally aligned?

3. Does the latest Stage1 evidence actually authorize any pause-port choice, or
   should the honest conclusion be "Stage1 does not identify a reliable pause
   position; if Stage2 proceeds, it is exploratory/on-policy rather than
   Stage1-validated"?

4. How should we treat the fact that some late Stage1 readouts can be stronger
   than early positions? Is that ever a reason to use late pause placement, or
   does the matched-horizon/lead-time framing rule that out for the mainline?

5. What exact wording is safe for a paper/report if we choose `before cot_4`
   from Stage1-only evidence?

6. What minimal follow-up would convert this from a weak exploratory choice to
   a stronger evidence-backed choice?
   - CPU-only analysis?
   - fresh preregistered Stage1?
   - on-policy Stage3 matched-horizon confirmation?
   - or something else?

7. Please give a decision label:
   - `COT4_WEAK_EXPLORATORY`,
   - `NO_STAGE1_DERIVED_POSITION`,
   - `LATE_POSITION_STAGE1_ONLY`,
   - `MULTI_POSITION_ABLATION_FIRST`,
   - or another explicit label.

## Desired Output

Please produce:

- Executive verdict.
- Position recommendation table.
- Exact answer to "after `<think>`, which token position?"
- Explanation of hidden@cot4 vs insertion-before-cot4 timing.
- Safe wording / do-not-claim list.
- Minimal follow-up needed before stronger claims.
- Final decision label.

