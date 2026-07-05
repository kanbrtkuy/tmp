# CLAUDE_FABLE5_STAGE1_ONLY_PAUSE_POSITION_REVIEW

Reviewer: `claude-fable-5`  
Date: 2026-07-06  
Scope: Stage1-only. All Stage2/3/4 outcome evidence excluded, per request.
Aggregate evidence only; no raw prompts/CoTs/completions/row-level predictions
inspected.

## 1. Executive Verdict

The team's local interpretation is essentially correct and appropriately
humble, and I endorse it with two sharpenings.

**What Stage1 supports:** if a single Stage2 pause candidate must be named, the
early window around `cot_4` is the only defensible choice. It is the only
horizon where the equal-horizon hidden-minus-text delta is positive (pooled
`+0.0584`; positive in both dev sources), the only k where the sign survives
the conservative A2 re-fit (`+0.0041`), and the only region compatible with a
prevention / lead-time motivation.

**What Stage1 does not support:** any claim that this position is validated,
optimal, significant, or generalizing. The `k=4` point is a max over a scanned
k-grid (winner's-curse exposure), the packet reports no CI for the `+0.0584`
primary delta despite `B=500`, A2 shrinks the effect to practical zero, and the
only out-of-sample confirmation designed for the early story - the
excluded-source lead-time gate - failed on both arms (A1 CI straddles zero; A2
significantly negative), yielding `decision = drop_leadtime_claim`. Under
strict standards, Stage1 authorizes an **engineering default for an exploratory
Stage2**, nothing more. For claims purposes, this is functionally equivalent to
"no Stage1-derived position."

One technical correction is required on token semantics (Section 4):
`hidden@cot_4` is measured at a state that has already consumed token `cot_4`;
insertion **before** `cot_4` is one token earlier than the measured state. This
must either be fixed (insert before `cot_5`) or explicitly documented as a
within-window approximation.

## 2. Position Recommendation Table

| Candidate | Verdict | Reasoning (Stage1-only) |
|---|---|---|
| before `cot_3` | Reject | No evidence isolating it; would be arbitrary; adds a forking path. |
| before `cot_4` | **Acceptable exploratory default** | Only positive matched-horizon k; sign consistent across A1/A2 and both dev sources; but off-by-one vs. the measured state. |
| after `cot_4` / before `cot_5` | **Most causally aligned with hidden@cot_4** | First insertion point at which the state that carried the probed signal is fully formed; register as the single permitted sibling variant. |
| `cot_8` | Reject | Matched-horizon approximately zero/negative; A2 negative (`-0.0235`); A1 k=8 effect not confirmed by A2. |
| `cot_16` / `cot_64` / `cot_120` | Reject for mainline | Strongly negative matched-horizon deltas; text wins at same horizon; coverage/censoring at high offsets; no lead time - readout, not prevention. At most a descriptive readout ablation with a censoring analysis. |
| Multiple positions (broad sweep) | Reject | Multiplies comparisons on evidence already too weak to rank positions; only a preregistered `{before cot_4, before cot_5}` pair is tolerable. |
| No Stage1-derived position | Correct **at the claims level** | Stage1 validates nothing; but Stage2 needs a default, and cot_4/cot_5 is the least-arbitrary one. |

## 3. Exact Answer: "After `<think>`, Which Token Position?"

Using the codebase convention (skip leading whitespace-only reasoning tokens
once; count non-whitespace CoT tokens as `cot_0, cot_1, ...`):

- **Team's current choice (`before cot_4`):** insert the pause tokens after
  **four** non-whitespace CoT tokens (`cot_0..cot_3`) and before the **fifth**:

```text
<think> t0 t1 t2 t3 <pause x3> t4 ...
```

- **Causally faithful mapping of `hidden@cot_4` (`before cot_5`):** insert
  after **five** non-whitespace CoT tokens (`cot_0..cot_4`) and before the
  **sixth**:

```text
<think> t0 t1 t2 t3 t4 <pause x3> t5 ...
```

Either is defensible as an exploratory default; Stage1 cannot adjudicate a
one-token difference. Pick one, preregister it, and do not switch post hoc. If
the criterion is fidelity to the measurement, `before cot_5` is the correct
answer; if it is tooling continuity, `before cot_4` is acceptable with the
caveat documented.

## 4. `hidden@cot_4` vs. Insertion-Before-`cot_4` Timing

The hidden state read at position `cot_4` is computed **with `cot_4` as the
input embedding at that position**, attending over `cot_0..cot_4`. It therefore
encodes information available only after the model has consumed five
non-whitespace CoT tokens. Inserting pauses *before* `cot_4` gives the model
only `cot_0..cot_3` - one token less context than the state the probe was fit
on. The first insertion point at which "the state that carried the Stage1
signal" fully exists and can causally influence subsequent reasoning is
**immediately after `cot_4`, i.e. before `cot_5`**.

Two mitigating facts, both from the packet:

1. The A2 estimator pools layer-28 vectors over `cot_j, j<=k`, so the `k=4`
   signal is really a *window* signal over the first about 5 tokens, not a point
   measurement at `cot_4`. Insertion anywhere at the window's trailing edge
   (`before cot_4` or `before cot_5`) is within the region the pooled evidence
   covers.
2. At this effect size (`+0.004` to `+0.058`), a one-token shift is almost
   certainly below the resolution of the evidence.

Conclusion: `before cot_4` is not wrong, but it may **never** be described as
"the position where the Stage1 signal was measured." Also verify, as a cheap
code-level audit, whether `matched_text@k` covers `cot_0..cot_{k-1}` or
`cot_0..cot_k`. If the former while `hidden@k` includes `cot_k`, the
equal-horizon comparison itself has a one-token asymmetry favoring hidden. This
should be checked and stated.

## 5. Safe Wording / Do-Not-Claim List

**Safe template:**

> In an equal-horizon reanalysis on the two development sources, `k=4` was the
> only horizon with a positive hidden-minus-text AUROC delta (pooled `+0.058`).
> The effect shrank to `+0.004` under a conservative re-fit estimator, and a
> preregistered lead-time confirmation on two held-out sources failed
> (`decision = drop_leadtime_claim`). We therefore selected pause insertion at
> `[before cot_4 / before cot_5]` as an *exploratory engineering default* for
> Stage2. Stage1 does not validate this position; overall, Stage1 is a
> negative/control result for the tested linear hidden-probe design, with
> full-trajectory surface baselines, including a length-only baseline,
> outperforming the hidden probe on all sources.

**Do NOT claim:**

1. "`cot_4` is optimal / the best position" - it is the max over a scanned grid;
   winner's curse.
2. "`cot_4` has a lead-time advantage" - the excluded-source gate failed on
   both arms; the claim was formally dropped.
3. "Hidden states beat text baselines" - all 16 hidden-minus-full-surface deltas
   are negative; `length_only` beats the hidden probe on all four sources.
4. Statistical significance for the `+0.0584` delta - no CI is reported in this
   packet; do not imply one.
5. Cross-source generalization of the `k=4` effect - it was measured on
   HarmBench/WildJailbreak only; StrongReject's per-source lead-time delta was
   significantly negative.
6. "Pause insertion at `cot_4` will causally improve safety" - a Stage1 readout
   claim can never license a causal intervention claim.
7. That the insertion point is "where the signal was measured" - off-by-one.
8. Any test-set-maximum figure as a headline number.

## 6. Minimal Follow-Up Before Stronger Claims

Ordered by cost, respecting that the equal-horizon Stage1 branch is closed and
must not accrue rescue analyses:

1. **Descriptive CPU-only, no new gates (hours):** (a) report the
   already-computed bootstrap CIs for the matched-horizon `k=4` deltas
   (per-source and pooled); (b) verify the `hidden@k` vs `text@k` one-token
   horizon alignment; (c) verify the pooled-AUROC computation (pooled `+0.0584`
   exceeding both per-source deltas warrants a one-line sanity check). These
   are reporting/audit items, not rescues.
2. **The decisive upgrade - preregistered on-policy matched-horizon
   confirmation (Stage3-style):** measure the signal at the actual insertion
   point, on-policy, with the position, estimator, gate, and CI thresholds fixed
   in advance. Offline Stage1 slicing cannot upgrade this claim further; only
   on-policy evidence at the intervention site can.
3. **Alternative if new offline evidence is insisted upon:** a fresh, fully
   preregistered Stage1 on new data with exactly one primary hypothesis
   (same-horizon hidden-vs-text at `k=4`, A2 estimator, excluded-source
   population). Do not re-slice the existing frozen data.

Do **not** run a wide multi-position sweep; at most the preregistered
`{before cot_4, before cot_5}` pair.

## 7. Final Decision Label

```text
COT4_WEAK_EXPLORATORY
```

Rider: at the claims level this is equivalent to
`NO_STAGE1_DERIVED_POSITION` - Stage1 does not identify a reliable pause
position. `before cot_4` (or the causally aligned `before cot_5`) is authorized
only as an exploratory engineering default; any Stage2 built on it is
exploratory/on-policy, not Stage1-validated.

Review complete. Verdict: `COT4_WEAK_EXPLORATORY` with the off-by-one
correction (`hidden@cot_4` causally maps to insertion before `cot_5`, not before
`cot_4`) as the one substantive technical flag.

