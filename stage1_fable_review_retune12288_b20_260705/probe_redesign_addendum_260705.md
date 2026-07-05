# Stage1 Probe Redesign Addendum

Date: 2026-07-05

This addendum asks Fable to move past the decision of whether current Stage1 is
negative/control, and instead design a scientifically credible way to fix or
replace the current probe method.

The packet remains sanitized: no raw prompts, raw CoTs, completions, or
row-level prediction JSONL are included.

## Starting Point

Fable's previous decision review concluded:

- Current Stage1 should be accepted as a negative/control result for the
  current hidden-probe design.
- Do not spend more GPU on the same design.
- Poor performance is more likely probe/method architecture than data quality.
- Approximate attribution:
  - Data problem: 10-15%
  - Probe/method problem: 70-75%
  - Claims/expectation problem: 10-15%

## What Failed

The current hidden-probe design:

- Extracts hidden states from `DeepSeek-R1-Distill-Llama-8B`.
- Uses frozen LOSO splits over HarmBench, ReasoningShield, StrongReject, and
  WildJailbreak.
- Trains mostly linear hidden-state probes over selected CoT positions/layers.
- Includes Stage1 and Stage1b variants plus simple multilayer combinations.
- Uses validation-selected probe selection, then reports test metrics.
- Compares against validation-selected surface text baselines.

Observed failure:

- All 16 hidden-minus-surface AUROC deltas are negative.
- Surface baselines on the same frozen splits are strong, roughly 0.91-0.97
  test AUROC.
- Validation-selected hidden probes are lower, roughly 0.70-0.84 test AUROC.
- Row audit does not show broad data corruption; remaining mismatches are
  localized to Stage1 linear high-CoT-offset positions (`cot_96`, `cot_128`).

Representative hidden-minus-surface AUROC deltas:

- HB Stage1 linear: -0.1259, CI [-0.1677, -0.0853]
- RS Stage1 linear: -0.2141, CI [-0.2492, -0.1814]
- SR Stage1 linear: -0.1229, CI [-0.1552, -0.0944]
- WJB Stage1 linear: -0.0927, CI [-0.1031, -0.0816]

## Core Question

If the current simple linear hidden-state probe is the wrong method, what is a
scientifically justified replacement?

Please do not merely propose "try more layers/positions/classifiers." The goal
is to identify a principled method that answers the original scientific
question more directly:

> Do model internal states contain safety-relevant trajectory information that
> is not reducible to easy surface text artifacts?

## Constraints

- The existing frozen Stage1 data should be reused if possible.
- Any new GPU run must be gated by non-GPU audits.
- No test-set hyperparameter fishing.
- Surface baselines are very strong and must remain part of the evaluation.
- Human QA, S-to-S diagnostics, and HT quarantine/external testing remain
  formal blockers for claims.
- GPU budget should be treated as expensive: propose the minimum credible
  redesigned experiment first.

## Specific Questions For Fable Pro Review

Please answer as a strict senior ML methods reviewer.

1. What is the most likely methodological reason the current hidden probe lost
   to surface baselines? Be concrete: representation choice, objective, label
   definition, positions/layers, classifier class, paired structure, LOSO
   mismatch, or something else?
2. What probe redesign would most directly test for latent safety information
   rather than surface text artifacts?
3. Should the next probe be:
   - a paired contrastive/difference probe within the same prompt,
   - a residualized probe that removes surface-text baseline signal,
   - a nonlinear probe,
   - a sequence/trajectory-level probe,
   - a causal/activation intervention diagnostic,
   - an early-position probe,
   - or something else?
4. What is the minimum GPU experiment that is worth running if we decide to
   redesign? Specify:
   - data inputs,
   - hidden positions/layers,
   - representation aggregation,
   - model/probe objective,
   - validation protocol,
   - required controls,
   - expected failure modes.
5. Which proposed changes are scientifically motivated, and which would count
   as hyperparameter fishing?
6. How should we compare the redesigned hidden probe to surface baselines?
   Should we use AUROC deltas, residual evaluation, matched truncation,
   adversarial surface controls, or another metric?
7. What non-GPU gates must pass before spending GPU?
8. What result would justify continuing Stage1 as a research direction, and
   what result should kill it?
9. If the answer is "do not redesign Stage1; move to Stage2/Stage3 instead,"
   explain why and what evidence would make that defensible.

Please end with:

- a ranked list of recommended probe redesigns,
- one minimal runnable experiment plan,
- one "do not do this" list,
- and a final decision: `REDESIGN`, `DO NOT REDESIGN`, or `ONLY AFTER GATES`.
