# Fable-5 Review Brief: Is Any Non-Fishing Stage1 Improvement Path Left?

Date: 2026-07-05

This packet is sanitized. It contains only aggregate documents, prior Fable
reviews, and plan/status notes. It excludes raw prompts, raw CoTs, completions,
hidden arrays, and row-level hidden features.

## Background

The user originally asked for an iterative loop:

1. send Stage1 results to Fable;
2. ask whether there is a better method;
3. write code if needed;
4. get Fable code review;
5. run the experiment;
6. send results back to Fable;
7. continue until Stage1 improves.

We completed that loop for the current frozen Stage1 setting:

- main post-HB LOSO Stage1/Stage1b;
- threshold/calibration reanalysis;
- matched-horizon reanalysis;
- A1 score-pooling diagnostic;
- A2 feature-level pooling confirmation;
- excluded-source lead-time confirmation on `strongreject_full` and
  `reasoningshield`.

Your prior review concluded:

```text
drop_leadtime_claim is correct.
Stage1 should now be treated as a negative/control result.
Further re-analysis of the frozen test set to find a positive cell would be
post-hoc fishing / multiplicity laundering.
```

## Evidence Files

Please read these files in this packet:

- `res/stage1_experiment_inventory_results_260705_zh.md`
- `res/stage1_post_hb_retune12288_b20_results_260705_zh.md`
- `review/fable5_excluded_leadtime_results_review_260705.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`

## Review Questions

Please answer all questions directly:

1. Given the completed Stage1 loop, is there any remaining analysis or code
   change on the current frozen Stage1 test set that would be a legitimate
   attempt to "make Stage1 results better," rather than post-hoc fishing?
2. If no, what is the correct stopping rule and how should we phrase it to the
   user?
3. If a fresh preregistered follow-up is scientifically justified, specify the
   minimum valid next experiment:
   - Is it still Stage1, or should it be Stage2/on-policy?
   - What is the primary endpoint?
   - What data must be fresh or held out?
   - What code should be written before seeing results?
   - What gates would stop the loop?
4. Are threshold/balanced-accuracy improvements allowed as operating-point
   reporting only, without changing the negative AUROC conclusion?
5. Should the active improvement loop be marked complete, blocked, or continued
   into a new preregistered objective?

## Requested Output

Use this structure:

- Verdict: `STOP_CURRENT_STAGE1` / `FRESH_PREREG_ONLY` /
  `CODE_AND_RUN_ALLOWED`
- Rationale
- Minimum next experiment, if any
- Code to write, if any
- What not to do
- User-facing summary in plain language
