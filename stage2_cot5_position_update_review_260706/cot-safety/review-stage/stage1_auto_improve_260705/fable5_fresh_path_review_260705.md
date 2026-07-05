# Fable-5 Review: Fresh Path After Stage1 Negative Result

Reviewer: `claude-fable-5`

Verdict:

```text
FRESH_PREREG_ONLY
STOP_CURRENT_STAGE1 for the frozen test set.
CODE_AND_RUN_ALLOWED is rejected for current frozen Stage1.
```

## Decision

No improvement-seeking code or run on the current frozen Stage1 test set is
scientifically valid. The only admissible positive path is a fresh, separately
preregistered Stage2/on-policy experiment.

On the frozen Stage1 set, allowed work is limited to:

- operating-point reporting;
- integrity/audit completion for the negative claim;
- sensitivity/power statement of what effect sizes the negative excludes;
- documentation of the row-audit coverage gap;
- write-up.

None of that should be described as "making Stage1 better."

## Why

- The confirmatory budget is spent: main Stage1/Stage1b, threshold reanalysis,
  matched-horizon, A1, A2, and excluded-source confirmation have all been run.
- The failure is not marginal:
  - all 16 hidden-minus-surface deltas are negative;
  - length-only beats hidden on all four sources;
  - matched-horizon is negative for k >= 8;
  - the k=4 hint shrank under A2 and to null in excluded-source confirmation.
- The preregistered stop rules fired:
  - `a1_fail -> drop_leadtime_claim`;
  - M1 kill criterion permanently stops Stage1 probing when hidden@k - text@k
    is not CI-separated positive at k <= 32.
- The conditional GPU permission is void because the Phase-1 continue criteria
  failed.

## Important Nuance

Fable explicitly distinguishes the weak positive from the failed strong claim:

- Hidden probes are above chance, with AUROC around `0.68-0.84`.
- This is a real cross-source decodable signal.
- The failed claim is superiority over matched/full surface baselines and
  length controls.

Supported framing:

> Stage1 shows linear prefix-hidden probes contain decodable safe/unsafe signal,
> but in this teacher-forced natural-pair setting they do not beat matched or
> full-trajectory surface baselines.

Unsupported framing:

> Stage1 demonstrates hidden-state superiority or stable lead-time advantage.

## Minimum Valid Next Experiment

Only if the user wants a new positive path:

- Stage: Stage2/on-policy, not another Stage1 rescue.
- Data: fresh prompts and fresh on-policy rollouts, frozen and hash-committed
  before hidden extraction.
- Primary endpoint: one preregistered early horizon k* in `{4, 8}`, comparing
  hidden@k* to matched-capacity surface@k* by paired/clustered delta AUROC.
- Gate: bootstrap CI-low > 0.
- Stop: if CI includes 0, stop the hidden-vs-surface early-warning line
  program-wide. No secondary rescue or horizon shopping.

## What Not To Do

- No new probe families, layers, positions, classifier heads, pooling schemes,
  calibration variants, nonlinear rescues, subgroups, re-splits, or reweights on
  the frozen Stage1 test set.
- No GPU regeneration of RS/SR hidden arrays.
- No threshold/BA correction phrased as improvement.
- No informal Stage2 peeking before preregistration and pipeline review.

## Bottom Line

Current frozen Stage1 is closed. Any continuation should be a new
preregistered Stage2/on-policy objective, not an extension of the Stage1
improvement loop.
