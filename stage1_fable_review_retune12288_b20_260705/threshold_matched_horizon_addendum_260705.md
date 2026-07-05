# Stage1 Threshold Accuracy and Matched-Horizon Baseline Addendum

Date: 2026-07-05

This addendum is sanitized. It contains aggregate metrics and methodological
questions only. It excludes raw prompts, raw CoTs, completions, and row-level
prediction JSONL.

## Context

The previous Fable-5 review concluded `ONLY AFTER GATES`:

- Keep the current Stage1 run as a negative/control result for the current
  linear hidden-probe design.
- Do not run another GPU sweep of the same position/layer/classifier design.
- The main issue is the evaluation contrast: hidden probes use a prefix-limited
  hidden snapshot, while surface baselines use full-trajectory hindsight text.
- The approved next step is CPU-only matched-horizon reanalysis on frozen data.

The user now asks two follow-up questions.

## Current Thresholded Metrics

The validation-selected hidden probes have non-trivial test AUROC but conservative
thresholded operating points. The table below is from
`val_fixed/val_fixed_probe_report.tsv`.

For approximate balanced accuracy, use:

```text
balanced_accuracy = (recall + specificity) / 2
specificity = 1 - FPR
```

### Stage1 linear

| Source | Position/layer | Test AUROC | Test recall | Test FPR | Approx balanced accuracy |
|---|---|---:|---:|---:|---:|
| HarmBench | cot_128 / layer 12 | 0.839595 | 0.427632 | 0.072368 | 0.677632 |
| ReasoningShield | cot_9 / layer 12 | 0.702548 | 0.188060 | 0.050746 | 0.568657 |
| StrongReject | cot_96 / layer 17 | 0.814668 | 0.438406 | 0.083033 | 0.677687 |
| WildJailbreak | cot_128 / layer 14 | 0.825248 | 0.393264 | 0.066501 | 0.663381 |

### Stage1b linear

| Source | Position/layer | Test AUROC | Test recall | Test FPR | Approx balanced accuracy |
|---|---|---:|---:|---:|---:|
| HarmBench | cot_4 / layer 20 | 0.810098 | 0.256579 | 0.072368 | 0.592105 |
| ReasoningShield | cot_7 / layer 32 | 0.679216 | 0.038806 | 0.008955 | 0.514925 |
| StrongReject | cot_4 / layer 20 | 0.746113 | 0.328520 | 0.104693 | 0.611914 |
| WildJailbreak | cot_4 / layer 18 | 0.781790 | 0.409609 | 0.139178 | 0.635216 |

Interpretation so far:

- The hidden probes are not random; AUROC is often 0.78-0.84.
- Thresholded balanced accuracy is much weaker because the chosen threshold
  yields low FPR but also low recall.
- We need to know whether this is a threshold-selection/calibration problem
  that can be fixed on validation data, or whether it reflects poor score
  separation that cannot be rescued without changing the probe/evaluation.

## User Questions

### Q1. Can thresholded accuracy be improved?

Please analyze whether the thresholded balanced accuracy can be improved
without changing the hidden representation or running new GPU extraction.

Please answer:

1. Is it scientifically legitimate to reselect thresholds on validation to
   optimize balanced accuracy, Youden's J, macro-F1, or a target FPR?
2. Would this likely improve test balanced accuracy given the current AUROC and
   recall/FPR pattern?
3. What exact threshold protocol should we use to avoid test leakage and avoid
   overfitting?
4. Should thresholds be global across sources, per held-out source, or selected
   only from train/validation sources in LOSO?
5. Should we use Platt scaling, isotonic calibration, temperature scaling, or
   simple threshold sweeping? Which is most defensible here?
6. What diagnostic tables/plots should be produced:
   - ROC curves
   - PR curves
   - validation-selected balanced accuracy
   - test balanced accuracy with validation threshold
   - oracle test threshold only as diagnostic
   - confidence intervals
7. What would constitute a meaningful improvement versus just moving along the
   ROC curve?

### Q2. How should hidden prefix and full-trajectory text be compared fairly?

The concern is that the current comparison is hidden prefix vs full-trajectory
hindsight text. The user asks whether we can put them at the same position or
same information horizon, for example by truncating the full trajectory text to
the same position as the hidden prefix.

Please answer:

1. Is matched-horizon truncation the right fix?
2. How exactly should `text@k` be constructed so it sees only the same prefix
   that `hidden@k` can depend on?
3. Should `text@k` include:
   - prompt only?
   - prompt + first k CoT tokens?
   - first k generated tokens only?
   - position-indexed tokens?
   - length-so-far / ended-by-k indicators?
4. Should `k` be defined in model tokens, whitespace tokens, or generated CoT
   positions already used by the hidden extractor?
5. How do we handle examples shorter than k so that censoring does not favor
   either arm?
6. Should we try to match feature dimension/capacity, or is matched information
   horizon more important than dimensionality?
7. What are the strongest fair surface baselines at matched horizon:
   - word/char n-grams over prefix
   - position-indexed token identities
   - sentence embeddings of prefix
   - prefix length / ended-by-k
   - small frozen LM embedding probe
8. Should full-trajectory surface baselines remain as hindsight ceilings rather
   than direct competitors?
9. What exact statistical comparison should be reported:
   - paired delta AUROC
   - within-pair ranking accuracy
   - residual delta log-loss
   - group bootstrap by pair/source
   - Holm correction over k?

### Q3. Prior work / literature analogues

Please identify prior research areas or specific papers that faced analogous
issues:

- probing classifiers with high AUROC but poor thresholded accuracy/calibration;
- avoiding test leakage in threshold/calibration selection;
- comparing internal representations against surface/text baselines with
  matched controls;
- avoiding "hindsight" baselines when the representation is prefix-limited or
  time-local;
- using control tasks/selectivity/MDL/residualization for probes;
- early prediction or incremental classification where baselines must be
  truncated to the same prefix.

For each reference, briefly state:

- what problem they encountered;
- what methodological control they used;
- what lesson applies to our Stage1 design.

Do not invent citations. If unsure about a citation, mark it as uncertain and
separate it from the actionable methodological advice.

## Requested Output

Please provide a decisive methods memo with:

1. Direct answer to Q1: can thresholded accuracy be improved, and how?
2. Direct answer to Q2: exact matched-horizon design.
3. A short preregistered analysis protocol we can implement on CPU.
4. A list of literature analogues and lessons.
5. A do-not-do list.
6. A final recommendation: `THRESHOLD_REANALYSIS_ONLY`,
   `MATCHED_HORIZON_REANALYSIS`, `BOTH_CPU_ONLY`, or `NO_ACTION`.
