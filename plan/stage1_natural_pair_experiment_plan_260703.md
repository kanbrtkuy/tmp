# Stage 1 Natural-Pair Experiment Plan - 2026-07-03

This note summarizes the updated Stage 1 plan after the OpenAI-rewrite A/B data exposed strong surface separability. It records which experiments have been run, which are diagnostic only, and what remains before making a calibrated Stage 1 claim.

## Goal

Stage 1 is meant to test whether safe vs unsafe reasoning trajectories are separable in model hidden states, beyond prompt-only or shallow text artifacts.

The current plan is a response to two concerns:

- In teacher-forced traces, early hidden states may classify the prompt rather than monitor the reasoning trajectory.
- Generalization across sources, formats, lengths, and generation styles must be checked before interpreting a high probe score as a safety-relevant latent signal.

## Dataset Tracks

| Track | Pair definition | Current role | Status |
|---|---|---|---|
| A prime OpenAI rewrite | OpenAI safe rewrite vs original/open unsafe side | Confound diagnostic only | Completed; not a primary Stage 1 claim dataset |
| Natural 8B generated/generated | Same prompt, R1-8B generated safe CoT vs R1-8B generated unsafe CoT | Main natural-pair pilot | Completed for surface baselines and multiple hidden-probe scans, including dense 8B-extractor rerun |
| Natural 8B generated-safe/original-unsafe | R1-8B generated safe CoT vs original unsafe reference CoT | Robustness check against original dataset trajectories | Surface baselines completed; hidden probe can still be added if we need a direct 8B safe/original comparison |
| Natural 32B generated/generated | Same prompt, R1-32B generated safe CoT vs R1-32B generated unsafe CoT | Model-scale/generator-scale diagnostic | Dense 32B-hidden and 8B-hidden scans completed on the frozen 359-pair export |
| Natural 32B generated-safe/original-unsafe | R1-32B generated safe CoT vs original unsafe reference CoT | Robustness experiment against original unsafe references | Stage1/Stage1b with 32B hidden extractor completed on the frozen 555-pair export |

The 32B generation inventory continued after the frozen exports. The latest judged inventory observed was 406 generated/generated quality-pass pairs and 607 generated-safe/original-unsafe pairs.

## Generator-vs-Extractor Matrix

The natural generated/generated experiments now include five completed generator/extractor combinations:

| CoT generator | Hidden extractor | Status | Role |
|---|---|---|---|
| R1-8B | R1-1.5B | Completed | Small extractor diagnostic on 8B natural pairs |
| R1-8B | R1-8B | Completed; dense rerun completed | Matched generator/extractor baseline |
| R1-8B | R1-32B | Completed | Cross-size extractor diagnostic; tests whether 8B-generated trajectory signals remain readable in 32B representations |
| R1-32B | R1-8B | Completed; dense rerun completed | Reverse cross-size transfer check |
| R1-32B | R1-32B | Completed; dense rerun completed | Matched 32B generator/extractor diagnostic |

## Completed Checks

- Prompt-only and pre-CoT baselines were added to Stage1b.
- Dense Stage1/Stage1b scans were completed for:
  - natural 8B generated/generated with 8B hidden extractor
  - natural 32B generated/generated with 8B hidden extractor
  - natural 32B generated/generated with 32B hidden extractor
- CPU surface baselines were run for natural 8B pair variants:
  - length-only
  - prompt-only TF-IDF
  - word TF-IDF
  - word BoW
  - character n-gram TF-IDF
  - first-sentence-removed TF-IDF
- Source provenance re-join and LOSO fold construction were implemented and tested on natural 8B generated/generated pairs.
- Hidden probes were run on:
  - A prime OpenAI rewrite with 1.5B hidden extractor
  - natural 8B generated/generated with 1.5B hidden extractor
  - natural 8B generated/generated with 8B hidden extractor
  - natural 8B generated/generated with 32B hidden extractor
  - natural 32B generated/generated with 8B hidden extractor
  - natural 32B generated/generated with 32B hidden extractor
  - natural 32B generated-safe/original-unsafe with 32B hidden extractor

## Current Interpretation Rules

- A prime is useful as a failure case: it shows that hidden probes can become near-perfect when rewrite/style artifacts are present.
- Natural generated/generated pairs are more relevant because both sides are produced by the same base model family under the same pipeline.
- Generated-safe/original-unsafe pairs are useful robustness checks, but they mix a model-generated safe side with original unsafe references that may come from different model and dataset distributions.
- Prompt-only/pre-CoT near-random results are necessary but not sufficient. They address prompt classification, but not text-surface confounding inside the CoT.
- Dense reruns show that prompt/pre-CoT controls remain near random, while CoT-position hidden states show measurable signal.
- High TF-IDF/BoW performance means the paper must report surface baselines next to hidden probes.
- 32B hidden results are diagnostic for representation scale, not a substitute for source-level transfer.
- We should avoid reporting "test-set max" as the main number. The main report should select position/layer on validation, then report test.

## Open Work Before a Stronger Stage 1 Claim

1. Decide whether to freeze a new 32B export from the larger generation inventory or keep the current 359/555-pair frozen exports for comparability.
2. Run hidden probes for natural 8B generated-safe/original-unsafe if we want a direct 8B comparison against generated/generated.
3. Build source-balanced natural pairs or recover more complete source-family provenance.
4. Run true leave-one-source-out once at least two valid source families have enough pairs.
5. For each fold, report:
   - prompt-only/pre-CoT hidden baseline
   - CoT-position hidden probe
   - length-only baseline
   - word/char TF-IDF and BoW baselines
   - first-sentence-removed baseline
   - token-matched truncation baseline
   - embedding-based surface baseline
6. Use paired bootstrap confidence intervals.
7. Report validation-selected position/layer, not post-hoc test maxima.
8. Keep residual confounds visible: length, refusal style, answer template, source family, generator model, and judge selection bias.

## Claim Boundary

Current evidence supports a cautious claim:

> In same-prompt natural-pair settings, prompt-only hidden states are near-random while CoT-position hidden states often contain a safe/unsafe signal. Dense reruns strengthen the prompt-only control result, but shallow CoT-text baselines are also strong and validation-selected scores can be substantially lower than test-set maxima. Therefore, the signal cannot yet be cleanly attributed to safety semantics rather than trajectory style, length, or generation artifacts.

The stronger claim that Stage 1 identifies a safety-relevant latent manifold requires the remaining source-transfer and surface-control experiments.
