# Fable-5 XHigh Review Prompt: Stage2.1 Clean Natural Pause Emission

Please use extra-high effort. We need a rigorous, paper-grounded plan to fix
SafeChain Stage2 pause-token SFT.

## Goal

We want the model to naturally learn clean pause insertion, not rely on forced
pause or decoding constraints as the main paper method.

Strict Stage2 success target:

- natural generation emits exactly three `<|pause|>` tokens;
- the pause block appears at the intended location after `cot_4` / before
  `cot_5`;
- off-target pause emission is near zero;
- target exact-3 success should be >=99% on GSM8K, MATH500, and safety-source
  prompts;
- capability and safety behavior should stay close to base;
- Stage3 can still test whether pause hidden states contain trajectory safety
  signal, so the training should not inject safety labels into pause states.

## Current Failure

The 8B Stage2 run used KL-transparent emit SFT:

- Model: DeepSeek-R1-Distill-Qwen-8B.
- Data: 18k high-quality reasoning rows from Sky-T1, Bespoke-Stratos, and
  OpenThoughts.
- Pause target: `<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...`
- Only the pause token embedding/unembedding rows are trainable.
- Loss:
  - emit CE on pause targets;
  - continuation KL to a pause-stripped teacher sequence;
  - suppression at teacher-forced non-pause targets;
  - no safety-label supervision.
- Early stopping and best checkpoint selection used eval loss, not
  natural-generation exact-3.

Teacher-forced metrics looked good:

- final eval loss: 0.007503
- pause target argmax rate: 1.0
- continuation KL: about 0.0006-0.0008

But natural pause emission failed:

| dataset | exact 3-pause | off-target pause rate | avg pause count |
|---|---:|---:|---:|
| GSM8K | 0.196 | 0.802 | 8.288 |
| MATH500 | 0.9267 | 0.0033 | 3.377 |
| HarmBench | 0.855 | 0.025 | 3.285 |
| JailbreakBench | 0.920 | 0.010 | 3.170 |
| OR-Bench hard | 0.7967 | 0.020 | 3.470 |
| StrongReject | 0.890 | 0.0133 | 3.217 |
| XSTest | 0.700 | 0.004 | 3.792 |

Prior Fable review diagnosed two failure modes:

1. target-location run-length overshoot: model begins pause block but emits 4-6
   pauses instead of exactly 3;
2. mid-generation re-triggering, especially on GSM8K: model emits extra pause
   blocks later at step-boundary-like contexts.

Prior conclusion:

> This is not an undertrained model; it is a correctly optimized wrong
> objective.

## Relevant Code Paths

Repo:
`https://github.com/kanbrtkuy/cot-safety`

Current public commit with the failure review:
`5452654`

Key files:

- `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_full_save25_2xa100.yaml`
- `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_save50_max400_4xa100.yaml`
- `configs/data/stage2_trusted_cot_18k.yaml`
- `scripts/run_stage2_sft.py`
- `src/cot_safety/formatting/pause_insertion.py`
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py`
- `legacy/PauseProbe/scripts/eval/summarize_model_comparison_eval.py`
- `scripts/audit_stage2_stage3_gate.py`
- `res/stage2_8b_full_results_260707_zh.md`
- `review-stage/stage2_8b_pause_emission_failure_260707/fable_stage2_emission_failure_review_260707.md`

If GitHub is inaccessible, rely on the above method and result summary. Do not
ask for secrets or raw data.

## Literature To Use

Please ground your proposal in papers such as:

- Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
  (Bengio et al., 2015): teacher-forcing vs inference mismatch /
  exposure bias. https://arxiv.org/abs/1506.03099
- Professor Forcing (Lamb et al., 2016): matching training and sampling
  dynamics. https://arxiv.org/abs/1610.09038
- DAgger / Dataset Aggregation (Ross et al., 2010/2011): training on the
  distribution induced by the learned policy. https://arxiv.org/abs/1011.0686
- Neural Text Generation with Unlikelihood Training (Welleck et al., 2019):
  negative-token training for repetitions/unwanted tokens.
  https://arxiv.org/abs/1908.04319
- InstructGPT / KL-regularized RLHF (Ouyang et al., 2022): changing behavior
  while controlling drift with KL. https://arxiv.org/abs/2203.02155

You may add better or more relevant papers, but cite them and explain why they
support the proposed fix.

## Questions

1. What is the cleanest Stage2.1 method if the paper must prioritize natural
   learned exact-3 pause emission over decoding constraints?
2. Should we keep rows-only pause-token training, or is that too constrained for
   >=99% exact-3 across GSM8K/MATH/safety prompts?
3. What losses should be implemented exactly?
   Please specify:
   - on-policy negative mining / DAgger-style data loop;
   - margin suppression or unlikelihood objective;
   - explicit stop-after-3 loss;
   - no-extra-pause at wrong positions;
   - continuation KL and whether it should remain;
   - whether to add pre-KL or not.
4. How should natural-generation validation and checkpoint selection work?
   Please specify exact metrics, thresholds, evaluation set composition, and
   stop/best-checkpoint criteria.
5. What minimal cheap experiment should we run before another full 8B training?
   1.5B? short 8B? existing checkpoint diagnostics?
6. What code changes should Codex implement first in this repo?
   Please be concrete: files, functions/classes, new configs, tests.
7. What should we ask Fable to review after Codex implements the code?

Please produce:

- a ranked diagnosis;
- the Stage2.1 algorithm;
- paper-backed justification;
- implementation checklist;
- test checklist;
- go/no-go criteria for the next GPU run;
- claims allowed/disallowed.
