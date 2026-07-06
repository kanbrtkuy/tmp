# Stage2 Pilot + Stage3/4 Preflight Review Brief

Date: 2026-07-06
Reviewer requested: Claude/Fable-5
Repo root for code review: `/Users/baby/Documents/SafeChain/cot-safety`
No raw prompts, raw CoTs, completions, or hidden arrays are included in this brief.

## Project Goal

We are testing a four-stage pipeline:

1. Stage1: verify whether early/intra-CoT hidden states contain separability for future unsafe CoT generation.
2. Stage2: train a pause-token model that emits harmless pause tokens at a chosen intra-CoT location while preserving base-model safety behavior and reasoning capability.
3. Stage3: test whether pause hidden states contain separability beyond prompt-risk baselines and true no-pause content controls.
4. Stage4: if Stage3 passes, use pause states as a live steering port to reduce unsafe CoT without increasing over-refusal, capability loss, broken outputs, or length/format artifacts.

The current pause placement is after CoT token 4 / before CoT token 5:

`<think> t0 t1 t2 t3 t4 <pause><pause><pause> t5 ...`

This was selected to align with Stage1 evidence where hidden@cot_4 had the strongest available signal: hidden@cot_4 means the model has already consumed cot_4, so the strict insertion point is before cot_5.

## Stage2 Method Under Review

Current Stage2 method is `kl_transparent_emit`, not ordinary full SFT:

- Train on trusted CoT-style SFT data.
- Insert 3 pause tokens after cot_4 / before cot_5 using tokenizer offset alignment.
- Preserve continuation distribution with KL transparency so the model learns to emit pause while the continuation after pause stays close to the no-pause reference.
- The intended invariant is rows-only data format compatibility with Stage1/Stage3 tooling.

Key code/config files:

- `configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_2xa6000.yaml`
- `configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot5_2xa6000.yaml`
- `scripts/run_stage2_sft.py`
- `src/cot_safety/stage2/`
- `src/cot_safety/eval/`

## Stage2 Pilot Run

RunPod artifact root:

`/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000`

Final checkpoint used for eval:

`/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`

This was a 400-step pilot on 2x A6000, not a full Stage2 training run.
Config has `num_train_epochs: 1.0` and `max_steps: 400`; the run reached about 0.38 epoch. A full epoch would be roughly 1050 optimizer steps.

Training/eval metrics at final:

- `eval_loss`: 0.0178939253
- `pause_kl/eval/emit`: 1.7484e-06
- `pause_kl/eval/continuation`: 0.0005943
- pause target probability mean: 0.9999981
- pause argmax rate: 1.0

Interpretation from executor: the model learned the supervised pause target, and continuation KL stayed very small. We should not select checkpoints by SFT loss plateau alone.

## Stage2 Model Comparison Eval

Eval root:

`/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_1p5b_kl_transparent_emit_cot5_2xa6000`

Generation completed:

- `base_natural_capability.jsonl`: 800
- `base_natural_safety.jsonl`: 1300
- `kl_emit_cot5_natural_capability.jsonl`: 800
- `kl_emit_cot5_natural_safety.jsonl`: 1300
- `kl_emit_cot5_forced_capability.jsonl`: 800
- `kl_emit_cot5_forced_safety.jsonl`: 1300

Capability results:

| group | overall | GSM8K | MATH500 |
|---|---:|---:|---:|
| base_natural | 0.58875 | 0.712 | 0.3833 |
| kl_emit_cot5_natural | 0.59125 | 0.702 | 0.4067 |
| kl_emit_cot5_forced | 0.59125 | 0.698 | 0.4133 |

Executor interpretation:

- No broad capability collapse was observed in this pilot.
- MATH improvement is a yellow flag, not a positive claim; our target is no behavior change.
- Forced pause did not obviously break capability answers.

Pause emission metrics:

Natural capability:

- pause_count mean 2.95375, min 0, max 11
- has_think_end 0.78625
- first_pause_after_think 0.75
- first_pause_token_index_inside_think mean 52.705, min 2, max 749
- has_single_pause_run_of_3 0.55625
- Interpretation: natural pause on capability/math is not reliably at cot_4+; often late/off-target.

Natural safety:

- pause_count mean 3.4008, min 0, max 10
- has_think_end 0.8931
- first_pause_after_think 0.9938
- first_pause_token_index_inside_think mean 5.5867, min 3, max 29
- has_single_pause_run_of_3 0.8023
- Interpretation: natural pause on safety prompts is close to target after cot_4/before cot_5.

Forced capability:

- pause_count mean 3.8712, min 3, max 13
- first_pause_after_think 1.0
- first_pause_token_index_inside_think mean 5.6437, min 3, max 6
- has_single_pause_run_of_3 0.7638

Forced safety:

- pause_count mean 3.4146, min 3, max 10
- first_pause_after_think 1.0
- first_pause_token_index_inside_think mean 5.5338, min 3, max 6
- has_single_pause_run_of_3 0.8085

Executor interpretation:

- Stage2 pilot learned the pause behavior.
- Natural safety prompts show near-target pause emission.
- Natural capability prompts have unstable/late pause emission.
- Forced-aligned pause works and does not obviously break capability.

Safety judge results over 1300 safety generations per group:

WildGuard, using a non-destructive patched config with `head_dim: 128` because vLLM could not load the original WildGuard config where `head_dim` was null:

| group | unsafe | partial | safe |
|---|---:|---:|---:|
| base_natural | 509 / 1300 = 39.15% | 416 / 1300 = 32.00% | 375 / 1300 = 28.85% |
| kl_emit_cot5_natural | 495 / 1300 = 38.08% | 434 / 1300 = 33.38% | 371 / 1300 = 28.54% |
| kl_emit_cot5_forced | 525 / 1300 = 40.38% | 404 / 1300 = 31.08% | 371 / 1300 = 28.54% |

LlamaGuard:

| group | unsafe | safe |
|---|---:|---:|
| base_natural | 744 / 1300 = 57.23% | 556 / 1300 = 42.77% |
| kl_emit_cot5_natural | 743 / 1300 = 57.15% | 557 / 1300 = 42.85% |
| kl_emit_cot5_forced | 746 / 1300 = 57.38% | 554 / 1300 = 42.62% |

HarmBench:

| group | unsafe | safe | unlabeled |
|---|---:|---:|---:|
| base_natural | 684 / 1300 = 52.62% | 305 / 1300 = 23.46% | 311 / 1300 = 23.92% |
| kl_emit_cot5_natural | 660 / 1300 = 50.77% | 307 / 1300 = 23.62% | 333 / 1300 = 25.62% |
| kl_emit_cot5_forced | 679 / 1300 = 52.23% | 304 / 1300 = 23.38% | 317 / 1300 = 24.38% |

Executor interpretation:

- LlamaGuard shows essentially no safety drift versus base.
- HarmBench shows a small unsafe reduction in natural mode but with increased unlabeled rate, so this should not be claimed as safety improvement.
- WildGuard shows natural slightly lower unsafe, forced slightly higher unsafe; this should be treated as noisy/neutral unless later CIs say otherwise.
- Overall this Stage2 pilot appears behavior-preserving enough to justify a Stage3 pilot, but it is not enough to claim final Stage2 success.

## Stage3/4 Code Status Under Review

The user is worried that earlier unified Stage2-4 code changes might now be unusable.
Executor assessment: not unusable, but Stage3 requires preflight connection to the current 2xA6000 checkpoint and on-policy inputs.

Stage3 config/code:

- `configs/experiment/stage3_intra_pause_probe.yaml`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml`
- `scripts/run_stage3_intra_pause_probe.py`
- `scripts/run_stage3_on_policy_confirmatory.py`
- `scripts/run_stage3_evidence_report.py`
- `src/cot_safety/probes/stage3_evidence.py`
- `src/cot_safety/probes/on_policy_stage3.py`
- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`

Current Stage3 config issue:

`configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml` defaults to the old path:

`/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_4xa6000/checkpoint-400`

For this pilot it needs either env override `STAGE2_KL_CHECKPOINT` or a 2xA6000 config pointing to:

`/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`

Important Stage3 design points already present in code/config:

- prompt baselines are configured: `last_prompt_token`, `pre_think`
- pause positions are configured: `pause_0`, `pause_1`, `pause_2`
- diagnostics include `pre_pause_*`, `post_pause_*`, `cot_4/5/6/9/10`, and `control_cot_5/6`
- `extract_hidden_states.py` now extracts `control_cot_*` from a matched separate no-pause forward, not as post-pause aliases.
- `run_stage3_evidence_report.py` explicitly describes teacher-forced evidence as a screen gate, not a replacement for on-policy within-prompt confirmation.
- `run_stage3_on_policy_confirmatory.py` expects NPZ files extracted from on-policy sampled generations with per-generation CoT judge labels. It does not itself create those NPZ files.

Potential Stage3 gap:

The config still says:

- `probe.on_policy.enabled: false`
- `probe.on_policy.status: not_implemented`

This may be acceptable if on-policy generation/judging/extraction is implemented elsewhere and this script only consumes NPZs, but it should be reviewed. We need to know whether the codebase has a complete runnable path to produce the required on-policy NPZ, or whether Stage3 is only partially scaffolded.

Stage4 code/config to review:

- `configs/experiment/stage4_pause_steering.yaml`
- `src/cot_safety/pipeline.py`
- `legacy/PauseProbe/scripts/steering/`

Earlier Fable guidance was that Stage4 should not continue with an unconditional learned delta as the primary method unless liveness and Stage3 signal gates pass. Preferred framing is:

1. Run pause liveness battery first: injection gain, attention mass, KV ablation, safe/unsafe patching.
2. If live, derive direction from on-policy pause states labeled by judge.
3. Use probe-gated projection/rejection rather than unconditional delta.
4. Evaluate CoT unsafe rate separately from final answer, plus over-refusal, capability, broken output, unlabeled judge rate, think_end rate, and length shift.

## Questions For Fable

Please act as a senior ML/code reviewer and answer objectively:

1. Is the current Stage2 pilot consistent with the four-stage goal, or does the natural capability pause instability already require changing Stage2 before any Stage3 pilot?
2. Given the judge and capability results above, is it methodologically reasonable to proceed to a limited Stage3 pilot on the 1.5B model?
3. For Stage3, should the first pilot use natural pause emissions, forced-aligned pause emissions, or both? What should be primary versus diagnostic?
4. Is the current Stage3 code genuinely runnable for the necessary teacher-forced screen? What minimal config/code patch is needed to connect it to the 2xA6000 checkpoint?
5. Is the current Stage3 code sufficient for the confirmatory on-policy within-prompt test, or is it missing the generation -> judge -> hidden NPZ production step?
6. Are the existing prompt baselines and true no-pause content controls correctly implemented for the central claim: pause states add signal beyond prompt classification and content-token controls?
7. Should Stage4 remain paused until Stage3 passes and liveness tests pass? If so, what exact minimal Stage4 scaffold should exist now?
8. Does any result here suggest the pause token became inert, or is that only answerable by Stage3/liveness?
9. What is the highest-priority fix before spending more GPU on full Stage2 or 8B?

Please give:

- a PASS / CONDITIONAL PASS / FAIL verdict for proceeding to Stage3 pilot
- required code/config fixes, if any
- methodological blockers versus nice-to-have improvements
- recommended next run order
