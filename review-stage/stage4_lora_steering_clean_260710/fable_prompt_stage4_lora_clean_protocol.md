# Fable Prompt: Clean Stage4 Protocol for LoRA/Pause-Port Steering

We need a critical experimental design review, not encouragement.

## Context

Project: SafeChain / pause-port steering.

Original four-stage idea:

1. Stage1: show early CoT hidden states contain safe/unsafe separability.
2. Stage2: insert three content-free pause tokens after `cot_4` / before `cot_5`.
3. Stage3: verify pause hidden states carry safety signal.
4. Stage4: steer at pause hidden states to reduce unsafe CoT without breaking reasoning, over-refusing, or corrupting format.

The professor's core critique for Stage2/3/4:

- The clean-intervention claim is not proven unless we run matched-strength steering on ordinary tokens (`token_3/token_4` or `post_pause`) and show pause steering preserves semantics/capability better.
- SFT can change capability/safety by itself, so Stage4 must separate base/SFT-or-LoRA effects from steering effects.
- Probe evidence was teacher-forced while steering is generation-time/on-policy; this distribution mismatch must be named or addressed.
- If steering improves GSM8K/MATH, that is suspicious: the vector may encode "structured/careful/refusal/length" rather than unsafe.

## Current method state

We moved from full SFT to Stage2.3 PPC:

- Runtime FSM/logits processor forces exact `<|pause|><|pause|><|pause|>` after `cot_4` / before `cot_5`.
- The base model remains mostly frozen.
- A position-masked LoRA / pause-row calibration is trained so the pause port carries on-policy safety signal.
- This makes the experimental question cleaner: compare `base + FSM + LoRA` with and without steering, rather than a full-SFT model whose capability shifted.

Fable's corrected Stage2.3 position:

- Claim is no longer "pause is special".
- Claim is "a dedicated calibrated content-free port can carry signal".
- Trained control-token parity is acceptable as exchangeability; the key comparators are untrained inserted control, no-insertion matched base hidden state, and text-prefix baseline.

Recent Stage3A paired teacher-forced diagnostic:

- StrongReject: `pause_concat_layers_concat AUROC=0.7238`, `pause_mean_layers_concat=0.7222`, content-control concat `0.7182`, top pre/post around `0.7289`.
- WildJailbreak: content-control concat `0.7952`, pause concat `0.7716`, pause mean `0.7605`.
- HarmBench: pause strong, `pause_mean_layers_concat=0.8193`, pause concat `0.8172`, content-control concat `0.7962`.
- ReasoningShield: post/content around `0.7266/0.7110`, pause concat `0.7004`, pause mean `0.6950`.

Fable's previous ruling: this closes only H1 port-readable/not-erased; it does not prove pause-vs-content privilege. No more paired diagnostics; move to no-GPU prefix ICC/ceiling and then on-policy/liveness/steering.

## Current code state / gap

- Old `learned_delta` generation/eval exists, but it is deprecated and confounds directions.
- New GPRS/projection scaffold exists:
  - `cot_safety.steering.gprs.projection_rejection_update`
  - `scripts/build_stage4_gprs_artifacts.py`
  - `scripts/run_stage4_liveness.py`
  - configs `stage4_pause_gprs*.yaml`
- But `scripts/run_stage4_steering.py` currently stops with:
  "GPRS generation is scaffolded but not wired into the legacy generation shell yet."
- `cot_safety.steering.scope` currently forbids non-pause targets, which is good for the main method but blocks the professor's diagnostic counterfactual unless we explicitly add a diagnostic-only matched target mode.

## What we need from you

Design a clean Stage4 experimental protocol for the LoRA/PPC setting that isolates steering effect.

Please be concrete:

1. What exact model conditions should be compared?
   - base no FSM/no LoRA?
   - base + FSM + untrained pause row?
   - base + FSM + trained pause-row/LoRA, no steering?
   - base + FSM + trained LoRA + pause steering?
   - matched ordinary-token steering?
   - random direction / sign-flip / safe prompts?

2. What exact steering targets should be used for the professor's clean-intervention counterfactual?
   - pause_0/1/2?
   - cot_4, cot_5?
   - post_pause_1?
   - no-insertion matched position?
   How should these be implemented without corrupting the main pause-only safety guard in the code?

3. What direction should be used?
   - mean-diff from on-policy pause states?
   - probe normal?
   - gated projection/rejection?
   - How to avoid a "structured/careful/refusal/length" vector?

4. What data/eval should be used for the first clean 1.5B run?
   - unsafe prompts / safe prompts / reasoning tasks?
   - sample counts?
   - judge(s)?
   - seeds?

5. What metrics and gates decide success?
   Must separate:
   - unsafe CoT rate
   - unsafe final answer rate
   - over-refusal on safe/hard-safe prompts
   - capability (GSM8K/MATH)
   - broken output / think_end / repetition / length shift
   - pause format correctness
   - steering hook actually applied to intended positions

6. What is the minimum clean run order?
   We want the smallest sequence that can answer whether LoRA-pause steering itself has a clean effect, before spending on 8B.

7. What code changes are required?
   Please distinguish must-have vs nice-to-have.

8. What claims would be supported if the result is:
   - pause steering reduces unsafe but normal-token steering also works similarly;
   - pause steering reduces unsafe with less capability/coherence damage than normal-token steering;
   - both reduce unsafe but increase refusal/length substantially;
   - neither changes unsafe.

Be critical. List blockers. Do not assume existing GPRS eval code is complete.
