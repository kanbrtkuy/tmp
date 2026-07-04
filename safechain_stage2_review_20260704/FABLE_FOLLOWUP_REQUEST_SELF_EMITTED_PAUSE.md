# Fable Follow-Up Request: Self-Emitted Pause Tokens

Date: 2026-07-04

Please perform a second, objective review of the SafeChain Stage2 design under
an important clarification that was not emphasized enough in the first review.

Read first:

- `CLAUDE_FABLE5_REVIEW.md`
- `CLAUDE_REVIEW_REQUEST.md`
- `PROFESSOR_CRITIQUE.md`
- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`

Then evaluate the clarified plan below.

## Clarified Four-Stage Plan

### Stage 1: Test whether latent separability exists

Current target:

- Use the original DeepSeek-R1-Distill-Qwen-1.5B model.
- Use SafeChain data from StrongReject + WildJailbreak.
- Generate CoT with two policies: Normal and MoreThink.
- Sample 10 completions per prompt at temperature 0.7.
- Label each CoT with Llama-Guard: if any position in the CoT is unsafe, the
  whole trace is labeled unsafe.
- Extract hidden states from the first few CoT tokens, especially mid-to-late
  layers.
- Train linear and MLP probes.
- Split train/test by prompt.
- Main success criterion: `paired_clean` balanced accuracy around 80%, showing
  safe/unsafe trajectory separability in latent space.

### Stage 2: Train a model that self-emits pause tokens

This stage should only happen if Stage 1 works.

Original plan:

- Full SFT, not LoRA.
- Use UltraFeedback chosen responses.
- Add K pause tokens before each response.
- Goal: make the model output pause tokens before CoT/response.
- Crucial constraint: the pause model should not materially change the base
  model's safety property or capability.
- Validation: unsafe CoT rate of the pause model should be close to the original
  model's unsafe CoT rate.

Important clarification:

The pause tokens are not merely a forced inference-time artifact. They are meant
to be **self-emitted by the model**. The ability to autonomously output pause
tokens is a key part of the causal chain.

### Stage 3: Test separability at pause positions

- Run the same SafeChain dataset with the pause model.
- Extract hidden states at the pause-token positions.
- Train probes to classify whether the subsequent CoT trajectory will be
  safe/unsafe.
- Compare:
  - pause-token positions vs ordinary early CoT token positions
  - pause model vs no-pause/base model
- Desired result: pause positions expose a cleaner or more stable safe/unsafe
  latent signal.

### Stage 4: Steering at pause positions

This stage should only happen if Stage 3 works.

- Use a probe or unsafe direction to identify an unsafe manifold/direction.
- Intervene on pause-token hidden states.
- Pull the model away from the unsafe region.
- Final goal: reduce unsafe CoT while not increasing over-refusal, not breaking
  output format, and not damaging general capability.

## My Current Understanding of the Role of Pause Tokens

Pause tokens are not primarily for "extra thinking" or task improvement. They
are intended to create a fixed, repeatable, accessible latent checkpoint:

1. **Position alignment:** every generation has a common intervention point:
   `prompt -> pause_1 ... pause_K -> CoT`.
2. **Signal exposure:** Stage 3 tests whether pause hidden states carry the
   Stage-1 safe/unsafe trajectory signal more cleanly than ordinary early CoT
   tokens.
3. **Steering interface:** Stage 4 uses that fixed position as the place where a
   probe/vector/loss can intervene before the unsafe CoT unfolds.

Therefore, Stage 2 should not substantially change the reasoning/safety
distribution. If Stage 2 full SFT improves math, changes refusal style, or
changes unsafe-CoT rate, that threatens the causal story.

## Candidate Alternative To Full SFT

I proposed replacing UltraFeedback full SFT with a more targeted objective:

**Pause-only CE + KL-to-base self-distillation**

Let `p_base` be the original model and `p_pause` be the pause-emitting model.

Train the model to output K pause tokens, but constrain the continuation after
the pause tokens to match the base model as closely as possible.

High-level objective:

```text
L = gamma * CE_pause
  + beta  * KL_continuation(p_base || p_pause)
  + alpha * KL_non_pause_suppression
```

Where:

- `CE_pause`: cross-entropy only at the K pause-token positions.
- `KL_continuation`: for ordinary tokens after the pause prefix, match
  `p_pause(. | prompt + pauses + prior continuation)` to
  `p_base(. | prompt + prior continuation)`.
- `KL_non_pause_suppression`: suppress pause-token probability at non-target
  positions.

Suggested parameterization:

- Freeze most or all base model weights.
- Train pause-token input embedding and output-head row.
- If that is not enough for self-emission, optionally add a very small adapter
  or small LoRA, but only with strong KL-to-base anchoring.
- Prefer self-generated base-model continuations over UltraFeedback chosen
  responses, to avoid learning a new response style.

Desired property:

```text
unsteered pause model ~= base model
steered pause model can still be influenced through pause hidden states
```

## Questions For This Second Review

Please be objective and blunt.

1. Under this clarified four-stage plan, is self-emitted pause a scientifically
   necessary requirement, or could forced insertion still support the intended
   claim? Explain the trade-off.
2. Is UltraFeedback full SFT fundamentally incompatible with the "minimal
   behavioral drift" requirement? If not, what controls would rescue it?
3. Is the proposed pause-only CE + KL-to-base objective a valid method for
   self-emitted pause tokens with minimal capability/safety drift?
4. Would this objective still preserve enough causal access for Stage 4
   steering, or would KL transparency make the pause hidden state too no-op to
   steer?
5. What is the best training method you would recommend, given that the model
   must self-emit pause tokens?
6. What are the minimal experiments needed to prove:
   - the model self-emits pauses at the correct position
   - unsteered behavior matches base
   - pause positions improve separability
   - pause-position steering reduces unsafe CoT without over-refusal or broken
     output
7. Are there better ideas than pause-only CE + KL-to-base? If yes, propose them
   concretely.
8. Please produce:
   - final verdict
   - method ranking
   - recommended Stage2 objective
   - concrete loss equations
   - ablation matrix
   - acceptance criteria
   - do-not-claim list
   - highest-priority next implementation tasks

Please do not assume that previous Stage2 full-SFT results are correct evidence
of pause-token learning, because the prior review noted that the evaluation may
force-insert pause tokens. Treat self-emission as a separate empirical question.
