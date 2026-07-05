# Stage1 / Stage2 Evidence Summary For Pause Placement

This summary is aggregate-only. It excludes raw prompts, raw CoTs,
completions, hidden arrays, and row-level prediction JSONL.

## User's Four-Stage Research Plan

The user wants the project to follow a staged logic:

### Stage 1 — latent separability

Original Stage1 concept:

- Use an R1-1.5B base model on SafeChain data, especially StrongReject and
  WildJailbreak.
- Generate Normal and MoreThink variants.
- Sample 10 rollouts per prompt at temperature 0.7.
- Label a CoT as unsafe if any point in the CoT is unsafe according to a judge
  such as Llama-Guard.
- Extract hidden states from early tokens, focusing on middle/later layers.
- Train linear and MLP probes with prompt-disjoint train/test splits.
- Target: paired-clean balanced accuracy around 80%, enough to show latent
  separability exists.

Actual latest Stage1 evolved beyond that original small plan:

- Model: DeepSeek-R1-Distill-Llama-8B.
- Data: natural same-prompt generated/generated safe-vs-unsafe pairs.
- Sources: HarmBench, ReasoningShield, StrongReject, WildJailBreak.
- Frozen LOSO setting, post-HB retune with `max_length=12288`.
- Labels are final safe/unsafe trajectory labels; prompt-only text baseline is
  uninformative by construction for same-prompt pairs.

### Stage 2 — train pause-token model

Only if Stage1 is sufficiently meaningful, train a pause-token model:

- Full SFT was the original idea; later Fable reviews pushed toward cleaner
  format-only / KL-transparent variants.
- Training data originally envisioned: UltraFeedback chosen responses.
- Insert K pause tokens before/inside the CoT.
- Goal: make the model emit or tolerate pause tokens before continuing CoT,
  without changing the safety property or drifting behavior.
- Validation: unsafe-CoT ratio and capability should remain close to the base
  model.

### Stage 3 — pause-position separability

Use the pause model on the same data distribution:

- Extract hidden states at pause token positions.
- Train probes on pause positions.
- Compare pause vs no-pause probe performance.
- Goal: pause hidden states should make safe/unsafe trajectory separability
  more accessible.

### Stage 4 — steering

If Stage3 is positive:

- Use a probe or unsafe direction to identify an unsafe manifold.
- Intervene at pause hidden states to pull away from unsafe regions.
- Goal: reduce unsafe CoT without increasing over-refusal, breaking output, or
  degrading capability.

## Latest Stage1 Status

The latest post-HB Stage1 run is best treated as a negative/control result for
the current linear hidden-probe design, not as hidden-state superiority
evidence.

Key facts from the sanitized result docs:

- Frozen post-HB LOSO keep pairs:
  - HarmBench: 152
  - ReasoningShield: 335
  - StrongReject: 277
  - WildJailbreak: 2019
  - Total: 2783
- Freeze audit: `n_input_pairs=2783`, `n_main_keep=2783`, `n_dropped=0`,
  `n_rejected_during_load=0`.
- All 16 validation-selected hidden-minus-surface AUROC deltas are negative.
- Validation-selected hidden test AUROC is roughly 0.679 to 0.840.
- Validation-selected full-trajectory surface text baselines reach roughly
  0.917 to 0.965.
- `length_only` beats the selected hidden probe on all four sources, showing
  that full-trajectory hindsight surface structure is extremely strong.
- Prompt-only TF-IDF is 0.500 AUROC on every split/source in the same-prompt
  paired design.
- Fable-5 previously concluded: current Stage1 should be accepted as a
  negative/control result for the current probe design; do not spend more GPU
  on the same Stage1 design; matched-horizon CPU analysis is the right
  follow-up if continuing Stage1.

## The cot120 vs cot4 Tension

There are two kinds of Stage1 evidence that point to different token locations:

### Late readout evidence

Natural same-prompt dense Stage1 diagnostics often found the strongest
post-hoc readout around much later CoT positions, e.g. `cot_120` to `cot_160`.

Interpretation under our current understanding:

- Late CoT positions are likely better hindsight/readout points.
- They see much more of the trajectory, so final safe/unsafe direction is
  easier to classify.
- They may be poor intervention points because the model has already committed
  to much of the reasoning trajectory.
- Coverage is worse at high offsets; short/refusal-like trajectories can be
  censored at `cot_96` / `cot_128`.

### Early intra-CoT evidence

Earlier dense Stage1/Stage1b prompt-baseline diagnostics found strong early-CoT
signal around `cot_3` and `cot_4`:

- For 8B, strict prompt-only heldout mean AUROC was about 0.801.
- `cot_3` heldout mean was about 0.912.
- `cot_4` heldout mean was about 0.921.
- The strongest single-layer heldout point for 8B leaned toward `cot_4`.
- These historical prompt-baseline LOSO tables were later caveated by Fable as
  not the main construct-validity evidence, but the early-position pattern
  remains relevant for engineering a pause port.

Interpretation:

- `cot_3` / `cot_4` are early enough to intervene before most of the reasoning.
- They have much better coverage than high offsets.
- They are intra-CoT, so they avoid the severe pre-CoT "only prompt
  classification" critique when decoding is sampled and multiple rollouts per
  prompt exist.

## Existing Stage2 Evidence

Existing DeepSeek-8B Stage2 format-only SFT comparisons:

### cot4 format-only

Pause placement: before `cot_4`.

| Metric | base | cot4 ckpt200 | cot4 ckpt250 |
|---|---:|---:|---:|
| GSM8K acc | 0.710 | 0.652 | 0.684 |
| MATH500 acc | 0.423 | 0.463 | 0.423 |
| overall acc | 0.603 | 0.581 | 0.586 |
| pause3_rate | 0.000 | 1.000 | 1.000 |

Unsafe-prompt unsafe_valid_rate:

| Judge | base | cot4 ckpt200 | cot4 ckpt250 |
|---|---:|---:|---:|
| HarmBench | 0.465 | 0.463 | 0.432 |
| LlamaGuard | 0.553 | 0.518 | 0.505 |
| WildGuard | 0.440 | 0.405 | 0.365 |

Prior interpretation: `cot4 checkpoint-250` is the healthier candidate among
format-only 8B checkpoints. It learns the requested pause format, stays close
to base capability, and reduces unsafe-prompt unsafe_valid_rate across all
three judges.

### cot3 format-only

Pause placement: before `cot_3`.

Unsafe-prompt unsafe_valid_rate was worse than base across all three judges for
all tested cot3 format-only checkpoints.

Prior interpretation: cot3 format-only behaves worse than cot4 for 8B and
should be treated as a position ablation rather than the main downstream
steering model.

### cot3 full-SFT diagnostic

The later cot3 full-SFT checkpoints behaved better than cot3 format-only, but
they are not clean format-only interventions. They are useful diagnostics, not
clean evidence about pause placement.

## Prior Fable Clarification

Fable previously clarified that cot3/cot4 placements are not the severe
pre-CoT problem:

- The severe critique applies when zero stochastic tokens precede the pause
  positions, or the pre-pause prefix is deterministic/teacher-forced identical
  within each prompt.
- cot3/cot4 pauses are intra-CoT if inserted after sampled tokens.
- Stage3 on cot3/cot4 is conceptually valid if decoding is sampled, position
  convention is audited, per-sample pause-state variance is checked, and paired
  protocols plus surface baselines are used.

## Current Provisional Interpretation

Our provisional interpretation before this review:

- `cot_120` is a strong readout/hindsight location, not necessarily a good
  causal intervention port.
- For 8B Stage2, the main pause placement should be `before cot_4`.
- `before cot_3` should remain an ablation.
- A late-pause ablation around `cot_64` or `cot_120` may be scientifically
  interesting, but is probably not the best first Stage2 mainline due to
  coverage, causality, and compute cost.

