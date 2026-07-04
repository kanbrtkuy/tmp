# Claude Review Request: SafeChain Stage2 Pause-Token SFT

Please act as a senior ML research reviewer and experimental designer. Review
this packet as if advising a paper/project before we spend more GPU budget.

The central question:

> How can we train a model to emit pause tokens at a target CoT position while
> changing the model's underlying reasoning/capability as little as possible?

We need concrete methods and experiments, not generic advice.

## Project Context

SafeChain investigates whether hidden states along chain-of-thought trajectories
contain a separable safe/unsafe signal, and whether pause-token interventions
can expose or steer that signal.

Pipeline:

1. Stage1: teacher-forced hidden-state probe over CoT token positions.
2. Stage1b: prompt-only / pre-CoT controls and source-family generalization.
3. Stage2: train models to emit three `<|pause|>` tokens inside `<think>` at a
   target CoT offset.
4. Stage3: probe SFT checkpoints at inserted pause positions.
5. Stage4: pause-only steering and end-to-end safety/capability evaluation.

## Reference Paper

Use this paper as conceptual background, but do not assume our setup is the
same:

- Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar,
  Vaishnavh Nagarajan. "Think before you speak: Training Language Models With
  Pause Tokens." arXiv:2310.02226. ICLR 2024.
  https://arxiv.org/abs/2310.02226

Important distinction: that paper studies pause-training as extra computation
and reports task gains when models are pre-trained and fine-tuned with delays.
Our goal is different: insert pause tokens as an instrumentation point for
safety monitoring/steering while minimizing unrelated capability drift.

## What Stage2 Currently Does

Primary Stage2 runner:

- `cot-safety/scripts/run_stage2_sft.py`

Data builder:

- `cot-safety/legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py`

Trainer:

- `cot-safety/legacy/COTPauseToken/src/trl_train.py`
- `cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`

Data:

- `cot-safety/configs/data/stage2_trusted_cot_18k.yaml`
- About 18k trusted high-quality CoT rows from Sky-T1, Bespoke-Stratos, and
  OpenThoughts.
- Train/val/test: 17000/500/500.
- Variants include `intra_pause_cot4`, `intra_pause_cot3`,
  `no_pause_matched`, and `pre_think_pause`.

Main 8B configs:

- Full SFT cot4:
  `cot-safety/configs/experiment/stage2_intra_pause_sft_8b_4xa100.yaml`
- Full SFT cot3 control:
  `cot-safety/configs/experiment/stage2_intra_pause_sft_8b_cot3_control_4xa100.yaml`
- Format-only cot4:
  `cot-safety/configs/experiment/stage2_intra_pause_format_only_8b_cot4_save50_max250_4xa100.yaml`
- Format-only cot3:
  `cot-safety/configs/experiment/stage2_intra_pause_format_only_8b_cot3_save50_max250_4xa100.yaml`

1.5B full-SFT config:

- `cot-safety/configs/experiment/stage2_intra_pause_sft_1p5b_4xa6000.yaml`

Model-comparison eval configs:

- `cot-safety/configs/experiment/stage2_model_comparison_eval.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_1p5b_cot3_ckpt200_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt250_3xidle.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt200_250_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt450_500_4xa100.yaml`

## Current Key Results

Read first:

- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`

Important observations from that file:

- Every tested SFT row learned the requested pause format:
  `pause3_rate = 1.000`.
- The earlier 8B cot4 full-SFT final model improved overall capability relative
  to base: `0.649` vs `0.603`, with GSM8K `0.762` vs `0.710`.
- The later cot3 full-SFT checkpoints also recover/exceed base overall
  capability: ckpt250 `0.646`, ckpt300 `0.636`, versus base `0.603`.
- This is a yellow flag for construct validity: full SFT is not merely teaching
  pause-token insertion; it may teach a general high-quality reasoning style.
- Format-only cot4 ckpt250 is currently the cleaner candidate: overall
  capability `0.586` vs base `0.603`, while unsafe-prompt unsafe_valid_rate
  decreases under HarmBench, LlamaGuard, and WildGuard.
- Format-only cot3 is a bad ablation: unsafe-prompt unsafe_valid_rate is higher
  than base for all tested cot3 format-only checkpoints.

Current default 8B downstream checkpoint:

- `cot-safety/configs/model/deepseek_r1_distill_llama_8b.yaml`
- Points to `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-250`

## Known Problems And Critiques

Read:

- `PROFESSOR_CRITIQUE.md`
- `cot-safety/res/stage1_stage1b_prompt_baseline_summary_20260630.md`
- `cot-safety/res/stage1_natural_pair_experiment_results_260703.md`
- `cot-safety/plan/stage1_natural_pair_experiment_plan_260703.md`

Relevant issues:

1. Stage1 teacher-forced probes may classify prompt/source/style rather than
   trajectory safety. Prompt-only/pre-CoT controls and LOSO were added, but
   natural-pair results still have surface baseline and source-balance concerns.
2. Stage2 full SFT can improve capability, which is suspicious for a "minimal
   pause insertion" goal.
3. Stage3/Stage4 steering may pick up a general structured/careful direction
   from high-quality SFT data rather than a clean unsafe axis.
4. Teacher-forced probe distributions differ from self-generated steering
   distributions.
5. Relative safety reductions must keep absolute residual unsafe rates visible.

## Specific Review Questions

Please answer these directly.

1. What is the cleanest Stage2 objective if we want pause insertion with minimal
   capability drift?
   - Compare full SFT, format-only embedding-row training, LoRA/adapters,
     constrained loss on only pause positions, KL-regularized SFT, DPO-style
     format preference, logit-bias or constrained decoding, and any other
     method you think is better.

2. Is current format-only training actually sufficient?
   - It freezes all model parameters except `<|pause|>` embedding row(s).
   - Does this still risk capability drift through changed token semantics?
   - What diagnostics would reveal subtle drift?

3. Can we decouple "insert pause token" from "learn high-quality reasoning"?
   - Propose controls that isolate the SFT-data effect from the pause-token
     effect.
   - For example: no-pause matched SFT, random-token insertion, frozen-model
     constrained decoding, copy-target loss on only inserted tokens, KL to base,
     matched high-quality SFT without pauses, etc.

4. What is the minimal experiment package to satisfy the professor's critique?
   - Assume limited GPU budget.
   - Prioritize experiments by acceptance lift per GPU-day.
   - Include exact model sizes, checkpoints, datasets, controls, and metrics.

5. How should we frame results if full SFT improves capability?
   - Is it fatal, or usable as a diagnostic?
   - What claim boundary is defensible?

6. How should Stage3/Stage4 characterize the steering vector?
   - What correlations should we measure: refusal rate, length, topic, style,
     output entropy, CoT length, answer directness, judge disagreement, etc.?
   - What would convince you the vector is safety-related and not merely a
     structured/careful or refusal-style direction?

7. Are there better training designs from the pause-token literature?
   - In particular, how should we interpret Goyal et al. 2024 given our goal is
     not to improve task performance but to add a minimally invasive monitoring
     point?

8. Please produce:
   - A prioritized action plan.
   - A results-to-claims matrix.
   - A "do not claim" list.
   - A recommended next Stage2 config family.
   - Any concise pseudo-code/objective equations for the preferred training
     objective.

Please be blunt. If the current Stage2 design cannot support the intended
claim, say so and explain the smallest redesign that could.

