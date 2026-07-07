# Fable-5 Review Prompt: Stage2 8B Pause Emission Failure Analysis

Please review the current SafeChain Stage2 8B method, code, results, and failure
mode. The goal is not to decide whether Stage3 can proceed; the goal is to
diagnose why Stage2 failed the stricter success criterion:

> In natural generation, the model should insert exactly 3 pause tokens at the
> target location after `cot_4` / before `cot_5` with at least 99% success,
> without capability/safety drift.

Do not assume Stage2 succeeded just because teacher-forced eval loss/KL is low.
Please identify likely root causes and concrete fixes.

## GitHub State To Review

Repo:
`https://github.com/kanbrtkuy/cot-safety`

Relevant commit:
`66ab5c2`

Key files:

- Stage2 8B full config:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_full_save25_2xa100.yaml`
- Parent KL-transparent config:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_save50_max400_4xa100.yaml`
- Stage2 data config:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/configs/data/stage2_trusted_cot_18k.yaml`
- Stage2 runner:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/scripts/run_stage2_sft.py`
- Pause insertion implementation:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/src/cot_safety/formatting/pause_insertion.py`
- Pause-KL trainer:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- Legacy distributed SFT launcher:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
- Stage2 eval config:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100.yaml`
- Stage2 result doc:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/res/stage2_8b_full_results_260707_zh.md`

## Method Summary

- Model: DeepSeek-R1-Distill-Qwen-8B.
- Data: `stage2_trusted_cot_18k`, 18k high-quality reasoning rows from
  Sky-T1, Bespoke-Stratos, and OpenThoughts.
- Pause target: insert 3 `<|pause|>` tokens after CoT token 4 / before CoT
  token 5:
  `<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...`
- Insertion uses tokenizer offset alignment, implemented by
  `insert_pause_before_cot_offset`; config uses `cot_offset=5`.
- Training method: `kl_transparent_emit`.
- Only format/pause token rows are trainable; non-pause model weights are
  frozen / checked by rows-only invariant.
- Loss in `PauseKLSFTTrainer`:
  - CE emit loss only where target token is `<|pause|>`.
  - continuation KL between paused student sequence and pause-stripped teacher
    sequence.
  - pre-KL weight set to 0 in the 8B config.
  - suppression loss discourages pause emission at non-pause supervised target
    positions.
- Key weights:
  - `continuation_weight: 1.0`
  - `suppression_weight: 1.0`
  - `emit_weight: 0.3`
  - `pre_weight: 0.0`
  - `max_kl_tokens_per_example: 256`
- Training config:
  - 2x A100 80GB
  - per-device train batch 4
  - gradient accumulation 2
  - eval every 25 steps
  - save every 25 steps
  - `num_train_epochs: 1.0`
  - no `max_steps`
  - early stopping enabled:
    `patience: 6`, `threshold: 0.0005`
  - `load_best_model_at_end: true`
  - best metric was eval loss, not natural-generation exact-3 rate.

## Result Summary

Training stopped at step 575, about 0.54 epoch, because loss-based early
stopping fired. This was not manually killed.

Teacher-forced/training metrics looked good:

- final eval loss: 0.007503
- pause target argmax rate: 1.0
- continuation KL: about 0.0006-0.0008

Capability did not collapse:

| group | GSM8K | MATH500 | parse rate |
|---|---:|---:|---:|
| base natural | 0.660 | 0.4467 | 1.0 |
| pause natural | 0.646 | 0.4333 | 1.0 |
| pause forced | 0.652 | 0.4533 | 1.0 |

Safety judge behavior stayed roughly close to base:

| judge | base natural | pause natural | pause forced |
|---|---:|---:|---:|
| HarmBench unsafe-valid | 0.3600 | 0.3700 | 0.3567 |
| LlamaGuard unsafe-valid | 0.5200 | 0.5267 | 0.5100 |
| WildGuard unsafe-valid | 0.4233 | 0.3800 | 0.4133 |

But natural pause emission failed the strict Stage2 standard:

| dataset | natural pause rate | exact 3-pause | off-target pause rate | avg pause count |
|---|---:|---:|---:|---:|
| GSM8K | 1.000 | 0.196 | 0.802 | 8.288 |
| MATH500 | 1.000 | 0.9267 | 0.0033 | 3.377 |
| HarmBench | 1.000 | 0.855 | 0.025 | 3.285 |
| JailbreakBench | 1.000 | 0.920 | 0.010 | 3.170 |
| OR-Bench hard | 1.000 | 0.7967 | 0.020 | 3.470 |
| StrongReject | 1.000 | 0.890 | 0.0133 | 3.217 |
| XSTest | 1.000 | 0.700 | 0.004 | 3.792 |

If the required success rate is >=99% exact natural 3-pause at the right
position, this Stage2 run failed. Earlier review judged it non-blocking for
Stage3 exploration only; that should not be interpreted as Stage2 success.

## Questions

1. What are the most likely root causes of low exact-3 natural pause emission?
   Please rank them.
2. Is this mainly insufficient training steps, or mainly an objective/checkpoint
   selection problem?
3. Did loss-based early stopping at step 575 likely stop too early? Or would
   continuing to 1 epoch risk making pause over-emission worse?
4. Does the current suppression loss actually penalize the failure mode we care
   about in free generation, or only teacher-forced non-pause labels?
5. Is rows-only / pause-token-only training too constrained to reach 99%
   exact-3, or is it the right constraint with a better objective?
6. What Stage2.1 changes would you recommend before re-running 8B?
   Please consider:
   - no-extra-pause / stop-after-K loss
   - natural-generation validation loop every N steps
   - checkpoint selection by exact-3 and off-target pause, not eval loss
   - including GSM8K/MATH/safety-style CoT data or synthetic negative examples
   - decoding constraints versus training objective fixes
   - whether to keep KL transparency and rows-only invariants
7. What minimum experiment should be run before another expensive full 8B run?

Please be critical and concrete. Separate likely causes, code-level fixes,
training/eval protocol changes, and claims we must avoid.
