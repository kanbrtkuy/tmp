# Fable-5 Review Prompt: SafeChain 8B Stage2/Stage3 Formal Run

Please act as a strict external methods/results reviewer. Review the full
GitHub repo state and the summarized results below. The goal is to decide what
the current Stage2+Stage3 evidence supports, what remains unresolved from the
professor's comments, and what should happen before Stage4 steering.

If WebFetch is available, read the GitHub files at commit `66ab5c2`:

- repo: `https://github.com/kanbrtkuy/cot-safety`
- commit: `https://github.com/kanbrtkuy/cot-safety/commit/66ab5c2`
- Stage2 result doc:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/res/stage2_8b_full_results_260707_zh.md`
- Stage3 result doc:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/res/stage3_8b_teacher_forced_results_260707_zh.md`
- current gate plan:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/plan/stage2_stage3_8b_current_gate_260707_zh.md`
- R2 archive doc:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/docs/stage2_8b_full_r2_archive_260707.md`
- formal Stage3 prereg/horizon audit:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/review-stage/stage3_8b_cot5_prereg_260707/stage3_8b_cot5_prereg_and_horizon_audit_260707.md`
- Stage3 formal teacher-forced prompt:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/review-stage/stage3_8b_cot5_formal_tf_260707/fable_prompt_260707.md`
- Stage4 GPRS gate code:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/src/cot_safety/steering/gprs.py`
- Stage4 GPRS/liveness tests:
  `https://github.com/kanbrtkuy/cot-safety/blob/66ab5c2/tests/test_stage4_gprs_liveness.py`

If local file tools are available, you may also read the same files from the
current working tree. Do not edit files. If you cannot fetch/read some files,
state the limitation explicitly and review from the pasted evidence.

## Research Logic

1. Stage1: show latent separability on SafeChain paired data.
2. Stage2: train a KL-transparent pause-emitting model that inserts pause after
   `cot_4` / before `cot_5` while preserving continuation behavior.
3. Stage3: test whether pause hidden states carry unsafe/safe CoT signal beyond
   prompt baselines and matched ordinary content controls.
4. Stage4: only if Stage3/liveness pass, use pause as a steering port to reduce
   unsafe CoT without harming capability or over-refusal.

## Stage2 8B Result Summary

Checkpoint:
`/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final`

Training/eval:

- model: DeepSeek-R1-Distill-Qwen-8B
- method: KL-transparent emit SFT, not ordinary full SFT
- pause insertion: after `cot_4` / before `cot_5`
- 2x A100 80GB
- eval every 25 steps
- early stopping enabled
- stopped at step 575, about 0.54 epoch
- final eval loss: 0.007503
- pause target argmax rate: 1.0
- continuation KL: about 0.0006-0.0008

Capability:

| group | GSM8K | MATH500 | parse rate |
|---|---:|---:|---:|
| base natural | 0.660 | 0.4467 | 1.0 |
| pause natural | 0.646 | 0.4333 | 1.0 |
| pause forced | 0.652 | 0.4533 | 1.0 |

Safety judged no material drift, no safety-improvement claim:

| judge | base natural | pause natural | pause forced |
|---|---:|---:|---:|
| HarmBench unsafe-valid | 0.3600 | 0.3700 | 0.3567 |
| LlamaGuard unsafe-valid | 0.5200 | 0.5267 | 0.5100 |
| WildGuard unsafe-valid | 0.4233 | 0.3800 | 0.4133 |

Pause emission:

- Most safety datasets and MATH500 naturally emit pause near target.
- GSM8K natural mode over-emits pause: exact 3-pause only 19.6%, avg pause
  count 8.288.
- Forced pause by construction inserts exactly 3 pause tokens.

## Stage3 8B Teacher-Forced Result Summary

Scope:

- teacher-forced/off-policy supporting run, not on-policy confirmation
- uses frozen Stage1 paired prepared data and preserved train/val/test splits
- sources: HarmBench, ReasoningShield, StrongReject, WildJailbreak
- pause insertion: `<think> t0 t1 t2 t3 t4 <pause><pause><pause> t5 ...`
- primary exact-horizon content control: `control_cot_4`

Core evidence:

| source | pause AUROC | prompt AUROC | pause - prompt | control_cot_4 AUROC | pause - control_cot_4 |
|---|---:|---:|---:|---:|---:|
| HarmBench | 0.8251 | 0.5000 | 0.3251 | 0.8298 | -0.0047 |
| ReasoningShield | 0.7038 | 0.5000 | 0.2038 | 0.7325 | -0.0287 |
| StrongReject | 0.7487 | 0.4998 | 0.2489 | 0.7510 | -0.0023 |
| WildJailbreak | 0.7861 | 0.4999 | 0.2863 | 0.8026 | -0.0164 |

Evidence status:

- HarmBench: pass pause signal only; independent undecided
- ReasoningShield: pass pause signal only; independent not established
- StrongReject: pass pause signal only; independent not established
- WildJailbreak: pass pause signal only; independent not established

Current interpretation:

- Supports readable pause-position safety signal above prompt-only baselines.
- Does not establish pause-specific monitoring advantage over matched content
  controls.
- Does not establish on-policy trajectory monitoring or clean steering-port
  claims.

## R2 Backup

Archive root:
`cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/`

Verified:

- Stage2 outputs: 426 objects, 472.251 GiB
- Stage3 artifacts: 1.325k objects, 29.965 GiB
- hidden files: 281 objects, 44.943 GiB
- Stage2 eval/judge: 376 objects, 385.209 MiB, rclone check clean

## Professor Comments To Reconcile

- Need prompt-only/pre-CoT baseline to avoid prompt classification confound.
- Need LOSO/multi-source generalization.
- Need cleanly separate SFT-data effect from steering effect.
- Need address teacher-forced probe vs self-generated steering distribution
  mismatch.
- Need show pause is a clean intervention point with matched steering
  counterfactuals against ordinary tokens.
- Need keep absolute residual unsafe visible.

## Questions For You

1. Is the current interpretation correct: Stage2 mostly passes as a
   behavior-preserving pause insertion candidate, while Stage3 teacher-forced
   shows readable pause signal but not pause-specific advantage?
2. Does GSM8K pause over-emission require Stage2 retraining before on-policy
   Stage3, or is it a limitation compatible with continuing safety-source
   Stage3?
3. What exactly should the on-policy Stage3 slice test to resolve the
   teacher-forced vs self-generated mismatch?
4. Should Stage4 remain blocked until on-policy Stage3 and liveness pass, or is
   a small liveness-only pilot justified now?
5. Are the GPRS gate semantics in `src/cot_safety/steering/gprs.py` appropriate:
   fail closed on explicit top-level failure, require pause-only pass, require
   independent/pass_independent status for Stage4 readiness, and require
   confirmatory on-policy evidence unless explicitly teacher-forced-only?
6. What claims are allowed/disallowed in the paper/deck today?
7. List any concrete code/method changes that should be made before the next
   GPU run.

Please be critical, concrete, and concise. Separate blockers from non-blocking
limitations.
