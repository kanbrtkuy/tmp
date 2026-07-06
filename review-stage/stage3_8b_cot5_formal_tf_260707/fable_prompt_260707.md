# Fable Review Prompt: Stage2+Stage3 8B CoT5 Formal Teacher-Forced Run

Please review this as a strict external methods/results audit. The goal is to
decide what the current Stage2+Stage3 evidence supports and what must happen
before Stage4.

## Research Logic

1. Stage1: show latent separability on SafeChain paired data.
2. Stage2: train a KL-transparent pause-emitting model that inserts pause after
   `cot_4` / before `cot_5` while preserving continuation behavior.
3. Stage3: test whether pause hidden states carry unsafe/safe CoT signal beyond
   prompt baselines and matched ordinary content controls.
4. Stage4: only if Stage3/liveness pass, use pause as a steering port to reduce
   unsafe CoT without harming capability or over-refusal.

## Stage2 8B Result

Checkpoint:
`/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final`

Key Stage2 metrics:

- final eval loss: 0.007503
- pause target argmax rate: 1.0
- continuation KL: about 0.0006-0.0008
- GSM8K: base 0.660, pause natural 0.646, pause forced 0.652
- MATH500: base 0.4467, pause natural 0.4333, pause forced 0.4533
- safety judges showed no material drift; no safety-improvement claim
- GSM8K natural pause over-emission remains a limitation, but previous Fable
  review judged it non-blocking for Stage3 safety-source probing

## Stage3 8B Teacher-Forced Run

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

Evidence report status:

- HarmBench: pass pause signal only; independent undecided
- ReasoningShield: pass pause signal only; independent not established
- StrongReject: pass pause signal only; independent not established
- WildJailbreak: pass pause signal only; independent not established

R2 archive root:
`cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/`

R2 backup checks:

- `workspace/cot-safety/stage3`: 1.325k objects, 29.965 GiB
- `workspace/cot-safety/legacy/PauseProbe/data/hidden`: 281 objects, 44.943 GiB
- `workspace/cot-safety/legacy/PauseProbe/runs/probes`: 3.417k objects, 500.734 MiB

Relevant local docs:

- `res/stage2_8b_full_results_260707_zh.md`
- `res/stage3_8b_teacher_forced_results_260707_zh.md`
- `review-stage/stage3_8b_cot5_prereg_260707/stage3_8b_cot5_prereg_and_horizon_audit_260707.md`

## Questions

1. Is the interpretation correct: readable pause signal above prompt baseline,
   but no pause-specific advantage over matched content controls?
2. Does this teacher-forced result justify moving to on-policy Stage3/liveness,
   or is any Stage2 retraining needed first?
3. What exactly should the on-policy Stage3 slice test so that it resolves the
   teacher-forced vs self-generated distribution mismatch?
4. Should Stage4 remain blocked until on-policy Stage3 passes, or can a small
   liveness-only pilot run now?
5. What claims should be allowed and disallowed in a paper/deck?

Please be critical and concise, but include concrete next steps.
