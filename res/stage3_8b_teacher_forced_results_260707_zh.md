# Stage3 8B Teacher-Forced 结果记录 - 2026-07-07

本文记录 8B KL-transparent pause checkpoint 上的 Stage3 teacher-forced
supporting run。该结果用于判断 pause 位置是否有可读信号；它不是
on-policy confirmatory Stage3 结果。

## 运行设置

- Stage2 checkpoint:
  `/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final`
- Stage1 paired freeze:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/`
- pause insertion:
  `cot_4` 之后、`cot_5` 之前，即 `<think> t0 t1 t2 t3 t4 <pause><pause><pause> t5 ...`
- Stage3 configs:
  `configs/experiment/stage3_intra_pause_probe_stage1_paired_*_8b_cot5_2xa100.yaml`
- RunPod:
  2x A100 80GB

## Core Evidence

| source | status | pause AUROC | prompt AUROC | pause - prompt | independent margin | pause-only vs best content |
|---|---|---:|---:|---:|---:|---:|
| HarmBench | pass pause signal only; independent undecided | 0.8251 | 0.5000 | 0.3251 | 0.0016 | 0.0044 |
| ReasoningShield | pass pause signal only; independent not established | 0.7038 | 0.5000 | 0.2038 | -0.0284 | -0.0302 |
| StrongReject | pass pause signal only; independent not established | 0.7487 | 0.4998 | 0.2489 | -0.0027 | 0.0003 |
| WildJailbreak | pass pause signal only; independent not established | 0.7861 | 0.4999 | 0.2863 | -0.0736 | -0.0088 |

Bootstrap CI for pause minus prompt baseline:

| source | CI |
|---|---|
| HarmBench | [0.2820, 0.3674] |
| ReasoningShield | [0.1686, 0.2377] |
| StrongReject | [0.2144, 0.2852] |
| WildJailbreak | [0.2731, 0.2998] |

## Exact-Horizon Content Control

The preregistered primary content control is `control_cot_4`, because both
`control_cot_4` and `pause_0` have seen original CoT content through `t4`.
Validation-selected best `control_cot_4` test AUROC:

| source | pause AUROC | control_cot_4 AUROC | pause - control_cot_4 |
|---|---:|---:|---:|
| HarmBench | 0.8251 | 0.8298 | -0.0047 |
| ReasoningShield | 0.7038 | 0.7325 | -0.0287 |
| StrongReject | 0.7487 | 0.7510 | -0.0023 |
| WildJailbreak | 0.7861 | 0.8026 | -0.0164 |

## Interpretation

The teacher-forced Stage3 run supports a limited positive result:

- pause/post-pause positions reliably contain unsafe/safe readout signal above
  prompt-only baselines across all four sources;
- however, pause positions do not beat matched ordinary content controls;
- therefore this run does not establish a pause-specific monitoring advantage.

Allowed wording:

- "The Stage2 pause model creates readable pause-position safety signal above
  prompt baselines in a teacher-forced Stage3 screen."
- "The stronger claim that pause is better than ordinary content-token readout
  is not established."

Not allowed from this run:

- on-policy trajectory-monitoring claim;
- clean steering-port claim;
- Stage4 readiness claim without liveness and matched-token steering
  counterfactuals.

## Next Gate

The next decisive Stage3 experiment is on-policy:

1. Generate multiple CoTs per prompt from the 8B pause model.
2. Judge each generated CoT for unsafe content.
3. Extract pause hidden states from generated trajectories.
4. Evaluate within-prompt separability against prompt and matched content
   controls.

Stage4 should remain gated on on-policy Stage3 plus liveness tests.

## R2 Backup

Stage3 teacher-forced artifacts have been backed up under:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/
```

Verified R2 sizes:

| path | objects | size |
|---|---:|---:|
| `workspace/cot-safety/stage3` | 1.325k | 29.965 GiB |
| `workspace/cot-safety/legacy/PauseProbe/data/hidden` | 281 | 44.943 GiB |
| `workspace/cot-safety/legacy/PauseProbe/runs/probes` | 3.417k | 500.734 MiB |
| `workspace/cot-safety/runs/logs` | 1 | 335.319 KiB |
| `workspace/cot-safety/configs/experiment` | 89 | 104.162 KiB |
| `workspace/cot-safety/configs/runtime` | 6 | 7.500 KiB |
