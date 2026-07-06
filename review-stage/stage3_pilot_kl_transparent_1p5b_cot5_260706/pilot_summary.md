# Stage3 Pilot Summary: KL-Transparent 1.5B cot5 Pause

Date: 2026-07-06

Scope: pilot sanity check only. The Stage2 model is a pilot checkpoint, not a full Stage2 SFT model. The goal of this run is to test whether the Stage3 framework works end to end on Stage1 paired data and whether pause positions show a usable safety signal beyond prompt-only baselines.

Model:
- `/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`

Data:
- Stage1 paired prepared data from `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/`
- Folds used: HarmBench, ReasoningShield, StrongReject, WildJailbreak
- Stage3 uses `--preserve_input_splits`, `--no_heldout_sources`, and `--skip_base_data_prep`, so it preserves the Stage1 paired train/val/test split.

Clean run:
- Previous mixed batch 32/48/40 HarmBench outputs were archived remotely under `/workspace/cot-safety/archive/stage3_mixed_batch_runs/20260706_081515_mixed_batch32_48_40/`.
- Previous old-style WJB outputs were archived remotely under `/workspace/cot-safety/archive/stage3_old_wjb_runs/20260706_093611_old_wjb_stage3_pre_clean_rerun/`.
- Clean rerun used extraction batch size 40.
- HarmBench, ReasoningShield, StrongReject, and WildJailbreak completed.

## Results

| Fold | Status | Pause AUROC | Prompt AUROC | Pause - Prompt | Pair-cluster CI | True Content Control AUROC | Independent Margin |
|---|---|---:|---:|---:|---:|---:|---:|
| HarmBench | pass_pause_signal_only_independent_not_established | 0.8120 | 0.5000 | 0.3120 | [0.2688, 0.3550] | 0.8225 | -0.0105 |
| ReasoningShield | pass_pause_signal_only_independent_not_established | 0.7196 | 0.5000 | 0.2196 | [0.1833, 0.2557] | 0.7263 | -0.0185 |
| StrongReject | pass_pause_signal_only_independent_not_established | 0.7347 | 0.5000 | 0.2347 | [0.1982, 0.2723] | 0.7218 | -0.0193 |
| WildJailbreak | pass_pause_signal_only_independent_not_established | 0.7983 | 0.5002 | 0.2980 | [0.2850, 0.3105] | 0.7552 | -0.0077 |

Interpretation:
- The pilot framework runs end to end on paired Stage1 data.
- Pause hidden states show a clear signal beyond prompt-only baseline in all four clean folds.
- The stronger claim that pause/post-pause beats matched true content control is not established in this pilot under the current confirmatory rule.
- WJB has a positive pause-only margin over true content control (0.0431, CI [0.0320, 0.0538]), but the confirmatory pause/post-pause selection still reports independent signal as not established.
- Because this uses a Stage2 pilot checkpoint, these results should not be treated as final evidence about the full method.

Artifacts:
- `evidence/harmbench_stage3_evidence_report.json`
- `evidence/reasoningshield_stage3_evidence_report.json`
- `evidence/strongreject_stage3_evidence_report.json`
- `evidence/wjb_stage3_evidence_report.json`
- `stage3_remaining_folds.log`
- `stage3_wjb_clean.log`
