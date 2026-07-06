# Stage2 8B Full R2 Archive - 2026-07-07

This document records the Cloudflare R2 archive locations for the 2026-07-06
to 2026-07-07 2xA100 8B full Stage2 run and the follow-up Stage3
teacher-forced supporting run.

It contains no R2 credentials, HuggingFace tokens, API keys, or local `.env`
files.

## Archive Root

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/
```

## Main Artifacts

Stage2 model/checkpoint outputs:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/
```

Stage2 eval/judge results:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100/
```

Local final checkpoint on the RunPod node:

```text
/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final
```

Local eval root on the RunPod node:

```text
/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100
```

Stage3 teacher-forced artifacts:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/workspace/cot-safety/
```

Important Stage3 subpaths:

```text
workspace/cot-safety/legacy/PauseProbe/data/hidden/stage3_stage1_paired_*_kl_transparent_8b_cot5_2xa100/
workspace/cot-safety/legacy/PauseProbe/data/stage3_stage1_paired_*_kl_transparent_8b_cot5_2xa100/
workspace/cot-safety/legacy/PauseProbe/logs/stage3_stage1_paired_*_kl_transparent_8b_cot5_2xa100/
workspace/cot-safety/legacy/PauseProbe/runs/probes/stage3_stage1_paired_*_kl_transparent_8b_cot5_2xa100_single/
workspace/cot-safety/legacy/PauseProbe/runs/probes/stage3_stage1_paired_*_kl_transparent_8b_cot5_2xa100_pooled/
workspace/cot-safety/stage3/
```

## Verified Eval Backup

The Stage2 eval/judge result directory was copied to R2 after completion and
checked with `rclone check --one-way --size-only`.

Verification result:

```text
Total objects: 376
Total size: 385.209 MiB
0 differences found
376 matching files
```

Top-level eval files archived:

```text
capability_summary.csv
pause_emission_summary.csv
safety_summary.csv
resolved_config.yaml
generations/
judges/
logs/
logs_failed_missing_vllm_260707/
shards/
```

## Verified Stage3 Backup

The Stage3 teacher-forced supporting artifacts were copied to the same archive
root after all four sources completed. Final `rclone size --fast-list` checks:

```text
workspace/cot-safety/stage3
Total objects: 1.325k
Total size: 29.965 GiB

workspace/cot-safety/legacy/PauseProbe/data/hidden
Total objects: 281
Total size: 44.943 GiB

workspace/cot-safety/legacy/PauseProbe/runs/probes
Total objects: 3.417k
Total size: 500.734 MiB

workspace/cot-safety/runs/logs
Total objects: 1
Total size: 335.319 KiB

workspace/cot-safety/configs/experiment
Total objects: 89
Total size: 104.162 KiB

workspace/cot-safety/configs/runtime
Total objects: 6
Total size: 7.500 KiB

workspace/cot-safety/review-stage/stage3_8b_cot5_prereg_260707
Total objects: 1
Total size: 5.156 KiB
```

No `rclone copy` process remained on the node after these checks.

## Restore Commands

Restore Stage2 eval/judge results:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100 \
  /workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100 \
  --transfers=16 --checkers=32 --fast-list --progress
```

Restore Stage2 outputs/checkpoints:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100 \
  /workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100 \
  --transfers=16 --checkers=32 --fast-list --progress
```

## Notes

- The final model was first moved from `/dev/shm` to `/workspace`.
- The R2 watcher uploaded checkpoints and the final model to avoid `/dev/shm`
  and `/workspace` space pressure.
- Stage3 teacher-forced supporting run completed on the same checkpoint.
- WJB Stage3 artifacts were generated under `/dev/shm/cot-safety-hot/stage3/`
  and copied to `/workspace/cot-safety/stage3/` before R2 backup.
- On-policy Stage3 and Stage4 steering are still gated.
