# Stage2 8B Full R2 Archive - 2026-07-07

This document records the Cloudflare R2 archive locations for the 2026-07-06
to 2026-07-07 2xA100 8B full Stage2 run.

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
- Stage3 has not been started from this checkpoint yet; it is gated on Fable
  review of the Stage2 results.
