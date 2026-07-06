# Stage2/Stage3 Pilot R2 Archive - 2026-07-06

This document records the Cloudflare R2 archive for the 2026-07-06 2xA6000
RunPod node used for the 1.5B KL-transparent pause Stage2 pilot and Stage3
paired-data pilot.

It contains no R2 credentials, HuggingFace tokens, API keys, or local `.env`
files.

## Archive Root

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/
```

## Scope

The backup captures the project-relevant contents of the active GPU node:

| R2 prefix | Source path | Contents |
|---|---|---|
| `workspace/cot-safety/` | `/workspace/cot-safety` | Code snapshot, configs, scripts, docs, plans, results, review packets, Stage3 probe artifacts, and hidden/probe outputs already under the repo. |
| `workspace/data/` | `/workspace/data` | Stage2 pause-SFT data and Stage2/Stage3 prepared data present on the node. |
| `workspace/outputs/` | `/workspace/outputs` | Stage2 pilot checkpoints and trainer outputs. |
| `workspace/logs/` | `/workspace/logs` | Training, eval, install, download, Stage3, and backup logs. |
| `workspace/models/` | `/workspace/models` | Downloaded base and judge model files present on the node. |
| `manifest/` | `/workspace/cot-safety/review-stage/stage2_stage3_backup_260706` | Backup README, node inventory, package freeze files, R2 size record, and backup log. |

The backup intentionally excludes:

- `/workspace/venvs`
- `/workspace/hf_cache`
- `.git`, `.pytest_cache`, `__pycache__`, `*.pyc`
- `.env*` files and local secrets

Rationale: virtualenvs and HF caches are reproducible and can contain local
credential state. The model/checkpoint artifacts that are not easily reproduced
are preserved under `workspace/models/` and `workspace/outputs/`.

## Key Artifacts

Stage2 pilot checkpoint:

```text
workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final
```

Stage3 paired-data pilot summary:

```text
workspace/cot-safety/review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/pilot_summary.md
```

Stage3 evidence reports:

```text
workspace/cot-safety/review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/evidence/
```

Fable reviews:

```text
workspace/cot-safety/review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/fable_stage3_pilot_concise_review_260706.md
workspace/cot-safety/review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/fable_stage3_pilot_wjb_followup_260706.md
```

Stage1 prepared paired data used by Stage3:

```text
workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/
```

## Backup Status

Backup command started on the RunPod node at `2026-07-06T10:18:55Z` and
completed at `2026-07-06T11:01:02Z`.

An additional `--copy-links` correction was run for the WildGuard vLLM
compatibility directory:

```text
workspace/models/judges/wildguard_vllm_head_dim128/
```

The first pass skipped this directory's symlinked files, while the correction
materialized them as ordinary files so the restored path is directly usable.

Final R2 size after completion:

```text
Total objects: 6.347k (6347)
Total size: 191.456 GiB (205574724820 Byte)
```

WildGuard vLLM compatibility directory after the `--copy-links` correction:

```text
Total objects: 39 (39)
Total size: 13.501 GiB (14496930379 Byte)
```

## Restore Commands

Restore the repository snapshot:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/cot-safety \
  /workspace/cot-safety \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore Stage2 outputs/checkpoints:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/outputs \
  /workspace/outputs \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore data:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/data \
  /workspace/data \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore downloaded models:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/models \
  /workspace/models \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore logs and manifest:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/workspace/logs \
  /workspace/logs \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress

rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/manifest \
  /workspace/cot-safety/review-stage/stage2_stage3_backup_260706 \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

## Notes

- `analysis_reports/latest` is a symlink and is skipped by the backup command;
  the underlying target directories are copied normally.
- `workspace/models/judges/wildguard_vllm_head_dim128/` is backed up as
  materialized files after a separate `--copy-links` correction pass.
- This is a pilot archive, not a final 8B Stage2/Stage3 archive.
- Hidden files are included if they live under `/workspace/cot-safety`; full
  future 1.5B/8B Stage3 hidden archives should also be backed up explicitly
  after formal runs.
