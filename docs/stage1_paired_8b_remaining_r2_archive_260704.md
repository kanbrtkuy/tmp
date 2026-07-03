# Stage 1 Paired 8B Remaining R2 Archive - 2026-07-04

This document records the Cloudflare R2 incremental backup for the follow-up
R1-8B remaining-prompt natural-pair generation run and small post-backup Stage 1
surface-control artifacts. It intentionally contains no R2 credentials, raw
prompts, or raw trajectories.

## Archive Root

Cloudflare R2 remote:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/
```

This is an incremental archive. It does not duplicate the earlier full A100
Stage 1 paired archive:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

## What It Contains

Top-level layout:

```text
code_snapshot/
logs/
manifest/
runs/
shm/
```

| Prefix | Contents |
|---|---|
| `runs/natural_cot_pair_full_n100_8b_remaining_v1/` | Follow-up R1-8B remaining-prompt generation, judging, selection, and merged-pair outputs. |
| `runs/stage1_loso_surface/` | Small Stage 1 LOSO/surface-control outputs present on the 8B A100 node. |
| `logs/natural_cot_full_n100_8b_remaining_v1/` | Driver, generation, judging, selection, merge, and summary logs for the R1-8B remaining run. |
| `logs/stage1_loso_surface_260703/` | Logs for the LOSO/surface-control run. |
| `logs/stage1_surface_parallel_260703/` | Logs for the parallel surface-control jobs. |
| `logs/stage1_trunc_split_260703/` | Logs for the truncation split jobs. |
| `logs/cot_natural_env_setup/` | Environment setup logs from the A100 node. |
| `code_snapshot/` | Small snapshot of the config, judge prompt, and scripts used for the remaining run and surface controls. |
| `manifest/` | Source disk usage, source file inventory, backup metadata, and R2 size record. |
| `shm/cot-safety-smoke/` | `/dev/shm/cot-safety-smoke` smoke hidden-state extraction artifacts from the 8B A100 node. |

`/dev/shm/cot-safety-hot/` was checked during backup and contained only empty
directory scaffolding, so there were no object-store files to copy for that
path. The `/dev/shm` inventory is stored under `manifest/`.

## 8B Remaining Run Result

The completed follow-up run produced:

| Item | Count |
|---|---:|
| Remaining prompts attempted | 613 |
| Generated candidates | 28,300 |
| Judged candidates | 28,300 |
| New selected high-quality safe pairs | 83 |
| Inherited safe/original pairs | 663 |
| Merged R1-8B safe/original pairs | 746 |

The merged pair output is:

```text
runs/natural_cot_pair_full_n100_8b_remaining_v1/natural_safe_pairs_merged.jsonl
```

## Verification Snapshot

Backup completed from the RunPod A100 node at:

```text
2026-07-03T16:36:48Z
```

The code snapshot was completed with two post-check additions at:

```text
2026-07-03T16:48:42Z
```

The `/dev/shm/cot-safety-smoke` artifacts were backed up and checked at:

```text
2026-07-03T16:51:45Z
```

R2 size after cleanup:

```text
Total objects: 244
Total size: 1.409 GiB (1512944797 Byte)
```

Size-only checks:

| Source | R2 target | Result |
|---|---|---|
| `/workspace/cot-safety/runs/natural_cot_pair_full_n100_8b_remaining_v1` | `runs/natural_cot_pair_full_n100_8b_remaining_v1` | `0 differences`, 55 matching files |
| `/workspace/logs/natural_cot_full_n100_8b_remaining_v1` | `logs/natural_cot_full_n100_8b_remaining_v1` | `0 differences`, 44 matching files |
| `/workspace/cot-safety/runs/stage1_loso_surface` | `runs/stage1_loso_surface` | `0 differences`, 76 matching files |
| `/dev/shm/cot-safety-smoke` | `shm/cot-safety-smoke` | `0 differences`, 16 matching files |

Operational note: Cloudflare R2 rejected bucket create/check operations for the
available token, so the upload used `--s3-no-check-bucket`. Direct object upload
and size-only verification succeeded.

## Related GitHub Documents

- `docs/stage1_paired_r2_archive_260703.md`
- `docs/stage1_paired_r2_archive_260703_zh.md`
- `docs/stage1_paired_8b_remaining_r2_archive_260704_zh.md`
- `res/stage1_natural_pair_experiment_results_260703.md`
- `res/stage1_natural_pair_experiment_results_260703_zh.md`
- `plan/stage1_natural_pair_experiment_plan_260703.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`

## Restore Examples

List the incremental archive:

```bash
rclone lsf \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/ \
  --recursive --max-depth 4 --s3-no-check-bucket
```

Restore the merged R1-8B remaining run:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/runs/natural_cot_pair_full_n100_8b_remaining_v1 \
  /workspace/cot-safety/runs/natural_cot_pair_full_n100_8b_remaining_v1 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

Restore the logs:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/logs/natural_cot_full_n100_8b_remaining_v1 \
  /workspace/logs/natural_cot_full_n100_8b_remaining_v1 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```
