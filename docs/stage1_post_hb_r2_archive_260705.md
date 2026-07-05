# Stage 1 Post-HB R2 Archive - 2026-07-05

This document records the Cloudflare R2 archive for the 2026-07-05 A100 RunPod
post-HB Stage1 LOSO/retune run. It contains aggregate artifacts only: no R2
secrets, raw prompts, raw CoTs, completions, or row-level prediction JSONL
contents are reproduced here.

## Archive Root

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/
```

The archive covers:

- `/workspace/cot-safety/runs`
- `/workspace/stage1-results`
- `/dev/shm/cot-safety-hot/runs`
- `/workspace/logs`
- code/config/plan/doc/test snapshots from `/workspace/cot-safety`

Core experiment restoration does not depend on HuggingFace or model caches.
Before shutdown, `/dev/shm/cot-safety-hot/hf_cache` was also uploaded as an
optional cache snapshot. It can save future download time, but it is not unique
experiment data and can be re-downloaded from HuggingFace.

## Key Prefixes

| Prefix | Contents |
|---|---|
| `runs/stage1_post_hb_260705_after_hb_n100_loso/` | Main post-HB Stage1 LOSO run directory. |
| `runs/stage1-results/stage1_post_hb_260705_retune12288_b20/` | GPU archive root copied from `/workspace/stage1-results`. |
| `runs/hidden_archives/` | Ordinary Stage1/Stage1b hidden archives, split out from the GPU archive root. |
| `runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly/` | Official excluded-source cot-only hidden archives for lead-time confirmation. |
| `runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500/` | Excluded-source lead-time confirmation outputs. |
| `runs/dev_shm/cot-safety-hot/runs/` | Directory backup of `/dev/shm/cot-safety-hot/runs`. |
| `runs/dev_shm/cot-safety-hot/runs.tar.gz` | Tar sidecar of the same `/dev/shm` runs directory. |
| `runs/dev_shm/cot-safety-hot/hf_cache/` | Optional HuggingFace cache blob snapshot. It is not required for experiment restore; HF snapshot symlinks were not archived as a tar sidecar. |
| `manifest/r2_full_backup_260705/` | Backup logs, status, manifests, and final size records. |
| `configs/`, `scripts/`, `pipelines/`, `src/`, `tests/`, `legacy/` | RunPod code/config/test snapshots. |
| `plan/`, `docs/`, `res/`, `review-stage/` | Planning, archive docs, result summaries, and Fable reviews. |

## Final Shutdown Audit

After the shutdown audit, one missing ordinary hidden archive was uploaded:

```text
runs/hidden_archives/stage1_natural_pairs_8b_a100_1x_loso_strongreject_full/
```

Final R2 size after the fix:

```text
Total objects: 38.469k (38469)
Total size: 64.545 GiB (69304583047 Byte)
```

An optional `/dev/shm/cot-safety-hot/hf_cache` blob snapshot was then uploaded
before shutdown:

```text
/dev/shm/cot-safety-hot/hf_cache
  -> runs/dev_shm/cot-safety-hot/hf_cache
  0 differences, 17 matching files
  Total size: 14.966 GiB (16070067110 Byte)
```

R2 size including that optional cache:

```text
Total objects: 38.486k (38486)
Total size: 79.511 GiB (85374650157 Byte)
```

One-way size checks before shutdown:

```text
/workspace/cot-safety/runs
  -> runs/
  0 differences, 1967 matching files

/workspace/stage1-results/stage1_post_hb_260705_after_hb_n100_loso
  -> runs/stage1-results/stage1_post_hb_260705_after_hb_n100_loso
  0 differences, 11441 matching files

/dev/shm/cot-safety-hot/runs
  -> runs/dev_shm/cot-safety-hot/runs
  0 differences, 11423 matching files

/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20
  -> runs/stage1-results/stage1_post_hb_260705_retune12288_b20
  0 differences, 12400 matching files
  excluding ordinary hidden_archives, which restore from runs/hidden_archives/
```

Code/config/doc/test snapshot checks also passed for:

```text
configs, scripts, src, pipelines, plan, res, review-stage, docs, tests, data,
legacy, logs
```

## Restore Commands

Restore the main run:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso \
  /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

Restore the GPU archive root:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20 \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

Restore ordinary hidden archives:

```bash
mkdir -p /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/hidden_archives \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

Restore excluded-source cot-only lead-time hidden archives:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

Restore `/dev/shm` runs:

```bash
mkdir -p /dev/shm/cot-safety-hot/runs
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/dev_shm/cot-safety-hot/runs \
  /dev/shm/cot-safety-hot/runs \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

## Related Files

- `docs/stage1_post_hb_r2_archive_260705_zh.md`
- `docs/stage1_post_hb_retune12288_b20_gap_audit_260705.md`
- `res/stage1_experiment_inventory_results_260705_zh.md`
- `res/stage1_post_hb_retune12288_b20_results_260705_zh.md`
- `review-stage/stage1_auto_improve_260705/REVIEW_STATE.json`
