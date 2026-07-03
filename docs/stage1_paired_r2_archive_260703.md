# Stage 1 Paired Natural-Pair R2 Archive - 2026-07-03

This document records the Cloudflare R2 backup layout for the 2026-07-03
Stage 1 paired natural-pair experiments. It intentionally contains no R2
credentials, raw prompts, or raw trajectories.

## Canonical Archive

Cloudflare R2 remote:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

This archive corresponds to the A100 Stage 1 natural-pair work and includes the
workspace artifacts that were present at backup time. It is separate from the
older A6000 Stage1/Stage1b archive:

```text
cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/
```

## Top-Level Layout

The archive root contains:

```text
analysis_reports/
configs/
data/
docs/
logs/
manifest/
pipelines/
plan/
res/
runs/
scripts/
```

Meaning of each prefix:

| Prefix | Contents |
|---|---|
| `analysis_reports/` | Redacted review notes, surface-audit summaries, and analysis reports copied from the repo. |
| `configs/` | Data, model, judge, and experiment configs used by the paired natural-pair runs. |
| `data/` | Small source or derived data files tracked in the workspace. |
| `docs/` | Repository documentation snapshot. |
| `logs/` | Stable RunPod logs captured at backup time. Current post-backup generation logs are not guaranteed to be included. |
| `manifest/` | Backup manifest files, size records, and inventory listings. |
| `pipelines/` | RunPod launcher and queue scripts. |
| `plan/` | Stage 1 plan documents. |
| `res/` | Stage 1 result summaries and compact result artifacts. |
| `runs/` | Stage 1 natural-pair run outputs, hidden-probe outputs, exported pair files, and surface-audit outputs present at backup time. |
| `scripts/` | Data-preparation, export, probe, audit, and backup scripts. |

## Verification Snapshot

Verification performed after upload:

- R2 size: `45,038` objects, `155.521 GiB`.
- Backup-time size record: `45,037` objects, `155.521 GiB`.
- The one-object difference is expected: `manifest/r2_size_after_backup.txt`
  was generated after the initial size measurement.
- `rclone check --one-way --size-only` found `0` differences for the stable
  backed-up trees after excluding post-backup dynamic files.
- R2 multipart cleanup was run after a dry-run. The bucket-level unfinished
  multipart upload list is now empty:

```json
{
  "cot-safety": []
}
```

The earlier Cloudflare dashboard display showing `1-a6000/...` under
`stage1-paired/` was not a completed object prefix. R2 multipart API checks
found no unfinished uploads under `stage1-paired/`; the stale uploads were from
the older `stage1/20260701-a6000/` path and have been cleaned.

## Known Post-Backup Differences

The backup was a point-in-time snapshot. The following files or runs were
created or modified after the archive was made and therefore should not be
expected to appear in the archived snapshot unless a later incremental backup is
performed:

- `configs/data/natural_cot_pair_full_n100_8b_remaining.yaml`
- `scripts/data/adaptive_natural_cot_full_n100_8b_remaining_remote.sh`
- the active follow-up generation run:
  `runs/natural_cot_pair_full_n100_8b_remaining_v1/`
- the active follow-up generation logs:
  `/workspace/logs/natural_cot_full_n100_8b_remaining_v1/`
- backup logs that were appended after their first upload.

## Related GitHub Documents

Planning:

- `plan/stage1_plan.md`
- `plan/stage1_plan_zh.md`
- `plan/stage1_natural_pair_experiment_plan_260703.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`

Results:

- `res/stage1_data_preparation_status_260702.md`
- `res/stage1_data_preparation_status_260702_zh.md`
- `res/stage1_natural_pair_experiment_results_260703.md`
- `res/stage1_natural_pair_experiment_results_260703_zh.md`

Operational setup:

- `docs/runpod_setup.md`
- `docs/stage1_paired_r2_archive_260703_zh.md`

## Restore Examples

List the archive:

```bash
rclone lsf \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/ \
  --recursive --max-depth 4
```

Restore the `res/` summaries:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/res \
  /workspace/cot-safety/res \
  --transfers=16 --checkers=32 --fast-list --progress
```

Restore run outputs:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/runs \
  /workspace/cot-safety/runs \
  --transfers=16 --checkers=32 --fast-list --progress
```
