# Stage2.3 PPC 1.5B R2 Archive - 2026-07-10

This document records the Cloudflare R2 archive for the 2026-07-09 to
2026-07-10 2xA6000 Stage2.3 PPC 1.5B run.

It contains no Hugging Face token, R2 credential, API key, or local `.env`
value. This is a backup/restore document only; it intentionally does not
summarize or interpret the experimental results.

## Archive Root

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/
```

Final verified R2 size:

```text
Total objects: 159
Total size: 15.568 GiB (16715922732 Byte)
```

## Main Contents

| R2 prefix | Original RunPod path | Local staging path | Contents |
|---|---|---|---|
| `workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000/` | `/workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000` | `cot-safety/runpod_backups/stage23_ppc_1p5b_full_260710/workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000` | Stage2.3 PPC model output directory, including `checkpoint-100`, `checkpoint-147`, `final`, `raw`, tokenizer/config files, and training logs. |
| `workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709/` | `/workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709` | `cot-safety/runpod_backups/stage23_ppc_1p5b_full_260710/workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709` | Key run artifacts, logs, generations, eval files, judge outputs, and Stage3 hidden/report files, excluding `tune_outputs/**`. |
| `manifest/BACKUP_STATUS.md` | local backup manifest | `cot-safety/runpod_backups/stage23_ppc_1p5b_full_260710/BACKUP_STATUS.md` | Backup status and operational notes captured before R2 upload. |

## Exclusions

The archive intentionally excludes:

- `workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709/tune_outputs/**`
- virtualenvs, HF caches, and local credential files

`tune_outputs/**` is not archived because the local emergency backup only had a
partial copy of that intermediate tuning/checkpoint directory. The verified R2
archive therefore preserves the final model output directory and the key run
artifacts, not every intermediate tuning checkpoint byte from the original
RunPod node.

## Verification

The R2 upload was performed from the local emergency backup after the original
RunPod node became unreachable. Because the active R2 token is bucket-scoped,
all R2 commands require `--s3-no-check-bucket`; otherwise rclone may attempt an
S3 bucket preflight and fail with `403 AccessDenied`.

Model output backup check:

```bash
rclone check \
  cot-safety/runpod_backups/stage23_ppc_1p5b_full_260710/workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000 \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000 \
  --s3-no-check-bucket --one-way --size-only --fast-list
```

Result:

```text
0 differences found
53 matching files
```

Key run artifact backup check:

```bash
rclone check \
  cot-safety/runpod_backups/stage23_ppc_1p5b_full_260710/workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709 \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709 \
  --s3-no-check-bucket --exclude 'tune_outputs/**' --one-way --size-only --fast-list
```

Result:

```text
0 differences found
105 matching files
```

Archive size check:

```bash
rclone size \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5 \
  --s3-no-check-bucket --fast-list
```

Result:

```text
Total objects: 159
Total size: 15.568 GiB (16715922732 Byte)
```

## Restore Commands

Restore Stage2.3 PPC model outputs:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000 \
  /workspace/outputs/deepseek_1p5b_stage23_ppc_read_lora_cot5_2xa6000 \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore key run artifacts:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709 \
  /workspace/cot-safety/runs/stage23_ppc_1p5b_full_batched_260709 \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

Restore backup manifest:

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260709-a6000-2x-stage23-ppc-1p5b-full-cot5/manifest \
  /workspace/cot-safety/review-stage/stage23_ppc_backup_manifest_260710 \
  --s3-no-check-bucket --transfers=4 --checkers=8 --fast-list --progress
```

## Notes

- The initial R2 upload attempt from the RunPod node failed before this archive
  because rclone tried an S3 bucket preflight without `--s3-no-check-bucket`.
- The final R2 archive was uploaded from the local emergency backup.
- This document is deliberately limited to backup structure and verification.
  Consult the archived raw JSON/JSONL files directly when reviewing the
  experimental outputs.

