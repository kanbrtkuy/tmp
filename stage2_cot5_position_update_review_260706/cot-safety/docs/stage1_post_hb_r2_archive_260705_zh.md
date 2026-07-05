# Stage 1 Post-HB R2 全量归档 - 2026-07-05

本文档记录 2026-07-05 A100 RunPod 节点上 post-HB Stage1 LOSO/retune
实验产物、hidden/archive 产物、日志和代码快照的 Cloudflare R2 归档结构。
本文档不包含 R2 密钥、raw prompts、raw CoTs、completions 或 row-level
prediction JSONL 内容。

## 归档根目录

Cloudflare R2 remote：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/
```

该归档是 2026-07-05 post-HB Stage1 节点的全量工作区快照，主要覆盖：

- `/workspace/cot-safety/runs`
- `/workspace/stage1-results`
- `/dev/shm/cot-safety-hot/runs`
- `/workspace/logs`
- `/workspace/cot-safety` 下的代码、配置、计划、文档和测试目录

不包含 HuggingFace/model cache；只复制了 `/dev/shm/cot-safety-hot/runs`。
`/dev/shm/cot-safety-hot/hf_cache` 可从 HuggingFace 重新下载，不作为实验归档保存。

## 顶层结构

最终顶层目录：

```text
analysis_reports/
configs/
data/
docs/
legacy/
logs/
manifest/
pipelines/
plan/
repo_root/
runs/
scripts/
src/
tests/
```

| Prefix | 内容 |
|---|---|
| `runs/stage1_post_hb_260705_after_hb_n100_loso/` | post-HB Stage1 LOSO 主 run 根目录。 |
| `runs/stage1-results/stage1_post_hb_260705_retune12288_b20/` | `/workspace/stage1-results` 下的 GPU archive root。 |
| `runs/hidden_archives/` | 普通 Stage1/Stage1b hidden archive 目录。它们从 `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives` 分离归档到顶层 `runs/hidden_archives/`，用于恢复 A1/A2/diagnostic hidden 输入。 |
| `runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly/` | excluded-source lead-time confirmation 使用的正式 cot-only hidden archive，只有 `cot_4,cot_8,cot_16,cot_32,cot_64`，不含 `think_last`。 |
| `runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500/` | excluded-source lead-time confirmation 的聚合结果、gate、diagnostics 和 prediction-score JSONL。 |
| `runs/dev_shm/cot-safety-hot/runs/` | `/dev/shm/cot-safety-hot/runs` 的目录式对象备份。 |
| `runs/dev_shm/cot-safety-hot/runs.tar.gz` | 同一 `/dev/shm` runs 目录的整包 sidecar 归档，用于快速恢复。 |
| `runs/dev_shm/cot-safety-hot/runs.tar.gz.sha256` | `runs.tar.gz` 的 sha256 校验文件。 |
| `logs/stage1_post_hb_260705_retune12288_b20/` | Stage1 sequence、R2 backup 和 sidecar 上传日志。 |
| `manifest/r2_full_backup_260705/` | 备份状态、rclone logs、R2 size、tar sidecar manifest。 |
| `configs/`, `scripts/`, `pipelines/`, `src/`, `tests/` | RunPod 节点上的代码快照。 |
| `plan/`, `docs/`, `data/`, `analysis_reports/`, `legacy/` | repo 中对应目录的快照。 |
| `repo_root/` | `README.md`、`pyproject.toml` 和临时 audit helper。 |

## 备份完成状态

备份启动：

```text
2026-07-05T04:57:53+00:00
```

主要目录复制完成后，root-level `copyto` 在 `repo_readme` 一步遇到
R2 `AccessDenied`。原因是该 token 对单文件 `copyto` 需要沿用
`--s3-no-check-bucket`。随后已用 resume 脚本补传 root files、manifest
和最终 size。

最终完成标记：

```text
ALL_DONE_RESUMED 2026-07-05T05:46:14+00:00
```

最终 R2 size：

```text
Total objects: 38.224k (38224)
Total size: 64.186 GiB (68919322192 Byte)
```

2026-07-05 追加关机前审计后，补传了一个漏掉的普通 hidden archive：

```text
runs/hidden_archives/stage1_natural_pairs_8b_a100_1x_loso_strongreject_full/
```

该目录补传 `9` 个文件，大小 `105.772 MiB`。补传后的最终 R2 size：

```text
Total objects: 38.469k (38469)
Total size: 64.545 GiB (69304583047 Byte)
```

关机前 one-way size checks：

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
  excluding ordinary hidden_archives, which are restored from runs/hidden_archives/
```

代码/文档目录的关机前 one-way checks 也已补齐并通过：

```text
configs, scripts, src, pipelines, plan, res, review-stage, docs, tests, data,
legacy, logs: 0 differences
```

`/dev/shm/cot-safety-hot/runs` 先完成目录式备份：

```text
DONE dev_shm_hot_runs at 2026-07-05T05:41:42+00:00
```

随后补充 tar sidecar。RunPod 节点没有 `zstd`，因此归档格式为 `tar.gz`。
sidecar 大小约 461 MiB，上传到：

```text
runs/dev_shm/cot-safety-hot/runs.tar.gz
runs/dev_shm/cot-safety-hot/runs.tar.gz.sha256
```

## 恢复示例

列出归档顶层：

```bash
rclone lsf \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/ \
  --dirs-only --s3-no-check-bucket
```

恢复 Stage1 主 run：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso \
  /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

恢复 GPU archive root：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20 \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

恢复普通 hidden archives：

```bash
mkdir -p /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/hidden_archives \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

恢复 excluded-source lead-time 的 cot-only hidden archive：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly \
  /workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

恢复 excluded-source lead-time confirmation 结果：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500 \
  /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

目录式恢复 `/dev/shm` runs：

```bash
mkdir -p /dev/shm/cot-safety-hot/runs
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/dev_shm/cot-safety-hot/runs \
  /dev/shm/cot-safety-hot/runs \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

整包恢复 `/dev/shm` runs：

```bash
mkdir -p /workspace/restore /dev/shm/cot-safety-hot
rclone copyto \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/dev_shm/cot-safety-hot/runs.tar.gz \
  /workspace/restore/runs.tar.gz \
  --s3-no-check-bucket --progress
rclone copyto \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/dev_shm/cot-safety-hot/runs.tar.gz.sha256 \
  /workspace/restore/runs.tar.gz.sha256 \
  --s3-no-check-bucket --progress
cd /workspace/restore
sha256sum -c runs.tar.gz.sha256
tar -C /dev/shm/cot-safety-hot -xzf runs.tar.gz
```

恢复备份 manifest：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/manifest/r2_full_backup_260705 \
  /workspace/logs/stage1_post_hb_260705_retune12288_b20/r2_full_backup_260705 \
  --s3-no-check-bucket --transfers=8 --checkers=16 --fast-list --progress
```

## 相关文档

- `docs/fable_decision_review_stage1_retune12288_b20_260705.md`
- `docs/stage1_post_hb_retune12288_b20_provenance_260705_zh.md`
- `res/stage1_experiment_inventory_results_260705_zh.md`
- `res/stage1_post_hb_retune12288_b20_results_260705_zh.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`
