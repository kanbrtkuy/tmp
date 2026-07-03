# Stage 1 Paired 8B Remaining R2 归档 - 2026-07-04

本文档记录 R1-8B remaining-prompt natural-pair follow-up 生成任务，以及
8B A100 节点上一些小型 Stage 1 surface-control 产物的 Cloudflare R2 增量备份。
本文档不包含 R2 密钥、raw prompts 或 raw trajectories。

## 归档根目录

Cloudflare R2 remote：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/
```

这是一个增量归档，不重复上传之前完整的 A100 Stage 1 paired archive：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

## 包含内容

顶层结构：

```text
code_snapshot/
logs/
manifest/
runs/
shm/
```

| Prefix | 内容 |
|---|---|
| `runs/natural_cot_pair_full_n100_8b_remaining_v1/` | R1-8B remaining-prompt follow-up 的生成、judge、select 和 merged-pair 输出。 |
| `runs/stage1_loso_surface/` | 8B A100 节点上的小型 Stage 1 LOSO/surface-control 输出。 |
| `logs/natural_cot_full_n100_8b_remaining_v1/` | R1-8B remaining run 的 driver、generate、judge、select、merge 和 summarize logs。 |
| `logs/stage1_loso_surface_260703/` | LOSO/surface-control run 的日志。 |
| `logs/stage1_surface_parallel_260703/` | parallel surface-control jobs 的日志。 |
| `logs/stage1_trunc_split_260703/` | truncation split jobs 的日志。 |
| `logs/cot_natural_env_setup/` | A100 节点环境配置日志。 |
| `code_snapshot/` | remaining run 和 surface controls 使用到的小型 config、judge prompt 和 script 快照。 |
| `manifest/` | 源端磁盘占用、源端文件清单、备份元数据和 R2 size 记录。 |
| `shm/cot-safety-smoke/` | 8B A100 节点 `/dev/shm/cot-safety-smoke` 下的 smoke hidden-state extraction artifacts。 |

备份时也检查了 `/dev/shm/cot-safety-hot/`，其中只有空目录结构，因此没有需要复制到 object store 的文件。
`/dev/shm` 的 inventory 已记录在 `manifest/` 下。

## 8B Remaining Run 结果

本轮 follow-up run 完成后得到：

| 项目 | 数量 |
|---|---:|
| 本轮尝试的 remaining prompts | 613 |
| 生成候选 | 28,300 |
| 已 judge 候选 | 28,300 |
| 新选出的 high-quality safe pairs | 83 |
| 继承的 safe/original pairs | 663 |
| 合并后的 R1-8B safe/original pairs | 746 |

合并后的 pair 输出为：

```text
runs/natural_cot_pair_full_n100_8b_remaining_v1/natural_safe_pairs_merged.jsonl
```

## 校验快照

备份从 RunPod A100 节点完成于：

```text
2026-07-03T16:36:48Z
```

code snapshot 的两个补充文件完成于：

```text
2026-07-03T16:48:42Z
```

`/dev/shm/cot-safety-smoke` artifacts 的备份和校验完成于：

```text
2026-07-03T16:51:45Z
```

清理测试 marker 后的 R2 size：

```text
Total objects: 244
Total size: 1.409 GiB (1512944797 Byte)
```

size-only 校验结果：

| 源路径 | R2 目标 | 结果 |
|---|---|---|
| `/workspace/cot-safety/runs/natural_cot_pair_full_n100_8b_remaining_v1` | `runs/natural_cot_pair_full_n100_8b_remaining_v1` | `0 differences`，55 个文件匹配 |
| `/workspace/logs/natural_cot_full_n100_8b_remaining_v1` | `logs/natural_cot_full_n100_8b_remaining_v1` | `0 differences`，44 个文件匹配 |
| `/workspace/cot-safety/runs/stage1_loso_surface` | `runs/stage1_loso_surface` | `0 differences`，76 个文件匹配 |
| `/dev/shm/cot-safety-smoke` | `shm/cot-safety-smoke` | `0 differences`，16 个文件匹配 |

操作说明：当前 Cloudflare R2 token 会拒绝 bucket create/check 操作，因此上传使用了
`--s3-no-check-bucket`。直接 object upload 和 size-only verification 均成功。

## 相关 GitHub 文档

- `docs/stage1_paired_r2_archive_260703.md`
- `docs/stage1_paired_r2_archive_260703_zh.md`
- `docs/stage1_paired_8b_remaining_r2_archive_260704.md`
- `res/stage1_natural_pair_experiment_results_260703.md`
- `res/stage1_natural_pair_experiment_results_260703_zh.md`
- `plan/stage1_natural_pair_experiment_plan_260703.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`

## 恢复示例

列出增量归档：

```bash
rclone lsf \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/ \
  --recursive --max-depth 4 --s3-no-check-bucket
```

恢复 merged R1-8B remaining run：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/runs/natural_cot_pair_full_n100_8b_remaining_v1 \
  /workspace/cot-safety/runs/natural_cot_pair_full_n100_8b_remaining_v1 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```

恢复日志：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/logs/natural_cot_full_n100_8b_remaining_v1 \
  /workspace/logs/natural_cot_full_n100_8b_remaining_v1 \
  --s3-no-check-bucket --transfers=16 --checkers=32 --fast-list --progress
```
