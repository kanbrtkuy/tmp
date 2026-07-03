# Stage 1 Paired Natural-Pair R2 归档 - 2026-07-03

本文档记录 2026-07-03 Stage 1 paired natural-pair 实验在 Cloudflare R2
上的备份结构。本文档不包含 R2 密钥、raw prompts 或 raw trajectories。

## Canonical Archive

Cloudflare R2 remote：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

这个归档对应 A100 节点上完成的 Stage 1 natural-pair 工作，并包含备份时
workspace 中存在的相关实验产物。它和更早的 A6000 Stage1/Stage1b 归档是分开的：

```text
cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/
```

## 顶层结构

归档根目录包含：

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

每个 prefix 的含义：

| Prefix | 内容 |
|---|---|
| `analysis_reports/` | 从 repo 复制的 redacted review notes、surface-audit summaries 和分析报告。 |
| `configs/` | paired natural-pair runs 使用的数据、模型、judge 和实验配置。 |
| `data/` | workspace 中跟踪的小型源数据或派生数据文件。 |
| `docs/` | 仓库文档快照。 |
| `logs/` | 备份时稳定的 RunPod 日志；备份之后仍在运行的生成日志不保证包含在内。 |
| `manifest/` | 备份 manifest、size 记录和 inventory listings。 |
| `pipelines/` | RunPod launcher 和 queue scripts。 |
| `plan/` | Stage 1 plan 文档。 |
| `res/` | Stage 1 结果总结和 compact result artifacts。 |
| `runs/` | 备份时存在的 Stage 1 natural-pair run outputs、hidden-probe outputs、exported pair files 和 surface-audit outputs。 |
| `scripts/` | 数据准备、导出、probe、audit 和备份脚本。 |

## 校验快照

上传后完成的校验：

- R2 当前大小：`45,038` objects，`155.521 GiB`。
- 备份时记录大小：`45,037` objects，`155.521 GiB`。
- 多出的 1 个 object 是预期差异：`manifest/r2_size_after_backup.txt`
  是初次 size measurement 之后生成的。
- 对稳定备份树执行 `rclone check --one-way --size-only`，排除备份后的动态文件后
  结果为 `0` differences。
- 先 dry-run 后执行了 R2 multipart cleanup。现在 bucket-level unfinished
  multipart upload 列表为空：

```json
{
  "cot-safety": []
}
```

Cloudflare dashboard 之前在 `stage1-paired/` 下显示的 `1-a6000/...`
不是 completed object prefix。R2 multipart API 检查显示 `stage1-paired/`
下没有 unfinished uploads；残留上传来自更早的 `stage1/20260701-a6000/`
路径，并且已经清理。

## 已知的备份后差异

这个归档是 point-in-time snapshot。下面这些文件或 run 是备份之后创建或修改的，
因此除非之后再做 incremental backup，否则不应期待它们出现在当前归档快照中：

- `configs/data/natural_cot_pair_full_n100_8b_remaining.yaml`
- `scripts/data/adaptive_natural_cot_full_n100_8b_remaining_remote.sh`
- 当前仍在推进的 follow-up generation run：
  `runs/natural_cot_pair_full_n100_8b_remaining_v1/`
- 当前 follow-up generation logs：
  `/workspace/logs/natural_cot_full_n100_8b_remaining_v1/`
- 首次上传后继续追加的 backup logs。

## 相关 GitHub 文档

计划文档：

- `plan/stage1_plan.md`
- `plan/stage1_plan_zh.md`
- `plan/stage1_natural_pair_experiment_plan_260703.md`
- `plan/stage1_natural_pair_experiment_plan_260703_zh.md`

结果文档：

- `res/stage1_data_preparation_status_260702.md`
- `res/stage1_data_preparation_status_260702_zh.md`
- `res/stage1_natural_pair_experiment_results_260703.md`
- `res/stage1_natural_pair_experiment_results_260703_zh.md`

运行与恢复说明：

- `docs/runpod_setup.md`
- `docs/stage1_paired_r2_archive_260703.md`

## 恢复示例

列出归档：

```bash
rclone lsf \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/ \
  --recursive --max-depth 4
```

恢复 `res/` 结果总结：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/res \
  /workspace/cot-safety/res \
  --transfers=16 --checkers=32 --fast-list --progress
```

恢复 run outputs：

```bash
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/runs \
  /workspace/cot-safety/runs \
  --transfers=16 --checkers=32 --fast-list --progress
```
