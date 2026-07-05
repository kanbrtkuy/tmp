# R1-8B Source-Expansion k300 监控记录 - 2026-07-04

本文档记录 RunPod A100 上的 R1-8B source-expansion k300 生成任务进度。
只包含聚合状态，不包含 raw prompts 或 raw trajectories。

## 任务配置

| 项目 | 值 |
|---|---|
| Run dir | `/workspace/cot-safety/runs/source_expansion_r1_8b_k300_v1` |
| Log dir | `/workspace/logs/source_expansion_r1_8b_k300_v1` |
| Generator | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Judge | `ReasoningShield/ReasoningShield-1B` |
| Sampling | temperature `0.6`, top_p `0.95`, max_new_tokens `8192` |
| 每轮策略 | 对仍未同时得到 quality-pass safe 与 unsafe CoT 的 prompts，每轮再生成 5 个 candidates |
| 上限 | 每个 prompt 最多 300 samples |

## 数据源目标

| 数据源 | 目标 prompts |
|---|---:|
| WildJailbreak `vanilla_harmful` | 3000 |
| StrongReject full | 313 |
| HarmBench standard | 200 |
| **总计** | **3513** |

## 进度快照

| UTC 时间 | 轮次 | 阶段 | Candidates | Judged | Valid pairs | Active remaining | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| 2026-07-04 03:20 | `045_049` 完成后 | selected | 93975 | 93975 | 2221 | 1292 | 自动进入 `050_054` |
| 2026-07-04 04:05 | `050_054` 完成后 | selected | 100435 | 100435 | 2249 | 1264 | 本轮新增 28 pairs，自动进入 `055_059` |
| 2026-07-04 04:08 | `055_059` | generating | 100435 | 100435 | 2249 | 1264 | GPU 满载，未见 OOM/Traceback |
| 2026-07-04 04:17 | `055_059` | generating | 102355 | 100435 | 2249 | 1264 | 本轮继续生成中，GPU 满载，未见 OOM/Traceback |
| 2026-07-04 04:22 | `055_059` | generating | 103315 | 100435 | 2249 | 1264 | 本轮继续生成中，距离本轮候选目标约 3440 条，未见 OOM/Traceback |
| 2026-07-04 04:23 | `055_059` | generating | 104275 | 100435 | 2249 | 1264 | 本轮继续生成中，距离本轮候选目标约 2480 条，未见 OOM/Traceback |
| 2026-07-04 04:28 | `055_059` | generating | 105235 | 100435 | 2249 | 1264 | 本轮继续生成中，距离本轮候选目标约 1520 条，未见 OOM/Traceback |
| 2026-07-04 04:34 | `055_059` | generating | 106195 | 100435 | 2249 | 1264 | 本轮生成接近完成，距离本轮候选目标约 560 条，未见 OOM/Traceback |
| 2026-07-04 04:36 | `055_059` | judging | 106755 | 100435 | 2249 | 1264 | 本轮生成完成并自动进入 judge，未见 OOM/Traceback |
| 2026-07-04 04:40 | `055_059` | judging | 106755 | 102739 | 2249 | 1264 | judge 正常推进，距离本轮 judge 完成约 4016 条，未见 OOM/Traceback |
| 2026-07-04 04:47 | `055_059` 完成后 | selected | 106755 | 106755 | 2282 | 1231 | 本轮新增 33 pairs，自动进入 `060_064` |
| 2026-07-04 04:51 | `060_064` | generating | 106755 | 106755 | 2282 | 1231 | 新一轮已满载生成，内部批次约 745/960，未见 OOM/Traceback |
| 2026-07-04 04:56 | `060_064` | generating | 107715 | 106755 | 2282 | 1231 | 本轮生成接近一个内部子批次末尾，GPU 满载，未见 OOM/Traceback |
| 2026-07-04 04:58 | `060_064` | generating | 108675 | 106755 | 2282 | 1231 | 本轮继续生成中，候选数正常增长，GPU 满载，未见 OOM/Traceback |
| 2026-07-04 05:03 | `060_064` | generating | 109635 | 106755 | 2282 | 1231 | 本轮继续生成中，距离本轮候选目标约 3275 条，未见 OOM/Traceback |
| 2026-07-04 05:04 | `060_064` | generating | 109635 | 106755 | 2282 | 1231 | 日志继续推进，内部批次约 700/960，GPU 满载，未见 OOM/Traceback |
| 2026-07-04 05:28 | `060_064` 完成后 | selected | 112910 | 112910 | 2311 | 1202 | 本轮新增 29 pairs，自动进入 `065_069` |
| 2026-07-04 06:11 | `065_069` 完成后 | selected | 118920 | 118920 | 2334 | 1179 | 本轮新增 23 pairs，自动进入 `070_074` |
| 2026-07-04 06:55 | `070_074` 完成后 | selected | 124815 | 124815 | 2362 | 1151 | 本轮新增 28 pairs，自动进入 `075_079` |
| 2026-07-04 07:40 | `075_079` 完成后 | selected | 130570 | 130570 | 2383 | 1130 | 本轮新增 21 pairs，自动进入 `080_084` |
| 2026-07-04 08:17 | `080_084` | judging | 136220 | 133642 | 2383 | 1130 | 本轮生成完成并处于 judge 阶段，未见 OOM/Traceback |
| 2026-07-04 08:18 | `080_084` | judging | 136220 | 134410 | 2383 | 1130 | judge 继续推进，GPU 满载，未见 OOM/Traceback |
| 2026-07-04 08:20 | `080_084` | judging | 136220 | 135178 | 2383 | 1130 | judge 继续推进，距离本轮 judge 完成约 1042 条，未见 OOM/Traceback |

## 最新已完成轮次的分数据源覆盖率

截至 `075_079` select 完成：

| 数据源 | 目标 | 有效 pair | 剩余 | 覆盖率 |
|---|---:|---:|---:|---:|
| WildJailbreak `vanilla_harmful` | 3000 | 1964 | 1036 | 65.5% |
| StrongReject full | 313 | 272 | 41 | 86.9% |
| HarmBench standard | 200 | 147 | 53 | 73.5% |
| **总计** | **3513** | **2383** | **1130** | **67.8%** |

## 当前判断

- `075_079` 已完成 generate / judge / select，链路健康。
- `080_084` 已完成 generate，当前处于 judge 阶段。
- 监控期间没有发现 CUDA OOM、Traceback 或任务中断。
- GPU 在生成阶段接近满载，当前无需人工干预。
