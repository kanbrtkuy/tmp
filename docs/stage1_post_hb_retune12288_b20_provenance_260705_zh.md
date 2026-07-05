# Stage 1 Post-HB retune12288/b20 Provenance - 2026-07-05

本文档回答一个窄问题：2026-07-05 最新 post-HB Stage1 LOSO
`retune12288_b20` 结果到底用了哪些代码、脚本和配置。本文档不包含 raw
prompts、raw CoTs、completions 或 row-level prediction JSONL。

## 范围

本文档对应的最新 Stage1 结果是：

```text
res/stage1_post_hb_retune12288_b20_results_260705_zh.md
```

它不是 2026-06-30 prompt-only / unified baseline，也不是 2026-07-03
early natural-pair pilot。对应主 run 和归档位置：

```text
RunPod main run:
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso

GPU archive root:
/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20

R2 archive root:
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/

GPU sequence completion:
ALL_STAGE1_SEQUENCE_DONE 2026-07-05T04:21:51+00:00
```

精确复现实验时，优先使用 GPU archive root 里的每个
`*_resolved.yaml`，而不是只看当前 GitHub 模板。当前模板已经和 RunPod/R2
同步，可作为再现入口；但 resolved YAML 才是当次 fold/run 的权威配置快照。

## Git 和代码状态

文档整理时 GitHub `main`：

```text
0b4371c84b39c910340c153311ad35cf146dc42d
```

与本轮实验相关的关键 commit：

| Commit | 作用 |
|---|---|
| `db18bbe715b667bcfce551eea6299e864962433d` | 关机前同步实验代码快照。 |
| `ffd8a92213684d2e971548534b6d119f989a96aa` | `run_stage1_sequence.sh` 增加 hidden archive before cleanup。 |
| `1d30c4094de670d4c72ddd78984ea2d5cd66660b` | 增加 threshold 和 matched-horizon reanalysis 脚本。 |
| `0b4371c84b39c910340c153311ad35cf146dc42d` | docs-only 修正：HF/model cache 不作为必须归档数据。 |

当前 GitHub/RunPod/R2 已同步的关键文件 sha256：

| 文件 | sha256 |
|---|---|
| `pipelines/run_stage1_sequence.sh` | `2d73e47a79ceb50d7feefa81527c4b2368c23e790174ee3477f3de6c5a29c98d` |
| `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh` | `c1d0c0969c430a5da08fd42734f342d03299a6554bf27f6944737b83a6300ce1` |
| `pipelines/runpod_stage1_env.sh` | `57af9e9b6597edd6b708d0b98ba3d2b92185d9dcdc3103f2641baa63c2a8c310` |
| `configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml` | `923e6d6db79acff2a2fe5120ac8aeba46bc4fea7126c64a04b906fb5a7803bcc` |
| `configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml` | `c04fc6207d30135831910878abd088bbca673ec3574f379bfa7b75cd7386ccaa` |

## 总控脚本

主入口：

```text
pipelines/runpod_stage1_post_hb_freeze_then_loso.sh
```

实际使用的核心环境变量/默认值：

| 变量 | 值 |
|---|---|
| `CONFIG` | `configs/data/source_expansion_r1_8b_k300.yaml` |
| `MODEL` | `r1-8b` |
| `RUN_DIR` | `runs/source_expansion_r1_8b_k300_v1` |
| `STAGE1_OUT_ROOT` | `runs/stage1_post_hb_260705_after_hb_n100_loso` |
| `FIXED_BUDGET_SAMPLE_START` | `0` |
| `FIXED_BUDGET_MAX_SAMPLE_IDX` | `100` |
| `REQUIRED_LOSO_SOURCES` | `reasoningshield,strongreject_full,wildjailbreak_vanilla_harmful,harmbench_standard` |
| `MIN_LOSO_SOURCE_PAIRS` | `150` |
| `WJB_TRAINVAL_CAP` | `700` |
| `STAGE1_SEQUENCE_SCRIPT` | `pipelines/run_stage1_sequence.sh` |

该脚本流程：

1. 等待 HB generation wrapper 退出。
2. snapshot HB raw state。
3. 重新运行 final select / summarize。
4. 固定 `sample_idx=0..99` 的 fixed-budget selection。
5. 合并 restored ReasoningShield pair file。
6. 跑 freeze audit、embedding dedup audit、LOSO freeze。
7. 生成 human QA sheet。
8. 跑 CPU text/surface baselines 和 bootstrap CI。
9. 在用户明确允许的情况下，bypass human QA gate，启动 provisional GPU Stage1。

注意：这次 overnight GPU run 没有等待人工 QA 完成，因此 human QA 是 claim gate
caveat；它不改变代码和配置 provenance。

## 数据冻结

fixed-budget selection 输出：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/fixed_budget_samples_000_099/
```

LOSO freeze 输出：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/
```

freeze 输入：

```text
runs/stage1_post_hb_260705_after_hb_n100_loso/fixed_budget_samples_000_099/natural_generated_pairs.jsonl
/workspace/cot-safety/runs/restored_stage1_reasoningshield_pairs_260705/reasoningshield_only_normalized_all.jsonl
```

post-freeze keep pairs：

| Source | Keep pairs |
|---|---:|
| `harmbench_standard` | 152 |
| `reasoningshield` | 335 |
| `strongreject_full` | 277 |
| `wildjailbreak_vanilla_harmful` | 2019 |
| Total | 2783 |

这意味着 Fable 之前指出的 source gate blocker 已修到正确位置：不仅 raw pair
输入检查每源下限，`stage1_loso_freeze_summary.json` 后也检查
`keep_pairs_by_source`。本轮 post-freeze 四源都满足 `MIN_LOSO_SOURCE_PAIRS=150`。

相关 summary/audit 文件：

```text
fixed_budget_samples_000_099/selection_gen_gen_summary.json
freeze_audit_fixed_budget_samples_000_099/audit_summary.json
embedding_dedup_fixed_budget_samples_000_099/embedding_dedup_summary.json
loso_freeze_fixed_budget_samples_000_099/stage1_loso_freeze_summary.json
human_qa_fixed_budget_samples_000_099/stage1_human_qa_sample_summary.json
```

freeze audit 聚合状态：

| 字段 | 值 |
|---|---:|
| `n_input_pairs` | 2783 |
| `n_main_keep` | 2783 |
| `n_dropped` | 0 |
| `n_rejected_during_load` | 0 |
| `n_keep_pairs` in LOSO freeze | 2783 |
| `n_dropped_pairs` in LOSO freeze | 0 |

embedding dedup 使用 TF-IDF fallback，主阈值 `0.90`，near band `0.80..0.90`，
top-k `50`。具体 histogram/top-50 见 `embedding_dedup_summary.json` 和同目录
输出文件。

## GPU Sequence

GPU launcher：

```text
pipelines/run_stage1_sequence.sh
```

该脚本对四个 held-out source 各跑 Stage1 和 Stage1b：

```text
reasoningshield
strongreject_full
wildjailbreak_vanilla_harmful
harmbench_standard
```

每个 fold 先用：

```text
scripts/data/export_normalized_pairs_for_stage1.py
```

生成：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/{source}/
```

再从模板生成 per-fold GPU config：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_gpu_configs/
```

本轮 8 个 GPU run：

```text
stage1_natural_pairs_8b_a100_1x_loso_harmbench_standard
stage1_natural_pairs_8b_a100_1x_loso_reasoningshield
stage1_natural_pairs_8b_a100_1x_loso_strongreject_full
stage1_natural_pairs_8b_a100_1x_loso_wildjailbreak_vanilla_harmful
stage1b_natural_pairs_8b_a100_1x_loso_harmbench_standard
stage1b_natural_pairs_8b_a100_1x_loso_reasoningshield
stage1b_natural_pairs_8b_a100_1x_loso_strongreject_full
stage1b_natural_pairs_8b_a100_1x_loso_wildjailbreak_vanilla_harmful
```

每个 run 的 log、generated fold config、resolved YAML、linear/multilayer 输出和
hidden archive 都在：

```text
/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/{run_name}/
```

对应 R2 路径：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20/{run_name}/
```

## 模板配置

Stage1 模板：

```text
configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml
```

Stage1b 模板：

```text
configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml
```

继承链：

```text
configs/experiment/stage1_natural_pairs_1p5b.yaml
configs/experiment/stage1b_natural_pairs_1p5b.yaml
configs/experiment/stage1_positionscan.yaml
configs/model/deepseek_r1_distill_llama_8b.yaml
configs/model/template_deepseek_r1_distill.yaml
```

模型配置：

```text
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model.max_length: 12288
runtime.torch_dtype: bfloat16
runtime.hidden.compression: uncompressed
runtime.num_gpus: 1
runtime.cuda_visible_devices: "0"
```

Stage1 positions：

```text
cot_0,cot_1,cot_2,cot_3,cot_4,cot_5,cot_6,cot_7,cot_8,cot_9,
cot_10,cot_12,cot_16,cot_24,cot_32,cot_48,cot_64,cot_96,cot_128
```

Stage1b positions：

```text
last_prompt_token,assistant_start,assistant_last,pre_think,think_last,
cot_2,cot_3,cot_4,cot_7
```

Layers：

```text
4,6,7,8,10,12,14,16,17,18,20,21,22,24,25,26,28,30,32
```

Probe runtime：

```text
single_scan_backend: batched
jobs: 16
worker_slots_per_gpu: 16
jobs_per_gpu: 16
scan_batch_size: 4096
scan_eval_batch_size: 8192
multilayer_jobs: 8
multilayer_fallback_device: cpu
cpu_threads_per_job: 2
```

实际 resolved YAML 中的 hidden extraction batch：

| Fold | Stage1 batch | Stage1b batch |
|---|---:|---:|
| `reasoningshield` | 20 | 20 |
| `harmbench_standard` | 18 | 18 |
| `strongreject_full` | 18 | 18 |
| `wildjailbreak_vanilla_harmful` | 18 | 18 |

resolved YAML 的 hot root：

```text
/dev/shm/cot-safety-hot-stage1_post_hb_260705_retune12288_b20
```

## Probe Code Path

Stage1 wrapper：

```text
scripts/run_stage1_positionscan.py
```

Stage1b wrapper：

```text
scripts/run_stage1b_prompt_baseline.py
```

legacy probe/extraction code：

```text
legacy/PauseProbe/scripts/probe/run_position_scan_full.py
legacy/PauseProbe/scripts/probe/extract_hidden_states.py
legacy/PauseProbe/scripts/probe/run_position_scan_batched.py
legacy/PauseProbe/scripts/probe/train_probe.py
```

`run_stage1_sequence.sh` 在每个 run 结束后会把 `/dev/shm` 下的 hidden arrays
先 archive 到 GPU archive root，再清理 `/dev/shm` 对应 hidden 目录。

## CPU Reanalysis And Diagnostics

本轮结果后追加的 CPU reanalysis / diagnostics 主要使用：

```text
scripts/data/run_stage1_threshold_reanalysis.py
scripts/data/run_stage1_matched_horizon_reanalysis.py
scripts/data/run_stage1_score_pooling_reanalysis.py
scripts/data/run_stage1_feature_pooling_reanalysis.py
scripts/data/run_stage1_excluded_leadtime_confirmation.py
scripts/data/run_stage1_hidden_surface_delta_ci.py
scripts/data/audit_stage1_prediction_rows.py
```

excluded-source lead-time 的 minimal hidden 补抽入口：

```text
pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh
```

相关结果文档：

```text
res/stage1_cpu_reanalysis_threshold_matched_horizon_260705.md
res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md
res/stage1_a1_a2_leadtime_diagnostics_260705.md
res/stage1_experiment_inventory_results_260705_zh.md
```

## 环境和恢复

RunPod 环境说明：

```text
docs/runpod_setup.md
```

R2 归档结构和恢复命令：

```text
docs/stage1_post_hb_r2_archive_260705_zh.md
docs/stage1_post_hb_r2_archive_260705.md
```

HF/model cache 不属于必须归档的实验数据，可在新 RunPod 上从 HuggingFace
重新下载。R2 中保留的是实验 run、hidden archive、日志、代码快照、配置和文档。

## Provenance Caveats

1. 这次 GPU run 是 human-QA gate bypass 后的 provisional run；论文式 claim
   仍需 human QA 和 Fable gate。
2. 当前结果已被 Fable-5 复审为 negative/control，不应表述为 hidden-state
   superiority。
3. 当前 GitHub 模板和 RunPod/R2 已同步，但 per-fold `*_resolved.yaml` 才是精确
   runtime config。
4. `stage1_prediction_row_audit_summary.json` 的 `passes=false` 主要用于标记
   coverage/row-count diagnostics，不代表 freeze source gate 失败；source gate
   应看 `stage1_loso_freeze_summary.json` 的 `keep_pairs_by_source`。
