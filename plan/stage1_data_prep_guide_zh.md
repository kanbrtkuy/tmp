# Stage 1 数据准备代码说明

这是 Stage 1 数据准备代码的中文说明入口。相关脚本位于
`scripts/data/`，当前 primary 输出是一组 completeness-clean frozen
manifests，用于 same-prompt `U->U` vs `U->S` probing。

项目级计划见：

- `plan/stage1_plan.md`
- `plan/stage1_plan_zh.md`

当前数据准备结果快照见：

- `res/stage1_data_preparation_status_260702.md`
- `res/stage1_data_preparation_status_260702_zh.md`

## Pipeline 结构

```text
raw source data
  -> extract rewrite seeds
  -> OpenAI safe rewrite / controlled-clean polish
  -> OpenAI unsafe-preserving paraphrase
  -> paraphrase quality stratification
  -> full A/B audit
  -> frozen A-prime/B-prime manifests
  -> completeness audit and clean manifests
  -> Stage 1 split freeze
  -> Stage 1 teacher-forcing export
  -> CPU text/surface baselines
```

Stage 1 primary 数据应使用 completeness-clean manifests：

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl
```

## Natural same-prompt CoT pair pilot

A-prime/B-prime 的 OpenAI rewrite audit 显示 surface baseline 太强后，下一条
诊断路径是自然采样 same-prompt pair：对已经有原始 unsafe CoT 的 prompt，用
产生该 unsafe CoT 的同一个 source model 重新采样，再用 safety judge 和本地
quality checks 选出一条高质量、自然生成的 safe CoT。

主要文件：

```text
configs/data/natural_cot_pair_pilot.yaml
scripts/data/run_natural_cot_pair_pipeline.py
```

GPU 节点上的典型命令顺序：

```bash
python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml prepare

python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml generate --model r1-8b

python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml generate --model r1-32b

python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml judge

python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml select

python3 scripts/data/run_natural_cot_pair_pipeline.py \
  --config configs/data/natural_cot_pair_pilot.yaml summarize
```

重要输出：

```text
runs/natural_cot_pair_pilot_v1/prompt_manifest.jsonl
runs/natural_cot_pair_pilot_v1/unsafe_reference_manifest.jsonl
runs/natural_cot_pair_pilot_v1/unsafe_reference_pool.jsonl
runs/natural_cot_pair_pilot_v1/candidates_<model>.jsonl
runs/natural_cot_pair_pilot_v1/judged_candidates_<model>.jsonl
runs/natural_cot_pair_pilot_v1/natural_safe_pairs.jsonl
runs/natural_cot_pair_pilot_v1/natural_pair_summary.json
```

`generation.max_new_tokens` 是生成上限，不是长度控制。pilot 先用 `8192`，
如果 hit-cap rate 明显，尤其是长 CoT 上被截断，再升到 `16384` 或 `32768`。
这一轮保留 safe/unsafe 的自然长度差，不强行做长度或风格匹配。

硬件提醒：full bf16 `r1-32b` 通常不能放进单张 48GB A6000。运行 32B leg
之前，需要更大显存、多卡 tensor parallelism，或者明确记录使用量化 checkpoint。

## 脚本索引

### 1. Raw source extraction

#### `extract_harmthoughts_rewrite_seeds.py`

用途：

- 将 HarmThoughts 原始 sentence-level annotations 转换为 grouped rewrite
  seed rows。
- 保留 prompt id、category、model name、word counts、trace labels 等元数据。

典型角色：

```text
data/harmthoughts_raw/*
  -> data 或 runs 中的 rewrite seed JSONL
```

#### `extract_reasoningshield_rewrite_seeds.py`

用途：

- 将 ReasoningShield rows 转换为 rewrite seed rows。
- 保留 prompt、unsafe reasoning、unsafe final answer、source split 和 category
  metadata。

典型角色：

```text
data/reasoningshield_raw/*
  -> data 或 runs 中的 rewrite seed JSONL
```

### 2. Safe rewrite and polish

#### `generate_safe_rewrites_openai.py`

用途：

- 准备、提交、监控、收集 OpenAI batch jobs。
- 用于 unsafe-to-safe rewrite 和 controlled-clean polish。
- 验证生成的 safe reasoning / final answer 长度和结构。

关键配置文件：

```text
configs/data/unsafe_to_safe_rewrite_harmthoughts_all1018_polish_v5_controlled_clean.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round2.yaml
configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round3.yaml
```

典型命令模式：

```bash
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> prepare
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> submit
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> status
python3 scripts/data/generate_safe_rewrites_openai.py --config <config.yaml> collect
```

复现已有 run 时，应优先使用对应 run 目录中记录的 exact config 和 subcommands。

#### `validate_safe_rewrite_pairs.py`

用途：

- 对生成的 safe rewrite pair files 做本地验证。
- 检查缺失字段、word counts 和明显结构问题。

### 3. Unsafe-side OpenAI paraphrase

#### `pilot_unsafe_preserving_paraphrase_openai.py`

用途：

- 小规模 pilot，用于测试 OpenAI 是否能做 label-preserving unsafe-side
  paraphrase，同时不新增危险细节、不洗白 unsafe 标签。

这个脚本用于 full unsafe-side paraphrase pipeline 之前的可行性测试。

#### `repair_openai_unsafe_paraphrases.py`

用途：

- Main batch / repair pipeline for unsafe-preserving paraphrase。
- 生成 OpenAI-processed unsafe side，用来降低 `U->U` 和 `U->S` 之间的
  provenance mismatch。

重要输出目录：

```text
runs/openai_unsafe_paraphrase_only_v1/
```

#### `stratify_openai_paraphrase_quality.py`

用途：

- 从 unsafe paraphrase 结果构建 A/B/holdout quality strata。
- A-tier 更严格，用作 primary A-prime dataset。
- B-tier 更大，用作 sensitivity。

重要输出目录：

```text
runs/openai_unsafe_paraphrase_only_v1/quality_strata_v1/
```

### 4. OpenAI audit and frozen manifest export

#### `audit_openai_control_samples.py`

用途：

- Safe/unsafe control samples 的 sample-level audit helper。
- 在提交 full audit batch 之前做抽样检查。

#### `audit_openai_full_ab.py`

用途：

- Full combined A/B row audit。
- 审核 unsafe paraphrase quality、safe rewrite mode、pair alignment。
- 导出原始 frozen manifests：

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/
```

重要输出：

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/B_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/dropped_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/manifest_hashes.json
```

当前原始 frozen counts：

- A-prime: `1097`
- B-prime: `1460`
- dropped: `60`

### 5. Completeness audit and clean manifests

#### `audit_rewrite_completeness.py`

用途：

- 对 generated safe/unsafe fields 做 structural completeness audit。
- 检查 empty fields、too-short fields、可疑结尾、未闭合 code fences、
  unbalanced `<think>` tags 等 hard incompleteness signals。
- 报告 diagnostics，但不打印文本 excerpts。

运行：

```bash
python3 scripts/data/audit_rewrite_completeness.py
```

默认输出：

```text
analysis_reports/rewrite_completeness_audit_260702.json
analysis_reports/rewrite_completeness_audit_260702.md
```

#### `filter_frozen_manifests_by_completeness.py`

用途：

- 创建 frozen A-prime/B-prime manifests 的 completeness-clean copy。
- 不修改原始 frozen manifest directory。
- 如果 `--input-dir` 和 `--output-dir` 相同会 fail fast。
- 检查 duplicate / missing `pair_id` rows。
- 检查 A-prime/B-prime 跨文件 `pair_id` overlap。

运行：

```bash
python3 scripts/data/filter_frozen_manifests_by_completeness.py --force
```

默认输出：

```text
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json
runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/manifest_hashes.json
```

当前 clean counts：

- A-prime: `1096`
- B-prime: `1460`
- completeness dropped: `1`

### 6. Stage 1 split freeze

#### `freeze_stage1_prompt_splits.py`

用途：

- 对一个或多个 manifest files 冻结 prompt-group train/validation/test splits。
- Grouping 使用 manifest 中的 normalized prompt hash，确保同一个 prompt 的
  safe/unsafe pairs 进入同一个 split。
- 应在 export 前对 clean A-prime 和 B-prime manifests 执行。

示例：

```bash
python3 scripts/data/freeze_stage1_prompt_splits.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --output-jsonl runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --summary-json runs/stage1_clean_prompt_splits/stage1_prompt_splits_summary.json
```

状态：

- Tiny synthetic dry-run 已通过。
- 真实 clean-manifest prompt split freeze 已在 RunPod CPU 节点通过。
- Frozen split summary：`2556` pairs、`1670` prompt groups，
  train/val/test prompt groups = `1503/84/83`。

### 7. Stage 1 export

#### `export_safe_rewrite_pairs_for_stage1.py`

用途：

- 导出 hidden-state extraction / probe training 使用的 teacher-forcing rows。
- 在 `--input-manifest` 模式中直接使用 frozen manifest 字段：
  - unsafe side: `unsafe_reasoning`
  - safe side: `safe_reasoning`
  - prompt: `prompt`
  - 默认校验 hashes。

Manifest mode 的重要默认行为：

- `render_mode = reasoning_only`
- `source = paired_openai_control_manifest`
- `require_manifest_hashes = true`
- 除非显式设置 `--allow-unfrozen-split`，否则必须提供 `split_manifest`。

A-prime export 示例：

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/A_prime_reasoning_only
```

B-prime sensitivity export 示例：

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/B_prime_reasoning_only
```

状态：

- 真实 `reasoning_only` export 已在 RunPod CPU 节点通过。
- A-prime export：`2192` rows / `1096` pairs。
- B-prime export：`2920` rows / `1460` pairs。

### 8. CPU text baselines

#### `run_stage1_text_baselines.py`

用途：

- 从导出的 `normalized/{train,val,test}.jsonl` rows 运行 CPU-only surface
  baselines。
- 用来检查 hidden-state probes 是否只是学到了浅层文本 artifact。

默认 baselines：

- length-only logistic regression。
- prompt-only TF-IDF logistic regression。
- word TF-IDF logistic regression。
- word bag-of-words logistic regression。
- character n-gram TF-IDF logistic regression。
- first-sentence-removed TF-IDF logistic regression。

示例：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
python3 scripts/data/run_stage1_text_baselines.py \
  --export-dir runs/stage1_exports/A_prime_reasoning_only \
  --output-dir runs/stage1_text_baselines/A_prime \
  --n-jobs 8
```

注意：

- 脚本默认拒绝 train/val/test 之间存在 `match_family` overlap。
- original-unsafe vs OpenAI-paraphrased provenance classifier 继续跳过，直到
  有经过 review 且 pair-id 对齐的 original unsafe source。
- 这个脚本的 tiny fixture tests 已在 RunPod CPU 节点通过。

### 9. External review bundle

#### `build_fable5_pipeline_review_bundle.py`

用途：

- 构建 Fable5/Claude review 使用的 compact、redacted summaries。
- Review bundle 不打印完整 unsafe text。

相关 review 输出：

```text
analysis_reports/fable5_pipeline_integrity_review_redacted_260702.md
analysis_reports/fable5_completeness_filter_review_260702.md
analysis_reports/fable5_completeness_filter_fix_review_260702.md
analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md
```

## Safe-prompt diagnostic data

Primary paired data 已经准备好，但 `S->S` diagnostic data 仍待准备。

已批准的 safe-prompt diagnostic sources：

- XSTest safe：hard-safe over-refusal diagnostic。
- WildJailbreak `adversarial_benign`：adversarial benign control。
- OR-Bench hard benign：hard-benign over-refusal diagnostic。
- GSM8K 和/或 Alpaca-style benign instructions：easy benign anchor。

轨迹生成：

- 优先使用 R1-1.5B natural rollouts。
- 使用 open-source judges 和 local completeness checks 过滤。
- 只为 provenance diagnostic 增加一个 150-200 row OpenAI-paraphrased `S->S`
  subset。

这些 rows 不能混入 primary `U->U` vs `U->S` train/test split。

## 可复现性说明

- `runs/` 下的 primary data outputs 是 run artifacts。如果它们没有被 git
  track，请使用 `res/stage1_data_preparation_status_260702.md` 或中文版本中的
  SHA-256 校验本地副本。
- 不要覆盖原始 frozen manifests。Stage 1 应使用 completeness-clean directory。
- 执行顺序是：freeze/export tests 和 tiny synthetic dry-run，真实
  clean-manifest prompt split freeze，真实 `reasoning_only` Stage 1 export，
  然后 CPU text/surface baselines。在 CPU/text baselines 和 prompt-only
  controls 运行并 review 之前，不启动 GPU extraction。
