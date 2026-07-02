# Stage 1 计划

最后更新：2026-07-02。

本文档记录 SafeChain 项目 Stage 1 的实验计划：验证
`DeepSeek-R1-Distill-Qwen-1.5B` 在 teacher forcing 读取轨迹时，
hidden states 是否包含能够区分 unsafe-continuation 与
safe-refusal / safe-redirection 轨迹的信号。

## 范围

Stage 1 不是 steering 实验，而是后续 pause-token 训练和 steering 的
measurement gate。

Stage 1 的主 claim 必须保持窄口径：

> 在固定同一个 unsafe prompt 的条件下，R1-1.5B teacher-forced hidden
> states 可以线性区分外部生成的 unsafe-continuation 轨迹和外部生成的
> safe-refusal 轨迹。

在证明 natural rollout transfer 和 benign-prompt specificity 之前，不要声称
我们已经得到一个通用的 on-policy unsafe-reasoning detector。

## 数据组成

### Primary paired data

| Cell | 来源 | 数量 | 轨迹来源 | 角色 | 状态 |
|---|---:|---:|---|---|---|
| `U->U` | A-prime HarmThoughts + ReasoningShield | 1096 pairs | OpenAI unsafe-preserving paraphrase | Primary positive | 已完成 |
| `U->S` | 同一批 A-prime prompts | 1096 pairs | OpenAI safe rewrite / controlled-clean | Primary negative | 已完成 |
| `U->U/U->S` | B-prime HarmThoughts + ReasoningShield | 1460 pairs | 同上 | Sensitivity only | 已完成 |

Primary train/test 只能使用 paired `U->U` vs `U->S`。`S->S` 不能进入
primary train/test，否则 probe 可能学到“prompt 是 benign”，从而重新引入
prompt-type confound。

### Safe-prompt diagnostic data

这些数据不是 Stage 1 主 claim 的训练/测试数据，但在投入 Stage 2 预算前必须补齐，
用于检查 specificity 和 over-refusal 风险。

| Cell | 推荐来源 | 目标规模 | 轨迹来源 | 角色 | 状态 |
|---|---:|---:|---|---|---|
| `S->S hard-safe` | XSTest safe | 全量 / 约 250 | R1-1.5B natural rollout，judge-filtered | Over-refusal specificity | 待做 |
| `S->S adversarial-benign` | WildJailbreak `adversarial_benign` | 约 500 | R1-1.5B natural rollout，judge-filtered | Jailbreak-style benign control | 待做 |
| `S->S hard-benign` | OR-Bench hard benign | 约 500 | R1-1.5B natural rollout，judge-filtered | Hard benign control | 待做 |
| `S->S easy anchor` | GSM8K 和/或 Alpaca-style benign instructions | 约 200 | R1-1.5B natural rollout，judge-filtered | 普通 benign floor | 待做 |
| `S->S provenance matched` | 从上述 rows 抽 150-200 条 | 约 150-200 | R1 rollout + OpenAI paraphrase | Provenance diagnostic | 待做 |

Fable5 建议 Stage 1 不使用 SQuAD 作为 `S->S` 来源，因为它是
context-passage QA 格式，和当前 chat-style prompt 分布差异过大。

## 当前状态清单

### Primary 数据准备

- [x] 选择 primary unsafe trajectory 来源：HarmThoughts 和 ReasoningShield。
- [x] 从原始数据中抽取 rewrite seed rows。
- [x] 使用 OpenAI 生成 same-prompt safe rewrites (`U->S`)。
- [x] 执行 safe polish / controlled-clean passes。
- [x] 对 unsafe side (`U->U`) 执行 OpenAI unsafe-preserving paraphrase。
- [x] 将 unsafe paraphrase 结果分层为 A/B/holdout tiers。
- [x] 执行 full A/B quality audit。
- [x] 冻结原始 A-prime 和 B-prime manifests。
- [x] 执行 structural completeness audit。
- [x] 生成 completeness-clean manifests。
- [ ] 基于 clean manifests 冻结 Stage 1 prompt-group splits。
- [ ] 从 clean manifests 导出 Stage 1 teacher-forcing JSONL。
- [ ] 编写并运行 unit tests + tiny synthetic freeze/export dry-run。
- [ ] 运行 CPU/text baselines。
- [ ] 运行 GPU hidden-state extraction 和 probe training。

### Safe-prompt diagnostic 数据

- [ ] 构建 safe-prompt source manifest：XSTest safe、WildJailbreak
  adversarial-benign、OR-Bench hard benign、GSM8K/Alpaca easy benign。
- [ ] 生成 R1-1.5B natural `S->S` rollouts。
- [ ] 使用 judge/filter 清理 malformed 或 unsafe outputs。
- [ ] 对 `S->S` trajectories 执行 completeness audit。
- [ ] 构建 150-200 row OpenAI-paraphrased `S->S` provenance-matched subset。
- [ ] 将 `S->S` false-positive / specificity evaluation 加入 Stage 1 报告。

## Frozen 数据产物

Primary clean manifests：

- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl`
  - count: `1096`
  - sha256: `abcd42b47e61511306dc207dfc05fe4333496ced745dead3d27445d1b9af5fd8`
- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl`
  - count: `1460`
  - sha256: `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0`
- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json`
  - clean filter 的完整 provenance。

Completeness filtering 删除了 1 条 A-prime row，因为它的 `unsafe_reasoning`
以省略号结尾，疑似没有闭合：

- dropped count: `1`
- drop manifest:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl`
- dropped manifest sha256:
  `471c3c5828effa922ddd26e5d03260f27231ce4144b27a18c1528eb5f6224ed3`

## 数据准备代码结构

所有 Stage 1 数据准备代码都位于 `scripts/data/`。
中文代码用法说明见 `plan/stage1_data_prep_guide_zh.md`；英文代码旁说明见
`scripts/data/README_stage1_data_prep.md`。

### Source extraction

- `scripts/data/extract_harmthoughts_rewrite_seeds.py`
  - 将 HarmThoughts 的 sentence-level annotations 组装成 rewrite seed rows。
- `scripts/data/extract_reasoningshield_rewrite_seeds.py`
  - 将 ReasoningShield 原始 rows 转换成 rewrite seed rows。

### Safe rewrite pipeline

- `scripts/data/generate_safe_rewrites_openai.py`
  - OpenAI batch 主入口，用于 unsafe-to-safe rewrite 和 controlled-clean polish。
  - 使用 `configs/data/unsafe_to_safe_rewrite_*.yaml`。
- `scripts/data/validate_safe_rewrite_pairs.py`
  - 本地验证 safe rewrite pairs。

### Unsafe-side paraphrase and quality

- `scripts/data/pilot_unsafe_preserving_paraphrase_openai.py`
  - unsafe-preserving paraphrase 的 pilot 工具。
- `scripts/data/repair_openai_unsafe_paraphrases.py`
  - full unsafe-preserving paraphrase batch / repair pipeline。
- `scripts/data/stratify_openai_paraphrase_quality.py`
  - 构建 A/B/holdout quality tiers。
- `scripts/data/audit_openai_control_samples.py`
  - sample-level audit 工具。
- `scripts/data/audit_openai_full_ab.py`
  - full A/B combined quality audit 和原始 frozen manifest export。

### Manifest integrity and Stage 1 export

- `scripts/data/audit_rewrite_completeness.py`
  - 对 generated reasoning/final fields 做 structural completeness audit。
- `scripts/data/filter_frozen_manifests_by_completeness.py`
  - 生成 completeness-clean A-prime/B-prime manifests，不覆盖原始 frozen manifests。
- `scripts/data/freeze_stage1_prompt_splits.py`
  - 对一个或多个 clean manifests 冻结 prompt-group train/validation/test splits。
- `scripts/data/export_safe_rewrite_pairs_for_stage1.py`
  - 导出 Stage 1 hidden-state extraction 使用的 teacher-forcing rows。
  - Manifest mode 中使用：
    - unsafe side: `manifest.unsafe_reasoning`
    - safe side: `manifest.safe_reasoning`
    - 默认 render mode: `reasoning_only`

### Review / audit bundle

- `scripts/data/build_fable5_pipeline_review_bundle.py`
  - 构建供 Fable5/Claude review 使用的 compact bundle。
- 当前 plan 依赖的 Fable5 reviews：
  - `analysis_reports/fable5_pipeline_integrity_review_redacted_260702.md`
  - `analysis_reports/fable5_completeness_filter_review_260702.md`
  - `analysis_reports/fable5_completeness_filter_fix_review_260702.md`
  - `analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md`

## 基本用法

从 repo 根目录 `cot-safety/` 执行。

运行 completeness audit：

```bash
python3 scripts/data/audit_rewrite_completeness.py
```

重新生成 completeness-clean manifests：

```bash
python3 scripts/data/filter_frozen_manifests_by_completeness.py --force
```

在 clean manifests 被接受后冻结 prompt-group splits：

```bash
python3 scripts/data/freeze_stage1_prompt_splits.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --output-jsonl runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-summary runs/stage1_clean_prompt_splits/stage1_prompt_splits_summary.json
```

导出 A-prime Stage 1 rows：

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/A_prime_reasoning_only
```

导出 B-prime sensitivity rows：

```bash
python3 scripts/data/export_safe_rewrite_pairs_for_stage1.py \
  --input-manifest runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl \
  --split-manifest runs/stage1_clean_prompt_splits/stage1_prompt_splits.jsonl \
  --output-dir runs/stage1_exports/B_prime_reasoning_only
```

在 unit tests 和 tiny synthetic freeze/export dry-run 通过前，不启动 CPU
baselines 或 GPU extraction。

## Stage 1 gate criteria

进入 Stage 2 前需要满足：

- A-prime hidden-state probe 以预注册 margin 超过 text/surface baselines。
- 至少三个 split seeds 下结果稳定。
- Prompt-only probe 接近 chance。
- Shuffled-label probe 接近 chance。
- B-prime sensitivity 与 A-prime 方向一致。
- Source-model / provenance controls 不能解释主效果。
- `S->S` diagnostic 在 paired validation 冻结阈值下 false-positive rate 可接受。
- 在投入 pause-token training 预算前检查 natural rollout transfer。
