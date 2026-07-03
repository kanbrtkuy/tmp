# Stage 1 数据准备状态

日期：2026-07-02。

本文档记录 Stage 1 当前的数据准备状态。它是数据准备结果快照，不是实验结果。

## 总结

用于 Stage 1 主实验的 paired unsafe-prompt 数据已经准备好，可以进入
prompt split freeze 和 export 阶段。

当前 primary 数据：

- A-prime primary：`1096` same-prompt pairs。
- B-prime sensitivity：`1460` same-prompt pairs。
- Pair cells：
  - `U->U`：unsafe prompt + OpenAI unsafe-preserving paraphrased reasoning。
  - `U->S`：同一个 unsafe prompt + OpenAI safe rewritten reasoning。

Primary paired data 所需的 OpenAI API 工作已经完成：

- safe rewrite：完成。
- controlled-clean safe polish：完成。
- unsafe-preserving paraphrase：完成。
- full A/B row audit：完成。
- manifest freeze 和 completeness-clean pass：完成。

Safe-prompt diagnostic data (`S->S`) 尚未准备。

后续 2026-07-03 natural-pair 工作已归档到 Cloudflare R2：
`cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/`。
归档结构见 `docs/stage1_paired_r2_archive_260703_zh.md`；natural-pair 结果记录在
`res/stage1_natural_pair_experiment_results_260703_zh.md`。

## Primary clean manifests

Stage 1 推荐使用 completeness-clean manifests：

| Manifest | Count | SHA-256 |
|---|---:|---|
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/A_prime_manifest.jsonl` | 1096 | `abcd42b47e61511306dc207dfc05fe4333496ced745dead3d27445d1b9af5fd8` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/B_prime_manifest.jsonl` | 1460 | `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0` |

Completeness-clean provenance：

- Summary:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_filter_summary.json`
- Compatibility hash summary:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/manifest_hashes.json`
- Completeness filter script:
  `scripts/data/filter_frozen_manifests_by_completeness.py`
- Completeness rule source:
  `scripts/data/audit_rewrite_completeness.py:any_strong_incomplete`

## 原始 frozen manifests

Full A/B audit 在 completeness filtering 前导出了以下原始 frozen manifests：

| Manifest | Count | SHA-256 |
|---|---:|---|
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/A_prime_manifest.jsonl` | 1097 | `86b654c9b6bc8e7bb899071ff8ea522637759c3f99878dc51d69f2fd66753e47` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/B_prime_manifest.jsonl` | 1460 | `002cabe87edd60a3539aec30f67a73cf4f90a1dfd6117c021eab2a05bfef8cf0` |
| `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/dropped_manifest.jsonl` | 60 | `4732b840250c0e5cc48c61d39ea64186ac4bbab977119ec3a55ebcef5b581d74` |

原始 A/B audit provenance 存储在：

- `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/manifest_hashes.json`

## Completeness filtering 结果

Completeness filtering 只移除了 1 条 A-prime row：

- Dropped count: `1`。
- Dropped field: `unsafe_reasoning`。
- Reason: `strong_completeness_incomplete`，具体为 `ends_with_ellipsis`。
- Dropped manifest:
  `runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean/completeness_dropped_manifest.jsonl`
- Dropped manifest SHA-256:
  `471c3c5828effa922ddd26e5d03260f27231ce4144b27a18c1528eb5f6224ed3`

Filter 后验证：

- clean A-prime hard incomplete counts:
  - `safe_reasoning`: `0`
  - `safe_final_answer`: `0`
  - `unsafe_reasoning`: `0`
- clean B-prime hard incomplete counts:
  - `safe_reasoning`: `0`
  - `safe_final_answer`: `0`
  - `unsafe_reasoning`: `0`
- Idempotency check:
  - 对 clean manifests 再跑一次 completeness filter，产生 `0` drops。
  - A-prime 和 B-prime 在 idempotency check 中 input/output hash 不变。

Fable5 已 review completeness filter，并同意进入 unit tests 和 tiny synthetic
dry-run：

- `analysis_reports/fable5_completeness_filter_review_260702.md`
- `analysis_reports/fable5_completeness_filter_fix_review_260702.md`

## 数据设计决定

Fable5 确认的 Stage 1 数据组成如下：

| Cell | Source | Size | Role | Primary train/test |
|---|---:|---:|---|---|
| `U->U` | A-prime HarmThoughts + ReasoningShield | 1096 | Primary positive | yes |
| `U->S` | A-prime same prompts | 1096 | Primary negative | yes |
| `U->U/U->S` | B-prime | 1460 pairs | Sensitivity only | no headline |
| `S->S` | XSTest safe | about 250 | Hard-safe diagnostic | no |
| `S->S` | WildJailbreak `adversarial_benign` | about 500 | Adversarial-benign diagnostic | no |
| `S->S` | OR-Bench hard benign | about 500 | Hard-benign diagnostic | no |
| `S->S` | GSM8K/Alpaca-style benign | about 200 | Easy benign anchor | no |
| `S->S` paraphrased | 150-200 sampled safe-prompt rows | 150-200 | Provenance diagnostic | no |

详细 review：

- `analysis_reports/fable5_stage1_dataset_composition_review_detailed_260702.md`

重要约束：

- `S->S` rows 不能进入 primary train/test。
- Stage 1 headline 应以 A-prime 为主。
- B-prime 只作为 sensitivity 报告。
- `S->S` false-positive 检查的阈值应在 paired validation data 上冻结，不能在
  `S->S` 上校准。
- `S->S` natural trajectories 应优先由 R1-1.5B 生成。
- 150-200 条 OpenAI-paraphrased `S->S` subset 只作为 provenance diagnostic。

## 当前下一步

已完成的本地 / RunPod 验证工作：

- 新增 `tests/test_stage1_manifest_freeze_export.py`。
- `python3 -m py_compile` 已通过，覆盖新增测试和两个 freeze/export 脚本。
- 临时 synthetic CLI dry-run 已通过：3 个 prompt groups、4 个 pairs、4 条
  export rows，并确认 prompt hash mismatch 会被拒绝。
- manifest freeze/export pytest target 已在 RunPod CPU 节点通过：
  `3 passed`。
- 真实 clean-manifest prompt split freeze 已在 RunPod 完成：
  2556 pairs、1670 prompt groups，train/val/test prompt groups =
  1503/84/83。
- 真实 `reasoning_only` Stage 1 export 已在 RunPod 完成：
  A-prime = 2192 rows / 1096 pairs，B-prime = 2920 rows / 1460 pairs。
- 新增 `scripts/data/run_stage1_text_baselines.py` 和
  `tests/test_stage1_text_baselines.py`。
- text-baseline readiness pytest target 已在 RunPod 通过：`2 passed`。
- 真实 CPU/text baselines 已在 A6000 RunPod 节点完成，并同步回本地：
  - `runs/stage1_text_baselines/A_prime/`
  - `runs/stage1_text_baselines/B_prime/`

CPU/text baseline 主要结果：

| Dataset | Baseline | Test balanced accuracy |
|---|---|---:|
| A-prime | prompt-only TF-IDF | 0.500000 |
| A-prime | length-only | 0.953846 |
| A-prime | word TF-IDF | 1.000000 |
| A-prime | word BoW | 1.000000 |
| A-prime | char TF-IDF | 1.000000 |
| A-prime | first-sentence-removed TF-IDF | 1.000000 |
| B-prime | prompt-only TF-IDF | 0.500000 |
| B-prime | length-only | 0.869565 |
| B-prime | word TF-IDF | 1.000000 |
| B-prime | word BoW | 1.000000 |
| B-prime | char TF-IDF | 1.000000 |
| B-prime | first-sentence-removed TF-IDF | 1.000000 |

在 review 这些 CPU/text baseline 结果之前，不启动 GPU hidden-state
extraction，因为浅层 reasoning-text baselines 已经接近完美。

随后又运行并同步回本地了 CPU-only surface audits：

- `runs/stage1_surface_audit/A_prime/`
- `runs/stage1_surface_audit/B_prime/`
- Summary:
  `analysis_reports/stage1_surface_audit_summary_260702.md`

这些 audit 没有发现可用的 early prefix window。只看 reasoning 前 4 个词，
A-prime 所有 surface baseline 的 test balanced accuracy 已经超过 `0.90`，
B-prime 已经超过 `0.96`。Cross-source transfer 也接近完美。因此当前
A-prime/B-prime as-is 仍然不应启动 GPU hidden-state extraction。

剩余 Stage 1 CPU-side 工作：

1. original-unsafe vs OpenAI-paraphrased provenance classifier 继续跳过，直到
   有经过 review 且 pair-id 对齐的 original unsafe source。
2. 设计 A-double-prime：两边对称处理、pairwise length targeting、格式和风格统一。

## 代码索引

代码级说明见：

- `scripts/data/README_stage1_data_prep.md`
- `plan/stage1_data_prep_guide_zh.md`

当前 primary 数据准备涉及的主要脚本：

- `scripts/data/extract_harmthoughts_rewrite_seeds.py`
- `scripts/data/extract_reasoningshield_rewrite_seeds.py`
- `scripts/data/generate_safe_rewrites_openai.py`
- `scripts/data/validate_safe_rewrite_pairs.py`
- `scripts/data/pilot_unsafe_preserving_paraphrase_openai.py`
- `scripts/data/repair_openai_unsafe_paraphrases.py`
- `scripts/data/stratify_openai_paraphrase_quality.py`
- `scripts/data/audit_openai_control_samples.py`
- `scripts/data/audit_openai_full_ab.py`
- `scripts/data/audit_rewrite_completeness.py`
- `scripts/data/filter_frozen_manifests_by_completeness.py`
- `scripts/data/freeze_stage1_prompt_splits.py`
- `scripts/data/export_safe_rewrite_pairs_for_stage1.py`
- `scripts/data/run_stage1_text_baselines.py`
