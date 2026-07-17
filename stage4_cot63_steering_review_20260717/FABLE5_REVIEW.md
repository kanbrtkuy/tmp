# Fable5 Review

Date: 2026-07-17

Bundle reviewed:

- Local path: `/Users/baby/Documents/SafeChain/tmp/stage4_cot63_steering_review_20260717`
- GitHub repo: `kanbrtkuy/tmp`
- Initial bundle commit: `5203fec`

## Verdict

可以进入下一轮 diagnostic，但先修一个分析 bug。

Generation / strict A1-replay 这部分看起来可以继续用；但 `analyze` 对 WildGuard 三分类标签的处理现在不可靠，不能直接拿 generic analyzer 的 `unsafe_rate / unsafe_or_deg_unknown` 当 paper 数字。

## Key conclusions

### 1. fixed-relative strict A1-replay 语义基本成立

代码流程是：

1. 加载 A1 target-resolved rows；
2. replay A1 tokens 到 `pause_2`；
3. 对三个 pause target 做 intervention；
4. common-random continuation。

没有看到 A2 outcome / judge label 的 generation-time 泄漏。

### 2. 最大代码风险是 WildGuard label mapping

`code/diagnose_stage4_cot63_target309.py` 里的 generic `label()` 逻辑优先读 `judge_label`。

但 WildGuard normalized 里：

- `judge_label = unsafe_valid / safe_refusal / ambiguous_partial`
- `binary_safety_label = unsafe / safe / partial`

所以 generic analyze 会把这些三分类标签误当 unknown，导致 GPRS summary 里出现离谱的：

- `unsafe_rate = 0`
- `unsafe_or_deg_unknown = 1.0`

下一轮前要改成优先读 `binary_safety_label`，并把 `partial` 计入 `unsafe_or_deg_unknown`。

### 3. score gating 有依据，但不是 oracle

`projection_scores.jsonl` 上 unsafe-vs-safe AUROC 是 `0.719`。

top60% gate 覆盖：

- `81/99` unsafe
- `104/209` safe

所以 gating 能减少一部分 collateral，但不会神奇解决所有问题。

### 4. fixed-relative 明显比 GPRS 好

fixed-relative alpha=0.1：

- baseline unsafe 里 `30/99` unsafe → safe

GPRS alpha=0.1：

- baseline unsafe 里只有 `9/99` unsafe → safe_refusal
- baseline-safe 里还出现大量 ambiguous/unsafe

所以目前不要回到 GPRS。

## Recommended next diagnostics

### Step 0: fix analyzer label mapping

Before the next formal diagnostic, fix `label()` / summary logic:

- prefer `binary_safety_label` over `judge_label` for binary rates;
- map:
  - `binary_safety_label=unsafe` → unsafe
  - `binary_safety_label=safe` → safe
  - `binary_safety_label=partial` → partial / deg_unknown bucket
- do not report generic `unsafe_rate` from three-class `judge_label` as paper metric.

### Step 1: score-gated fixed-relative

Run:

- layer: `14`
- tokens: all three pause tokens
- alpha: `0.05, 0.07, 0.10, 0.13`
- gate top: `40%, 50%, 60%`

### Step 2: adaptive-alpha fixed-relative

Run:

- threshold: top50/top60 score quantile
- `alpha_max`: `0.10, 0.15, 0.20`

### Step 3: two-layer small-norm gated version

Only if Steps 1/2 are promising:

- layers: `{14, 15}`
- same gate as best Step 1/2
- total norm matched;
- per-layer strength divided by `sqrt(2)`;
- do not start with complex multi-layer concat steering.

## Suggested pass criteria

Continue if:

- `unsafe_or_deg_unknown` improves from current about `-3.24 pp` to around `-7 pp`;
- baseline unsafe → safe remains near `25–30%`;
- safe → unsafe collateral clearly drops;
- degeneration does not rise;
- result is not only from removing ReasoningShield.

## What not to do

Do not:

- use WildGuard baseline label as official generation-time gating condition;
- choose alpha per row after seeing A2 / judge;
- treat `compose-existing` as formal result;
- run a huge grid on 309 rows and report the best as the main result without frozen selection;
- cite generic GPRS analyzer fields like `unsafe_rate=0`;
- upload raw diagnostic generation JSONL with prompt/generated text publicly.

## Bottom line

最值得试的是：

1. 修分析标签；
2. score-gated fixed-relative；
3. adaptive-alpha fixed-relative；
4. 如果有效，再试一个保守 `{layer14, layer15}` two-layer small-norm。

不建议现在换位置、回 GPRS、或先做复杂多层 controller。
