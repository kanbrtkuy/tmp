# Fable Review: Stage2.1-pure Code

Date: 2026-07-07

Scope:
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `scripts/run_stage2_sft.py`
- `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
- `configs/experiment/stage21_pause_pure_dagger_*.yaml`
- `configs/data/stage21_pure_dagger_pool.yaml`
- Stage2.1-pure tests for trainer, natural metrics, and on-policy mining

Local verification before review:
- `python3 -m py_compile ...` passed.
- Local Python environment has no `pytest`; a temporary pytest-stub runner executed the relevant test functions.
- 25 related test functions passed before the non-blocking config guard.
- Pure 1.5B / 8B dry-runs showed:
  - `PAUSE_KL_PAUSE_TOKENS=["<|pause|>","<|pause|>","<|pause|>"]`
  - `PAUSE_KL_N_PAUSE_TOKENS=3`
  - `FORMAT_ONLY_TRAINABLE_TOKENS=["<|pause|>"]`
  - `PAUSE_KL_STOP_WEIGHT=2.0`
  - `PAUSE_KL_SUPPRESSION_LOSS_TYPE=margin`

## Raw Review

Review 完成。核心 trainer 逻辑、init、stop mask、emit/suppression、KL 对齐路径都读过了。结论如下。

## Blockers

**没有发现 blocker。** 重点检查的几个风险点均已正确处理：

1. **pure repeated 不会被当成 indexed**：init 中 `pause_chain_token_ids=(id,id,id)` 与 `pause_token_ids=dict.fromkeys(...)→(id,)` 分离（pause_kl_trainer.py:147-148）。所有按"身份"操作的路径（suppression 的 `torch.isin`、KL 的 pause strip、`_select_kl_pairs` 的 `pause_token_id_set`）都用去重集合，语义正确；按"序列"操作的 stop mask 用 chain ids。
2. **stop loss 不会在第 1/2 个 pause 后误触发**：旧的 `len(pause_token_ids)<=1` 快捷路径（对每个 pause 后都触发 stop——这正是旧 bug）已删除，现在统一走 width=3 滑窗，只有连续 3 个 pause 结尾的位置才置 True。新测试 `test_repeated_single_pause_chain_masks_stop_only_after_third_pause` 三个 case（exact-3 触发、2 个不触发、拆开不触发）直接覆盖了这一点。窗口 end_pos 对应预测"第 3 个 pause 之后那个 token"的位置，与 shift 对齐正确。
3. **emit 与 stop 无冲突**：emit CE 作用在 target 是 pause 的位置（pause1→2、pause2→3），stop margin 作用在 chain 完整后 target 是非 pause 的位置，两者位置不重叠（前提是 gold 恰好 3 个 pause，DAgger relabel 保证了这一点）。
4. **符合 Goyal 精神**：token 完全同质（单一 `<|pause|>` 词表项、单一 embedding row），"数到 3"不靠 token 身份而靠 frozen backbone 对不同 pause 深度产生的不同 hidden state——这是 pure repeated 的正统做法。config 里 `format_only.trainable_tokens=["<|pause|>"]` 与 rows-only invariant check 一致。

## Non-blocking suggestions

1. **n_pause_tokens 与 pause_tokens 长度不一致时静默**：如果 `pause_tokens` 给了 3 项但 `n_pause_tokens=4`，init 直接用 3 无警告。建议加一个 `len mismatch` 的 assert/log，防未来 config 手误。
2. **pause head 维度语义变化**：`Linear(bottleneck, len(self.pause_token_ids))` 现在输出维度为 1（dedup 后）。config 里 pause_head 是 disabled，无影响；但若以后启用需确认下游索引也用 dedup 集合。
3. **gold 中 >3 连续 pause 的病态样本**：若数据里出现 4 连 pause，滑窗会在 target=第 4 个 pause 的位置触发 stop，与 emit 直接对抗。mining relabel 已保证 exact-3，但建议 data-prep validation 里加一条 assert（max run == 3）作为防线。
4. **rows-only 容量是本方法真正的科学风险**：stop/emit 都只能通过单个 pause 的 input/output embedding row 传梯度，"stop after 3" 依赖 frozen hidden state 在 pause 深度上可区分。这不是代码 bug，而正是 25-step smoke 要回答的问题——建议 smoke 时顺带 log 一下 "after-pause-1/2/3 位置的 pause logit margin" 三条曲线，能最快判断可行性。
5. `stop_weight=2.0` 与 suppression 在 stop 位置有轻微双重计入（该位置同时属于 non_pause_mask），是有意加权的话没问题，写进 run notes 即可。

## Verdict

**可以跑 1.5B 25-step smoke 和 data-prep validation。** 代码路径正确、测试覆盖到位（包括 exact-3 stop mask、dedup init、pure repeated relabel、natural metrics），dry-run 环境变量与设计一致。主要不确定性不在实现，而在 rows-only 下 pure repeated 的可学习性——smoke run 正是验证这个的正确下一步。
