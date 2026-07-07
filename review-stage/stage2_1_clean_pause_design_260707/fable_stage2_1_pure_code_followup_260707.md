# Fable Follow-up: Stage2.1-pure Config Guard

Date: 2026-07-07

Change after first review:
- Added a `PauseKLSFTTrainer` guard that raises when `pause_kl.pause_tokens` length is neither `1` nor `pause_kl.n_pause_tokens` when `n_pause_tokens` is set.
- Kept single-token expansion to repeated pure chains.
- Added `test_init_rejects_pause_chain_length_mismatch`.

Local verification:
- `python3 -m py_compile ...` passed.
- Pure 1.5B and 8B dry-runs passed.
- Temporary pytest-stub runner executed 26 related tests successfully.

## Raw Review

核对完毕，改动符合我上一轮的建议。

**结论：无新增 blocker，verdict 维持 OK。** 可以继续 1.5B data-prep 校验和 25-step smoke。

核对要点（pause_kl_trainer.py:125-146）：
- 长度守卫逻辑正确：`len ∉ {1, n_pause_tokens}` 且 `n_pause_tokens > 0` 时抛 ValueError，单 token 仍正确展开为重复链；`n_pause_tokens` 未设置（=0）时不误伤。
- 空 `pause_tokens` 回退到 `pause_token × max(1, n)`，与旧配置兼容。
- 测试 test_stage2_pause_kl_trainer.py:178 覆盖了 mismatch 路径，match 断言到具体错误信息。

一个非阻塞的小提醒：`n_pause_tokens=0` 但 `pause_tokens` 显式给了多个不同 token 时守卫不触发——这是有意的宽松行为，若想更严可要求显式 n，但当前配置（单 `<|pause|>`×3）不受影响，无需改。
