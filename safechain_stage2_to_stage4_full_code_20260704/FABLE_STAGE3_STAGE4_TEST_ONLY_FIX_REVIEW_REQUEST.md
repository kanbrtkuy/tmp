# Fable Review Request: Stage3/4 Test-Only Final Fix

Please review the current full tree under:

`safechain_stage2_to_stage4_full_code_20260704/cot-safety`

This pass responds to your previous review:

`CLAUDE_FABLE5_STAGE3_STAGE4_NEW_HIGH_FIX_REVIEW.md`

You found one remaining HIGH/pre-land item:

- `test_liveness_gate_status_reads_report_and_fails_closed` had a fixture with green metrics but still asserted `allow_yellow=False` should make it not ready.

Fix:

- `tests/test_stage4_gprs_liveness.py` now keeps the green metrics fixture only for the ready case.
- It then writes a sub-threshold red fixture:
  - `pause_vs_content_gain: 0.10`
  - threshold floor remains `0.25`
- It asserts this report is not ready.
- The model-mismatch fail-closed assertion remains.

Local validation:

- `python3 -m py_compile tests/test_stage4_gprs_liveness.py src/cot_safety/steering/liveness.py` passed.
- Because local pytest is unavailable, I directly executed the same fixture flow against `liveness_gate_status`; it passed:
  - missing report => `decision == missing`
  - green metrics + green positive control => ready
  - sub-threshold metrics => not ready
  - wrong model => not ready

Please answer:

1. Is the remaining test-only HIGH closed?
2. Is this now safe to land into main as fail-closed Stage3/4 scaffolding?
3. Are there any remaining blocker/high/medium issues that must be fixed before landing?

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_TEST_ONLY_FIX_REVIEW.md`

Do not edit code.
