# Fable Review Request: Stage3 On-Policy Confirmatory + Stage4 Liveness Kernels

Please review the current full code tree in:

`safechain_stage2_to_stage4_full_code_20260704/cot-safety`

Context:

- The project logic is not to delete/rewrite old Stage3/4 wholesale, but to replace the evidence standard.
- Stage3 must no longer be judged by pause-probe AUROC alone. It must ask whether pause hidden states add signal beyond prompt baselines and true no-pause content controls, then confirm this on on-policy sampled generations with within-prompt labels.
- Stage4 must no longer use legacy learned-delta as the primary route. It should first run a liveness battery to verify that pause positions are live steering ports, then proceed to GPRS/projection steering.

Changes since the last Fable pass:

1. Stage3 on-policy confirmatory endpoint:
   - Added `src/cot_safety/probes/on_policy_stage3.py`.
   - Added `scripts/run_stage3_on_policy_confirmatory.py`.
   - The new endpoint consumes hidden-state NPZ files from on-policy generations where `labels` are per-generation CoT judge labels.
   - It computes within-prompt AUROC using prompt groups, with prompt-only constant baseline fixed at 0.5.
   - It can also evaluate true no-pause content-control positions.
   - `run_stage3_evidence_report.py` can attach this report via `--on_policy_report`.
   - `pipeline.py` now includes `stage3_on_policy_confirmatory`.

2. Stage4 liveness kernels:
   - Added `src/cot_safety/steering/liveness_kernels.py`.
   - `scripts/run_stage4_liveness.py` now executes implemented GPU kernels in non-dry-run mode instead of always stopping at a plan.
   - Implemented `injection_gain`: inject a norm-controlled random direction at pause / matched content / BOS positions and compare next-token KL slopes.
   - Implemented `attention_mass`: compute last-token attention mass to target pause positions.
   - `pause_kv_ablation` and `safe_unsafe_patching` remain explicitly `incomplete`, so the full four-test liveness gate does not open prematurely.

Validation I could run locally:

- `python3 -m py_compile` over the touched Stage3/4 files passed.
- `run_stage3_on_policy_confirmatory.py --dry_run` works for 1.5B and 8B configs.
- `cot_safety.cli pipeline plan` shows the new Stage3 confirmatory step.
- `run_stage4_liveness.py --dry_run` writes the liveness plan.
- A synthetic `liveness_decision` check stays `incomplete` when all four configured tests are required, and only becomes `green` when restricted to the two implemented tests.

Local limitations:

- This Mac environment lacks `numpy`, `pytest`, `torch`, and `transformers`, so fixture tests and real GPU kernels were not executed locally.
- Please treat compile/dry-run checks as weak evidence and review the code for runtime shape/device/model-hook bugs.

Review questions:

1. Does the new Stage3 on-policy endpoint correctly address the prompt-classification critique, or is it still vulnerable to prompt-only leakage?
2. Are the train/test prompt split, within-prompt AUROC, bootstrap CI, and true-content-control comparison conceptually and mechanically correct?
3. Does the Stage3 evidence report attach/represent the confirmatory endpoint honestly?
4. Are the Stage4 `injection_gain` and `attention_mass` kernels valid enough as the first implemented liveness battery pieces?
5. Are the pause/content/BOS masks correct under left padding and forced pause insertion?
6. Does the liveness runner remain fail-closed, especially because `pause_kv_ablation` and `safe_unsafe_patching` are still incomplete?
7. What blockers/high/medium issues must be fixed before this should land in main?

Please write a complete Markdown review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_ONPOLICY_LIVENESS_REVIEW.md`

Do not edit code. Finish with headline verdict: `PASS to land`, `PASS with required follow-ups`, or `NEEDS FIXES`.
