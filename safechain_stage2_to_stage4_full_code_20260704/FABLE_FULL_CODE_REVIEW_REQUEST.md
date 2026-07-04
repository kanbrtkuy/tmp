# Request: Full Code-Tree Review of SafeChain Stage2 -> Stage4

Please review this as a complete codebase snapshot, not as a diff. The previous diff review was useful, but the user specifically wants a full-code review because cross-file assumptions, hidden aliases, config mismatches, and evaluation gaps can be invisible in a patch-only view.

## Files to Read

Start with:

- `README.md`
- `cot-safety/README.md`
- `cot-safety/docs/migration_from_pauseprobe_cotpausetoken.md`
- `review_context/safechain_stage3_stage4_review_20260704/README.md`
- `review_context/safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_TO_STAGE4_FLOW_REVIEW.md`
- `review_context/safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_TO_STAGE4_CODE_REVIEW.md`
- `review_context/safechain_stage2_review_20260704/PROFESSOR_CRITIQUE.md`
- `review_context/safechain_stage2_review_20260704/REFERENCE_ARXIV_2310_02226.md`

Then inspect the full code tree under `cot-safety/`, especially:

- `cot-safety/scripts/run_stage2_sft.py`
- `cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `cot-safety/legacy/COTPauseToken/scripts/data_generation/pause_sft/`
- `cot-safety/legacy/COTPauseToken/scripts/training/`
- `cot-safety/scripts/run_stage3_intra_pause_probe.py`
- `cot-safety/legacy/PauseProbe/scripts/probe/`
- `cot-safety/scripts/run_stage4_liveness.py`
- `cot-safety/scripts/run_stage4_steering.py`
- `cot-safety/src/cot_safety/steering/`
- `cot-safety/configs/experiment/stage2_*`
- `cot-safety/configs/experiment/stage3_*`
- `cot-safety/configs/experiment/stage4_*`
- `cot-safety/tests/`
- `cot-safety/pipelines/`
- `cot-safety/res/`

## Project Plan Being Reviewed

The original four-stage plan:

1. Stage1: verify latent separability on original R1 models using SafeChain data. Generate Normal and MoreThink CoTs, label a trajectory unsafe if any CoT position is unsafe by Llama-Guard, extract early hidden states, and train probes.
2. Stage2: train a pause-token model. Insert K pause tokens in each response, using full SFT initially. Target: model emits pause tokens before/in CoT while preserving safety and capability.
3. Stage3: verify separability at pause positions. Use the pause model on the same data distribution, extract hidden states at pause positions, and compare pause vs no-pause/control positions.
4. Stage4: use the pause hidden states as a steering port to reduce unsafe CoT while preserving capability, avoiding over-refusal and broken output.

The current revised Stage2 is no longer ordinary full SFT. It is intended to be `kl_transparent_emit`: teach pause emission while constraining the continuation to match the no-pause baseline as much as possible.

## Key Scientific Concerns to Audit

Please directly evaluate these concerns against the full code:

1. Does Stage2 actually implement KL-transparent pause emission, or are there leaks that still make it ordinary SFT in practice?
2. Does Stage2 preserve the existing Stage1 data format and avoid breaking Stage3 assumptions?
3. Does the training objective risk making the pause token harmless but inert? If yes, what code-level diagnostics or modifications should be added before GPU runs?
4. Does Stage3 truly test trajectory signal beyond prompt risk, or can it still collapse into prompt classification?
5. Are `cot3`, `cot4`, post-pause, prompt-baseline, and content-control positions implemented in a way that supports the intended comparison?
6. Are Stage3 labels on-policy per generation, or teacher-forced/reference labels? Where exactly does the code enforce or fail to enforce this?
7. Is Stage4 liveness sufficiently implemented before steering, or are there placeholder paths that could accidentally be treated as evidence?
8. Is GPRS / gated projection steering correctly scoped to pause hidden states, with norm caps and no accidental broad-layer intervention?
9. Are eval endpoints separated cleanly: unsafe CoT rate vs final answer unsafe rate, over-refusal, capability, broken output, length shift, judge failures, and think-end rate?
10. Are config defaults, paths, checkpoint selection, and output paths consistent enough to launch Stage2 1.5B safely?

## What We Need From You

Please write a rigorous review to:

1. Give a top-level verdict:
   - `GO for Stage2 1.5B`
   - `GO only after small code fixes`
   - `NO-GO; serious design/code blocker`
2. List concrete findings with severity: blocker, high, medium, low.
3. For each finding, cite exact files/functions/config keys and explain the failure mode.
4. Separate:
   - implementation bugs,
   - scientific-design risks,
   - missing diagnostics,
   - documentation/config clarity issues.
5. Give the minimal code changes needed before running Stage2.
6. Give the minimal Stage3/Stage4 framework changes needed before trusting those stages.
7. Specifically answer: under the current code, can the trained pause token still be used for Stage4 steering, or must liveness/probe evidence be required first?
8. Suggest stronger alternatives if the current KL-transparent pause emission approach is insufficient.

Please be skeptical and concrete. If something is incomplete but intentionally scaffolded, say whether it is safe as scaffolding or dangerous because it can be misread as a completed experiment.

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_FULL_CODE_TREE_REVIEW.md`

Do not edit code. The review should be self-contained and readable by the project owner.
