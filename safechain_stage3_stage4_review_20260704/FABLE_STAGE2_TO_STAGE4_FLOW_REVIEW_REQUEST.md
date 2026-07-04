# SafeChain Stage2-to-Stage4 Flow Review Request

Date: 2026-07-04.

Please review the end-to-end plan and initial implementation direction for
SafeChain Stage2 -> Stage3 -> Stage4 under the new Stage2 method.

Write the complete Markdown review to:

`safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_TO_STAGE4_FLOW_REVIEW.md`

Do not edit code.

## Critical current status

The new Stage2 `kl_transparent_emit` SFT has **not been run yet**. We are still
before the first training run of the new Stage2 method.

The user's intended sequencing is:

1. First run Stage2 with the current `kl_transparent_emit` method.
2. After the Stage2 checkpoint exists, run a liveness battery to test whether
   pause tokens are live steering ports.
3. If liveness is green, do not change Stage2; proceed with fixed Stage3 and
   GPRS-style Stage4.
4. If liveness is yellow/red, then add Stage2.5 as a conditional fallback and
   retrain.

Please verify whether this sequencing is correct.

## Project logic

SafeChain has four stages:

1. Stage1: establish hidden-state separability of safe vs unsafe CoT.
2. Stage2: train a pause-token model that emits pause tokens without changing
   model behavior as much as possible.
3. Stage3: test whether pause positions carry safety-relevant separability
   beyond prompt artifacts.
4. Stage4: intervene at pause-token hidden states to reduce unsafe CoT without
   increasing over-refusal, damaging capability, or producing broken output.

## New Stage2 method

The new Stage2 candidate is `kl_transparent_emit`, not ordinary full SFT:

- pause-slot CE encourages pause emission;
- pause-stripped continuation KL encourages the continuation with pauses to
  match the continuation without pauses;
- pause suppression discourages extra pause emission;
- rows-only training should update only the pause embedding row;
- Stage2.5 should be a conditional fallback, not the default.

Relevant files in this review packet:

- `configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
- `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `scripts/run_stage2_sft.py`
- `tests/test_stage2_pause_kl_trainer.py`

## Existing Fable reviews to use

Please read these before answering:

- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_FULL_FLOW_REVIEW.md`
  if present in context; otherwise use
  `safechain_stage3_stage4_review_20260704/stage2_context/CLAUDE_FABLE5_STAGE2_FULL_FLOW_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_FOLLOWUP_REVIEW.md`

The follow-up review introduced the key conclusion:

- liveness must be measured first;
- green means no Stage2.5;
- yellow/red may require Stage2.5;
- Stage2.5 should prefer label-free near-pause KL exemption and/or injection
  gain hinge, with contrastive/probe-margin only in reserve.

## Planned code direction for Stage3/4

We want an initial Stage3/4 framework, not the full GPU experiment in one step.
The intended framework:

### Stage3 framework

- Add prompt baselines to Stage3 configs: `last_prompt_token`, `pre_think`,
  and eventually a prompt-only text classifier baseline.
- Fix or at least mark `control_cot_3/4` as invalid aliases; true controls
  should be pause-free matched content tokens or matched-depth content tokens
  away from the pause.
- Add config fields for on-policy relabeling and within-prompt AUROC as the
  primary endpoint after 10 samples/prompt are generated and CoTs are judged.
- Keep current teacher-forced extraction as a compatibility path, but do not
  claim pause-specific signal from it alone.

### Stage4 framework

- Add an explicit liveness stage before steering.
- Add config support for liveness battery controls:
  base model with pasted pauses, old full-SFT checkpoint as positive control,
  new KL checkpoint as test.
- Add config support for GPRS / gated projection steering:
  direction artifact, safe centroid, probe checkpoint, gate threshold, norm cap,
  and steering layer chosen from live layers.
- Keep legacy `learned_delta` available only as a baseline/control, not the
  primary method.
- Add forced/natural/hybrid pause modes and cot3/cot4 insertion offset plumbing.
- Add eval intent for CoT-vs-answer judging, over-refusal, capability, broken
  output, unlabeled rate, termination, and length shift.

## Questions for Fable

Please answer directly and operationally:

1. Is the user's current sequencing correct: run Stage2 `kl_transparent_emit`
   first, then liveness, then Stage2.5 only if needed?
2. Given Stage2 has not run yet, what is the minimal code framework we should
   add now to Stage3 and Stage4?
3. Which Stage3 code/config changes are mandatory before the first post-Stage2
   probe run?
4. Which Stage4 code/config changes are mandatory before any steering pilot?
5. Should we implement Stage2.5 code now but keep it disabled, or wait until
   liveness fails?
6. What exact kill criteria should decide:
   - proceed to Stage3;
   - proceed to Stage4 GPRS;
   - branch to Stage2.5;
   - stop and write a negative result?
7. What would you put in the initial code PR/commit, and what should be
   explicitly left for after the Stage2 checkpoint exists?

Please be blunt. The user wants a practical implementation plan that avoids
burning GPU on invalid Stage3/Stage4 runs.
