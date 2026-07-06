# Fable Review: Stage2 Pilot To Stage3 Preflight

Date: 2026-07-06
Reviewer: Claude/Fable-5 via `claude -p --model claude-fable-5 --effort max`
Brief: `stage2_pilot_stage3_preflight_brief_260706.md`

## Verdict

**CONDITIONAL PASS**: proceed to a teacher-forced, forced-aligned Stage3 screen pilot now.

Fable's key distinction:

- The Stage2 pilot is behavior-preserving enough to probe.
- The teacher-forced Stage3 screen chain is runnable after fixing the checkpoint/runtime config.
- The confirmatory on-policy endpoint is not yet runnable because the production chain is missing: multi-sample generation, CoT-segment judging, and generation-to-hidden-NPZ conversion.
- Stage4 should remain paused until Stage3 signal and liveness gates pass.

## Required Before Stage3 Screen

1. Use the 2xA6000 Stage2 checkpoint:
   `/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`
2. Run the Stage3 screen as forced-aligned, teacher-forced.
3. Treat the primary gate as pause-only margin, not post-pause or combined pause/post-pause headline:
   `pause_only_margin > 0.01` over the best of prompt baselines and true no-pause content controls.
4. Keep natural pause rows diagnostic-only for now.

## Stage3 Interpretation Rules

- Primary: forced-aligned pause positions `pause_0,pause_1,pause_2`.
- Required baselines: `last_prompt_token`, `pre_think`, and true matched no-pause `control_cot_5,control_cot_6`.
- Do not claim causal/on-policy trajectory monitoring from teacher-forced labels.
- Do not claim Stage4 readiness from this screen alone.

## Blocking Before Confirmatory Stage3 Claim

Fable says the analysis side exists:

- `src/cot_safety/probes/on_policy_stage3.py`
- `scripts/run_stage3_on_policy_confirmatory.py`

But the producer side is missing:

1. multi-sample on-policy generation honoring `samples_per_prompt: 10`
2. implemented `cot_segment_judge`
3. converter from judged generations into extractor-format rows plus matched no-pause controls

## Blocking Before Stage4

Stage4 remains paused. Before steering:

1. Stage3 screen must pass.
2. On-policy confirmatory Stage3 should pass.
3. Liveness battery must pass.
4. Two liveness kernels still need implementation before using the gate as final evidence:
   `pause_kv_ablation` and `safe_unsafe_patching`.

## Recommended Run Order

1. Add/fix 2xA6000 Stage3 config and stale control flag.
2. Run teacher-forced forced-aligned Stage3 screen.
3. Build on-policy producer chain in parallel.
4. If screen passes, run confirmatory on-policy Stage3.
5. If confirmatory passes, finish liveness battery.
6. Only after those gates consider full Stage2 or 8B port.

## Executor Notes

After this review, a new config was added:

`configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5_2xa6000.yaml`

It points to the 2xA6000 Stage2 final checkpoint, isolates Stage3 outputs under `_2xa6000` paths, and sets hidden extraction to two workers/four train shards.
