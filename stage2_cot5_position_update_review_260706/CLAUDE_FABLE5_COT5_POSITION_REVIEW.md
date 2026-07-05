# Fable Review: Stage2 Cot5 Pause Position Update

Packet: `stage2_cot5_position_update_review_260706` @ `415967d`. All paths below are relative to `cot-safety/` in the packet unless noted. Review only; no files were edited.

## 1. Verdict

**`PASS_WITH_REQUIRED_FIXES`**

The core position semantics are correct and consistently implemented, and the cot5 change has propagated correctly through every currently wired Stage2 to Stage4 path. Nothing invalidates the planned Stage2 KL-transparent cot5 SFT: the 8B chain may launch as-is; the 1.5B chain trains at the right offset too, but should not launch until required fix RF-1 lands, because its resolved-config audit artifact would otherwise record a contradictory position. RF-2 blocks Stage4 eval, not Stage2.

## 2. Blocking Issues

**RF-1 — 1.5B model chain still inherits `pause.cot_offset: 3` from the template.**

- `configs/model/template_deepseek_r1_distill.yaml:14` sets `pause.cot_offset: 3`. The 8B model config overrides it (`configs/model/deepseek_r1_distill_llama_8b.yaml:19` -> `5`), but `configs/model/deepseek_r1_distill_qwen_1_5b.yaml` does not. The 1.5B Stage2 KL config sets only `sft.cot_offset: 5`, so its merged config carries `pause.cot_offset: 3` and `sft.cot_offset: 5` simultaneously.
- Why this matters:
  - Stage2 is safe only by precedence: `scripts/run_stage2_sft.py:170` reads `sft.get("cot_offset", pause.get("cot_offset", 3))` -> 5. But Stage3 uses the opposite precedence: `scripts/run_stage3_intra_pause_probe.py:144` reads `pause.get("cot_offset", ...)` first. Shipped Stage3 cot5 configs set `pause.cot_offset: 5` explicitly, but future cloned configs could silently build probe data at cot3 under cot5-looking names.
  - `run_stage2_sft.py` writes a resolved config artifact; for 1.5B cot5 runs this would record `pause.cot_offset: 3` inside a run named cot5.

**RF-2 — Stage4 legacy eval shell hard-codes the old insertion point.**

- `legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh:297` passes literal `--insert_pause_after_cot_tokens 3`, and `scripts/run_stage4_steering.py` does not set a corresponding env var.
- This is currently fenced because GPRS does not reach that shell and learned_delta is deprecated, but once GPRS generation is wired to the shell it would force cot3 while writing cot5-named outputs. Fix before any Stage4 eval.

## 3. Non-Blocking Risks

1. Several legacy defaults still fall back to cot3/cot4 if explicit configs are dropped:
   - `src/cot_safety/schemas.py:38`
   - `scripts/run_stage2_sft.py:170`
   - `scripts/run_stage3_intra_pause_probe.py:136-144`
   - `src/cot_safety/steering/liveness_kernels.py:523`
   - `scripts/run_model_comparison_eval.py:200-201`
   - `legacy/PauseProbe/scripts/probe/extract_hidden_states.py:469`
   - `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py:30,199-203`
   - `legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py:337`
   - `legacy/PauseProbe/scripts/steering/run_intra_pause_multiseed_generation_judge.sh:51`
   - `scripts/plot_stage3_heatmaps.py:151`
   - `configs/model/deepseek_r1_distill_qwen_32b.yaml:18`
2. Deprecated Stage4 steering configs do not pin insertion point.
3. Base Stage2 config has a stale cot3 output_dir.
4. Stage3 controls `[control_cot_5, control_cot_6]` cover two of the three pause slots; adding `control_cot_7` would cover pause_2 as well.
5. Review packet omitted `legacy/PauseProbe/scripts/data/`, which contains Stage3 insertion/verification code. Include it in future packets.
6. Some tests still use synthetic `control_cot_3/4` names generically; not wrong, but may confuse reviewers.

## 4. Answers To Review Questions

**Q1 — Is `cot_offset: 5` correct for after-cot4/before-cot5? Yes.**

Fable checked three paths and found them consistent:

- Stage2 formatter: `pause_insertion.py` inserts before token index `first_idx + cot_offset`.
- Stage3 token-id data prep uses `insert_idx = content_start + insert_cot_offset`.
- Generation-side forced insertion with `N=5` generates exactly 5 CoT tokens, then appends 3 pauses.
- `tests/test_pause_insertion.py` explicitly asserts the intended string.

**Q2 — Did propagation happen consistently? Yes, with RF-1/RF-2 caveats.**

Fable found Stage2 data prep, KL configs, model-comparison eval, Stage3 extraction/probing, matched controls, and Stage4 GPRS/liveness configs wired to cot5 on active paths.

**Q3 — Are `control_cot_5`/`control_cot_6` the right matched controls? Yes.**

They are computed from a true no-pause matched forward, not aliasing post-pause positions. They cover pause_0/pause_1 absolute slots; adding `control_cot_7` would cover pause_2.

**Q4 — Any active path silently running cot3/cot4 under cot5 names? No active path.**

Latent traps remain: RF-2 hard-coded shell, RF-1 template inheritance plus precedence mismatch, and legacy direct-CLI defaults.

**Q5 — Blockers before Stage2 cot5 KL SFT?**

No blocker invalidates the run. For 1.5B, land RF-1 before launch. RF-2 is a pre-Stage4-eval fix.

## 5. Minimal Patch Recommendations

- P1 required: add `pause.cot_offset: 5` to `configs/model/deepseek_r1_distill_qwen_1_5b.yaml`.
- P2 required before Stage4 eval: make legacy shell insertion point configurable, pass `INSERT_PAUSE_AFTER_COT_TOKENS` from `run_stage4_steering.py`, and similarly update multiseed generation judge.
- P3 cleanup: convert silent cot3/cot4 defaults to fail-loud or update to 5.
- P4 cleanup: rename stale cot3 base output_dir; pin or archive deprecated Stage4 configs.
- P5 process: include `legacy/PauseProbe/scripts/data/` in future review packets and consider `control_cot_7`.

Bottom line: position semantics are right, propagation is real, and the remaining risk is concentrated in one stale template default and one hard-coded legacy shell argument.

