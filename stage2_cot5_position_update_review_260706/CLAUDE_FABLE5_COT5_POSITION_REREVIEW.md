# Fable Re-Review: Stage2 Cot5 Fixes

Re-review complete. Fable verified everything statically. It reported that its sandbox blocked Python execution, so it traced the config merge by hand through `src/cot_safety/config.py:49-98`; the merge logic is deterministic.

## Verdict: `PASS_TO_RUN_STAGE2`

## RF-1 Fixed

- `cot-safety/configs/model/deepseek_r1_distill_qwen_1_5b.yaml:14-15` now pins `pause.cot_offset: 5`.
- Merge chain verified: experiment config -> `stage2_intra_pause_sft.yaml:2` -> 1.5B model config -> template.
- `deep_merge` is recursive and include-then-override, so the model-level `5` overrides the template's `3`.
- Resolved 1.5B Stage2 cot5 config has `pause.cot_offset=5`, `sft.cot_offset=5`.
- Stage2 `sft`-first precedence and Stage3 `pause`-first precedence now agree.

## RF-2 Fixed

- `scripts/run_stage4_steering.py:153-155` sets `INSERT_PAUSE_AFTER_COT_TOKENS`, `N_INSERT_PAUSES`, and `MODEL_LABEL` from `steering.*`, and passes the env dict to the shell subprocess.
- `stage4_pause_gprs.yaml:46-47` and `stage4_pause_gprs_8b_4xa100.yaml:26-27` pin `insert_pause_after_cot_tokens: 5` and `n_insert_pauses: 3`.
- `legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh:29-31` consumes the env vars and passes them at generation time; the old literal `--insert_pause_after_cot_tokens 3` is gone.
- `legacy/PauseProbe/scripts/steering/run_intra_pause_multiseed_generation_judge.sh:16-18` does the same.

## Packet Completeness Fixed

- `cot-safety/legacy/PauseProbe/scripts/data/` is present with 14 scripts, including `prepare_intra_pause_probe_data.py`.

## Remaining Non-Blocking Notes

1. Both legacy steering shells still default `MODEL` to an old cot3 checkpoint if invoked directly. The wired `run_stage4_steering.py` path is safe because it sets `MODEL` from the cot5 model config. Fable suggests either updating those fallback paths or making `MODEL` required before direct shell use.
2. Stage4 configs do not set `steering.model_label`, so 8B would use a size-agnostic `deepseek_intra_pause_cot5_sft` label. Cosmetic / Stage4-only.
3. The template default `cot_offset: 3` and 32B config `cot_offset: 4` remain known cleanup items.
4. Fable did not re-run tests due sandbox execution limits; it checked file state and found it consistent with the reported passed checks.

## Bottom Line

No blockers remain for Stage2 cot5 KL-transparent SFT on either the 1.5B or 8B chain.

