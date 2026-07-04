Verification complete — the diff since Round 2 touches exactly the two claimed code files (plus docs). Here is the Round 3 review.

---

# Fable Round 3 Review — Stage2 `kl_transparent_emit` Packet (2026-07-04, commit 3f3048d)

Coverage: `git diff 170d2f4..HEAD` confirms only the 1.5B config (2 lines) and `run_model_comparison_generation.py` (+14 lines) changed besides docs. All other files retain their Round 1/2 verdicts. Both fixes were traced end-to-end through the launch chain, not just at the edit site.

## Q1 — Are NEW-B1 and NEW-B2 fixed?

**NEW-B1: FIXED** (Round 2's recommended option (c), implemented correctly).
- Config now has `load_best_model_at_end: false` (line 33) and `early_stopping.enabled: false` (line 37), keeping `save_steps: 25` / `eval_steps: 50`. The HF "save steps must be a round multiple of eval steps" `ValueError` is gated on `load_best_model_at_end=True`, so 25/50 is now legal at `TrainingArguments` construction.
- Plumbing verified: config → `run_stage2_sft.py:317,321` env vars → `run_4gpu_intra_pause_sft.sh:138,141` Hydra overrides → `trl_train.py`. Critically, `add_early_stopping_callback` (trl_train.py:178–181) returns early on `enabled=false`, so `EarlyStoppingCallback.on_train_begin`'s own assert requiring `load_best_model_at_end=True` is never reached.
- Checkpoint math holds: 16 checkpoints (25…400) ≤ `save_total_limit: 20` → the eval battery sees every checkpoint; density matches the `save25` filename.
- Audited side effect — it's a bonus, not a bug: `keep_best_hot` (run_stage2_sft.py:462) now defaults false. Coherent, since with `load_best=false` HF never writes `best_model_checkpoint` and never re-reads a checkpoint at train end. This actually removes a latent hazard in the old config, where the hot-checkpoint watcher's `--remove-hot-after-sync` could have deleted the best checkpoint that `load_best_model_at_end` needed to reload.

**NEW-B2: FIXED** (exactly as prescribed, plus the follow-on).
- `SamplingParams(skip_special_tokens=False, spaces_between_special_tokens=False)` (lines 266–267). The second flag keeps consecutive `<|pause|>` runs contiguous, so `pause_spans` run detection works.
- EOS follow-on landed: `strip_terminal_eos` (lines 137–143, loops over repeated EOS + whitespace) is applied at both places it must be — vLLM forced-prefixes (line 412, *before* pause-strip/paste) and all responses (line 441, before `generated`, `generated_for_judge`, and both metric computations).
- Backend symmetry preserved: the transformers path truncates at the first EOS id (lines 248–251), so the strip is a no-op there; natural-emission metrics are comparable across backends. vLLM stops at EOS, so EOS text is terminal-only — the strip fully covers it.

## Q2 — Did the fixes introduce new blockers?

**No.** Three minor, non-blocking notes:
1. If `llm.get_tokenizer()` fails, `eos_token_text` is None and the EOS strip silently no-ops (judge text would carry an EOS tail). Graceful degradation, near-impossible for these models.
2. `metric_for_best_model` / `greater_is_better` are now dead keys in the 1.5B config — cosmetic.
3. If the pod's vLLM predates `spaces_between_special_tokens` (very old), `SamplingParams` fails loudly with a TypeError, not silently — acceptable.

## Q3 — Acceptable for pod pytest → 1.5B single-GPU smoke?

**Yes.** Both blockers closed, no new ones, dry-runs re-verified per README. The one hard gate is unchanged: `test_stage2_pause_kl_trainer.py` has still never executed anywhere — it must pass on the pod before the smoke.

## Q4 — Claim-blocking but not code-blocking

1. **Uncapped KL-drift eval (§4.4) still absent** → transparency-preservation claims remain blocked (carried from R1/R2). Emission claims are now supportable post-NEW-B2.
2. **Eval configs point at `.../final` placeholders** — and with `load_best=false`, `final/` is now the *last-step* model, not best. Repointing to battery-chosen checkpoints is now mandatory, not just recommended (slightly elevated from Round 2).
3. **Model YAML `sft_checkpoint` still points at the old format-only run** → Stage3 repointing gate, carried.
4. **Rows-only invariant must be confirmed on the actual trained checkpoint** (callback firing green at step 1 on all ranks).
5. Definitional flag carried: `first_pause_token_index_inside_think` ≈ `cot_offset + 1` due to counting the `\n` after `<think>` — account for it at analysis time.

## Final go/no-go

| Milestone | Verdict | Gates |
|---|---|---|
| Code review packet | **GO** (unconditional) | NEW-B1 and NEW-B2 verified landed; no new blockers. |
| Pod pytest run | **GO — do this first** | Suite has never executed; it is the sole gate before smoke. |
| 1.5B single-GPU smoke | **GO** once pytest green | Config is launch-clean. |
| 1.5B 4-GPU smoke | NO-GO until single-GPU green | Watch: invariant callback on all ranks; no DDP hang on pause-free batches. |
| 1.5B full 400-step pilot + eval battery | NO-GO until 4-GPU smoke green | NEW-B2 already landed, so the battery is unblocked; watch suppression term in loss-parts logs. |
| 8B pilot | NO-GO until 1.5B battery analyzed | Sequencing gate only; 8B config launch-clean. |
| Stage3 handoff | NO-GO | Repoint `sft_checkpoint` to a battery-chosen checkpoint (now mandatory — `final/` = last step, not best); rows-only invariant verified on trained weights. |

**Bottom line:** Both Round 2 blockers are genuinely closed — NEW-B1 via the exact option (c) semantics (and it incidentally fixed a latent hot-sync/best-checkpoint hazard), NEW-B2 via the exact two flags plus correctly-placed EOS stripping at both consumption sites. Nothing new was broken. Proceed: pod pytest → 1.5B single-GPU smoke.