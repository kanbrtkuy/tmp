Both blocker sites re-confirmed (config lines 29–33; `generate_vllm_batch` lines 247–259). Here is the final Round 2 review.

---

# Fable Round 2 Review — Stage2 `kl_transparent_emit` Packet (2026-07-04, commit 170d2f4)

Coverage note: `git show --stat 170d2f4` shows exactly 14 files changed since round 1. Every changed file was reviewed in full; all untouched files (data prep, validator, `trl_train.py`, config loader, Stage3 runner, eval data prep) retain their round-1 PASS verdicts.

## Q1 — Are prior C1–C8 fixed?

| Item | Verdict | Evidence / notes |
|---|---|---|
| C1 raw Hydra `pause_token` override | **FIXED** | Diff confirms `+trainer.pause_kl.pause_token="$PAUSE_KL_PAUSE_TOKEN"` deleted from `run_4gpu_intra_pause_sft.sh`; trainer default `<\|pause\|>` used. New overrides (`suppression_chunk_size`, `post_step_invariant_check`) are bare ints/bools — lexable. Minor: `PAUSE_KL_PAUSE_TOKEN` env var (line 45) is now dead; harmless. |
| C2 rows-only guard + invariant | **FIXED** | `_assert_rows_only_training` (pause_kl_trainer.py:142–160) raises on `weight_decay != 0.0` and on any non-embedding trainable param. `_RowsOnlyInvariantCallback` (25–82) snapshots both embedding matrices at `on_train_begin`, compares post-step-1 excluding the pause row, raises with sample row indices, frees snapshots. Timing analysis in Q2. |
| C3 per-token `.item()` syncs | **FIXED** | `_pause_stripped_batch` and `_select_kl_pairs` now do one `.detach().cpu().tolist()` per tensor per batch. Remaining device syncs are O(1)/batch (`pause_mask.any()`, `nonzero()`). |
| C4 chunked suppression | **FIXED**, mathematically identical | Analysis in Q3. |
| C5 1.5B eval cadence | **APPLIED BUT BROKE LAUNCH** | `eval_steps: 50` + `patience: 4` landed, but `save_steps: 25` + `load_best_model_at_end: true` retained → `TrainingArguments` ValueError at construction. This is **NEW-B1** (Q6). Round 1's 25/25 config was launchable; this one is not. |
| C6 8B `save_total_limit` | **FIXED** | `save_total_limit: 8`; save 50 / eval 50 is a valid multiple. Patience 2 retained — fine, round 1 only required the 1.5B change. |
| C7 natural eval + emission metrics | **LARGELY FIXED** — one new blocker, one carried gap | Natural conditions (`insert_pause_after_cot_tokens: -1`, `n_insert_pauses: 0`) added for base and SFT in both eval configs; strip/paste correctly gated on `insert >= 0`, so natural is truly natural. `extended_pause_metrics` + `summarize_pause_emission` → `pause_emission_summary.csv` (natural_pause_rate, exact-single-run-of-3 rate, off-target rate, first-pause index). But **NEW-B2**: the vLLM path strips `<\|pause\|>` from output text, so every natural metric would read 0 (Q6). Carried gap: the §4.4 uncapped KL-drift eval is still absent — transparency-preservation claims remain blocked; emission claims OK once NEW-B2 lands. Definitional flag, not a bug: `first_pause_token_index_inside_think` counts the `\n` after `<think>`, so its expected value is ≈ `cot_offset + 1` relative to the training convention. |
| C8 missing packet files | **FIXED** | `template_deepseek_r1_distill.yaml` matches generation-script defaults; `runpod_base_env.sh` present, stage-agnostic. |

## Q2 — Is the rows-only invariant callback correct under HF Trainer/DDP timing?

**Yes.**

- HF `CallbackHandler` passes `model=self.model` — the **unwrapped** model, not the DDP wrapper — so `get_input_embeddings()`/`get_output_embeddings()` resolve without `.module` gymnastics.
- `on_train_begin` fires after model placement and after DDP broadcasts initial params → snapshot is the true starting state on every rank.
- `on_step_end` fires after `optimizer.step()` and after `global_step` increments → the `global_step >= 1` + run-once gate compares exactly post-step-1 weights. With gradient accumulation, `on_step_end` fires only on real optimizer steps — no premature comparison.
- DDP: post-step params are identical across ranks, so all ranks pass or all raise identically — no desync or hang.
- Resume-safe: on resume the check fires once against the resume-point snapshot; rows-only must hold across any interval, so still valid.
- Memory: CPU fp32 snapshot ≈ 2.1 GB transient/rank on 8B, freed after the check. Chunked compare (2048 rows) avoids a giant allocation; pause-row exclusion is correctly indexed within its owning chunk.

Residual (acceptable): drift is only checked after step 1. The masking hooks are stateless and identical every step, so a step-1 pass generalizes; a paranoid end-of-training re-check would need base weights, which the callback deliberately frees.

## Q3 — Is the chunked suppression loss mathematically identical?

**Yes, exactly.** Per chunk it computes `chunk_logits[:, pause_id] − logsumexp(chunk_logits, dim=-1)`, which **is** `log_softmax(x)[pause_id]` by definition; both ops are row-wise, so row-chunking changes nothing. Same fp32 accumulation → agreement to fp noise. T-4 proves this numerically with `suppression_chunk_size = 1`, the maximal-chunking edge case.

Nuance, not a blocker: autograd retains each chunk's fp32 input (logsumexp saves its input), so retained backward memory is still ~N×V×4 B. The win is peak memory (~halved: no full log_softmax output resident alongside the input; ~10 GB → ~5 GB worst-case transient on 1.5B) — fits A6000-48G, and matches the formula round 1 itself proposed.

## Q4 — Do the unit tests cover the minimum before a GPU smoke?

**Yes on content, with two named gaps — but the real gate is execution.**

Mapping to the round-1 battery: T-1 ✔ (mapping `{0:0, 3:1, 4:2}` + padding), T-2 ✔ (pair alignment incl. the generic invariant `input_ids[b, s+1] == teacher_ids[b, t+1]` for every pair), T-3 ✔ (KL ≈ 0 despite +1000 pause logit on teacher — mask works), T-4 ✔ (emit CE + suppression vs. hand computation), T-5 **partial** (weight-decay rejection ✔, callback detection ✔ — but via manual weight mutation, not a real optimizer step through the `mask_embedding_gradients` hooks), T-6 **absent** (teacher-equals-base identity), T-7 ✔ (mixed pause/no-pause batch: finite loss, backward, grads only on embedding matrices).

The T-5/T-6 gaps are exactly what `_RowsOnlyInvariantCallback` verifies on the real model at step 1 — and the callback's detection logic *is* unit-tested. That is an acceptable division of labor.

Hard gate: the README concedes pytest/torch are absent locally, so **this suite has never executed anywhere**. It must pass on the pod before any smoke. A never-run test file is a syntax-checked promise, not a test.

## Q5 — Does the natural/forced eval make self-emission measurable?

**Yes — conditional on NEW-B2.** The design is right: base_natural vs kl_emit_natural vs kl_emit_forced; natural is genuinely natural; the CSV metrics are precisely what an emission claim needs; per-response metrics persist in `generations/*.jsonl` for post-hoc histograms.

But as committed, all six conditions use `generation_backend: vllm`, and `generate_vllm_batch` silently strips `<|pause|>` → natural emission reads **exactly 0 regardless of model behavior**. That's worse than no eval — a false negative disguised as a clean result. After the one-line fix, emission claims are supportable. Transparency-preservation claims additionally require the still-missing uncapped KL-drift eval.

## Q6 — New bugs introduced by the fixes?

**Two.**

**NEW-B1 — 1.5B config un-launchable (launch blocker).**
`configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml:29–33` — `save_steps: 25`, `eval_steps: 50`, `load_best_model_at_end: true`, both strategies `"steps"` → `TrainingArguments.__post_init__` raises "`--load_best_model_at_end` requires the saving steps to be a round multiple of the evaluation steps" (25 % 50 ≠ 0). Introduced by the C5 edit (eval 25→50 without touching save). Fix — pick one:
- (a) `save_steps: 50` (simplest; halves checkpoint density the filename encodes — rename if chosen);
- (b) `eval_steps: 25` (undoes C5's cost goal);
- (c) keep 25/50, set `early_stopping.enabled: false` **and** `load_best_model_at_end: false` — viable since round 1 already said to pick checkpoints from the battery, not `final/`.
Recommend (a) or (c).

**NEW-B2 — vLLM strips `<|pause|>` from generated text (eval/claim blocker).**
`legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py::generate_vllm_batch` (lines 247–259). `SamplingParams` defaults are `skip_special_tokens=True` (removes special added tokens — `<|pause|>` was added with `special_tokens=True`) and `spaces_between_special_tokens=True` (space-joins consecutive specials, which would break `pause_spans` contiguous-run detection even if the first flag were fixed). Fix: `SamplingParams(..., skip_special_tokens=False, spaces_between_special_tokens=False)`. The transformers path already decodes with `skip_special_tokens=False` (line 243), confirming intent. Follow-on cosmetic: with skipping off, the eos token text can appear at response tails — strip it before judge input.

Not-bugs, for the record: dead `PAUSE_KL_PAUSE_TOKEN` env var; `_zero()` full-tensor sum for a scalar (negligible; keeps the DDP graph connected); eval configs point at `.../final` (placeholder — swap to battery-chosen checkpoints); model YAML `sft_checkpoint` still points at old format-only runs (Stage3 repointing gate, carried).

## Q7 — Final go/no-go

| Milestone | Verdict | Gates |
|---|---|---|
| Code review packet | **GO (conditional)** | Land NEW-B1 + NEW-B2 — both are few-line, single-file fixes. Everything else is clean. |
| 1.5B single-GPU smoke | **NO-GO → GO** after: NEW-B1 fixed; test suite passes on pod | NEW-B2 not needed for the smoke itself. |
| 1.5B 4-GPU smoke | **NO-GO** until single-GPU smoke green | Watch: invariant callback passes step 1 on all ranks; no DDP hang on pause-free batches (zero-pair fallback path). |
| 1.5B full 400-step pilot | **NO-GO** until 4-GPU smoke green + NEW-B2 landed (eval follows the pilot) | Loss-parts logs should show the suppression term not dominating. |
| 8B pilot | **NO-GO** until 1.5B battery analyzed | The 8B config itself is now launch-clean (50/50, limit 8) — this is a sequencing gate only. |
| Stage3 handoff | **NO-GO** until: rows-only invariants verified on the actual trained checkpoint; model YAML `sft_checkpoint` repointed from format-only to the chosen KL-emit checkpoint | The interface itself is compatible (normal HF checkpoint/tokenizer; pause offsets unchanged). |

**Bottom line:** C1–C8 are substantively addressed — six cleanly, one (C5) fixed-in-intent-but-broke-launch, one (C7) fixed-but-for-vLLM-detokenization. The two new blockers total a handful of lines across two files. Fix NEW-B1 and NEW-B2, execute the test suite on the pod, then proceed in the round-1 run order.