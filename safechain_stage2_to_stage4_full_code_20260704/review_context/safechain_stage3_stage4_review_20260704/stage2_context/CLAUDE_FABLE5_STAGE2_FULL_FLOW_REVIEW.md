# SafeChain Stage2 Full-Flow Review — `kl_transparent_emit` packet (2026-07-04)

Scope reviewed: every file listed in `README.md` plus the config inheritance chains, the legacy Hydra configs, the eval generation/summarize scripts, and the Stage3 runner. No files were edited. Verdict first, then per-stage analysis, the loss/indexing proof, and the answers to all 10 questions.

---

## 0. Executive Verdict

**The method implementation is correct. The launch plumbing is not proven, and the eval cannot yet measure the thing this method exists to produce.**

- `PauseKLSFTTrainer` implements the stated objective (pause-slot CE + pause-stripped continuation KL + non-target suppression) **correctly**. I attempted to construct indexing counterexamples and failed; a positive proof is in §3.
- The same-model-as-teacher trick is **valid under this exact configuration** (rows-only training + pause-logit masking + weight_decay 0), but the runtime assertion that guards it is necessary, not sufficient. It must be backed by a post-step invariant test.
- **One high-confidence launch blocker**: the new Hydra CLI override `+trainer.pause_kl.pause_token=<|pause|>` in `run_4gpu_intra_pause_sft.sh` almost certainly fails Hydra's override lexer. It is the *only* new CLI pattern in this packet that has never been exercised — the dry runs stop before Hydra is invoked.
- **The current model-comparison eval force-inserts pauses and strips natural ones.** As shipped, it is structurally incapable of confirming or refuting self-emission. Training may proceed before this is fixed; **no self-emission claim may be made** until it is.
- Two performance issues in the trainer (per-token `.item()` sync storm, full-vocab fp32 `log_softmax` in suppression) are not correctness bugs but will burn real GPU-hours; fix before the 8B run at minimum.
- Zero tests exist. Given the trainer's indexing subtlety, that is not acceptable for GPU spend. The minimum battery is small (§6) — roughly half a day of work.

**Overall: GO for the code-review packet as an experimental branch (with the fixes below). NO-GO for any GPU job until B1–B3 are closed. NO-GO on self-emission claims until the eval gap (E1) is closed.**

---

## 1. Blockers / Must-Fix Before GPU

### B1 — Hydra will likely reject the unquoted `<|pause|>` override (launch failure)

`cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`, PAUSE_KL block:

```bash
"+trainer.pause_kl.pause_token=$PAUSE_KL_PAUSE_TOKEN"
```

Bash strips the quotes; Hydra's override parser receives the bare argument `+trainer.pause_kl.pause_token=<|pause|>`. Hydra 1.3's override grammar only permits a restricted character set in *unquoted* values — `<`, `>`, and `|` are not in it. Expected failure mode: `LexerNoViableAltException` / `mismatched input '<'` at launch, i.e. the job dies on the pod after you've paid for it.

Why this one slipped through every prior run:

- The proven format-only launches pass the token *inside JSON string quotes* that survive to Hydra: `'...special_tokens_to_add=["<|pause|>"]'` and `FORMAT_ONLY_TRAINABLE_TOKENS='["<|pause|>"]'`. Hydra parses those as quoted strings. This is the first time the raw token is passed as a bare scalar override.
- The README's "Local Verification Already Run" (`py_compile`, `bash -n`, two `--dry_run --skip_data_prep` runs) never reaches Hydra parsing. `bash -n` checks bash syntax only; `--dry_run` in `run_stage2_sft.py` prints the command without executing it.

**Fix (pick one):**

1. **Preferred — delete the override.** `PauseKLSFTTrainer`'s default `pause_token` is already `<|pause|>`; the CLI override is redundant. Remove the line from the `HYDRA_ARGS` block.
2. Or embed quotes that survive to Hydra: `"+trainer.pause_kl.pause_token='$PAUSE_KL_PAUSE_TOKEN'"`.

**Mandatory verification either way** — 1 minute on the pod, before any torchrun:

```bash
python -c "from hydra.core.override_parser.overrides_parser import OverridesParser; \
p = OverridesParser.create(); \
p.parse_overrides(['+trainer.pause_kl.pause_token=<|pause|>']); print('bare OK')" ; \
python -c "from hydra.core.override_parser.overrides_parser import OverridesParser; \
p = OverridesParser.create(); \
p.parse_overrides([\"+trainer.pause_kl.pause_token='<|pause|>'\"]); print('quoted OK')"
```

I could not verify this empirically here (no hydra-core in the review environment; install requires approval I didn't have), so I am flagging it as high-confidence rather than proven. The one-liner settles it definitively.

### B2 — No unit tests for a trainer whose entire value is index arithmetic

`pause_kl_trainer.py` does three nontrivial index transformations (§3). Any off-by-one silently trains the wrong objective and you find out $1000 later via a confusing eval. The minimum test set (§6.1) is mandatory before the first GPU job. Special mention: `F.kl_div` with an exactly-zero target probability (the masked pause slot) relies on PyTorch's `xlogy(0, ·) = 0` convention — almost certainly fine on any recent torch, but it's a one-assert test and NaN here poisons the whole run. I could not check locally (no torch in this environment).

### B3 — `_assert_rows_only_training` cannot see what it claims to protect

The assertion (in `pause_kl_trainer.py`) checks that the only `requires_grad=True` parameters are the input/output embedding **matrices**. The teacher-equals-base guarantee needs more: (a) the gradient-mask hooks from `trl_train.py:mask_embedding_gradients` are actually registered and zero every non-pause **row**, and (b) `weight_decay == 0.0` (decoupled AdamW decays *all* `requires_grad` params, gradient or not — a nonzero weight decay would silently shrink every frozen embedding row every step, and the teacher would drift from base *during* training, invalidating the KL target). Currently weight_decay is the TrainingArguments default 0.0 and nothing overrides it — correct, but unguarded.

**Fix:** extend `_assert_rows_only_training` (or add a `TrainerCallback.on_step_end` check for the first step) to (1) assert `self.args.weight_decay == 0.0`, (2) snapshot non-pause embedding rows before step 1 and assert bit-identity after it. ~20 lines. This converts the core soundness assumption from "believed" to "checked".

### Strongly recommended before GPU (not launch-blocking)

- **P1 — `.item()` sync storm.** `_pause_stripped_batch` and `_select_kl_pairs` call `int(tensor[i].item())` per token → ~6–8k GPU synchronizations per micro-batch (×8 GA micro-steps on the 8B config). Fix: `rows = input_ids.cpu().tolist()` / `labels.cpu().tolist()` once per batch, then pure-Python loops. Same semantics, one sync.
- **P2 — suppression memory spike.** `_pause_losses` computes `log_softmax(shift_logits[non_pause_mask].float())` over the *full vocab* for what can be nearly every completion token: ~2×4096×151936×4B ≈ 5 GB transient on the 1.5B config (on top of student+teacher logits). Fits on A6000-48G but it's the tallest spike in the step. Fix: compute only the pause column, `pause_logprob = logits[..., pause_id].float() - torch.logsumexp(logits.float(), dim=-1)`, chunked over rows (e.g. 1024 positions per chunk).
- **P3 — eval cost/cadence.** `compute_loss` runs the teacher forward during eval too (unavoidable, the eval loss is composite), so each eval pass is ~2× a train pass over 500 rows — while `eval_steps: 25` on the 1.5B config means you train only ~400 examples between evals. Eval will dominate wall time. Set `eval_steps: 50` or subsample eval to ~128 rows. Also `_maybe_log_loss_parts` gates on `global_step % logging_steps`, and `global_step` is frozen during eval — eval part-logging is all-or-nothing per eval pass depending on where the step landed. Cosmetic; fix opportunistically.
- **P4 — packet completeness.** `configs/model/template_deepseek_r1_distill.yaml` is referenced by both model configs via `defaults:` but **absent from the packet** (confirmed by glob) — `load_config` on the packet copy would `FileNotFoundError`. `runpod_base_env.sh` is sourced by `run_4xa100_model_comparison_eval.sh` and also absent. Your dry runs succeeded from the real repo, so this is packet hygiene, not a code bug — but it means this review could not resolve the template's contents (chat template / model kwargs are assumed sane from prior runs).

---

## 2. Stage-by-Stage Correctness Review

### 2.1 Data prep (Q1 part, Q2) — **PASS, unchanged, one fragility caveat**

`build_intra_think_pause_sft_splits.py` / `validate_intra_think_pause_sft_format.py`:

- `kl_transparent_emit` introduces **zero changes** to data prep. The runner builds the same triplet (`intra_pause_cot{3,4}`, `no_pause_matched`, `pre_think_pause3_matched`) with the same shared split (seed 260615), same JSON schema, same manifests. **Q2 answer: yes, format fully preserved; no Stage1/Stage3 breakage.** The KL/stripping logic lives entirely inside the trainer at batch time — the on-disk data with embedded `<|pause|><|pause|><|pause|>` text is byte-identical to what format-only training consumed.
- Insertion convention: pause text inserted at the char offset of token `first_nonspace_token_index + cot_offset` via `offset_mapping` — this mirrors the PauseProbe hidden-extraction convention exactly, which is what makes Stage3's `cot_3`/`cot_4`/`pause_k` positions line up. Verified consistent with `stage3_intra_pause_probe.yaml` (`cot_offsets: [3, 4, 7, 8]`, positions `pause_0..2`).
- The validator independently re-derives the expected offset from pause-stripped reasoning and fails hard (exit 1) on mismatch, across all three variants. Good.
- **Caveat:** `split_triplets` raises if accepted rows < 18000, and raw is exactly 18000 — the pipeline tolerates *zero* rejections. It has passed before (variants exist for cot3 and cot4), but any future raw-data change or a row whose first think segment has < cot_offset+1 tokens breaks prep. Not a blocker; know that this failure mode exists.

### 2.2 Runner / config plumbing (Q1) — **PASS**

`run_stage2_sft.py`:

- `sft.method ∈ {kl_transparent, kl_transparent_emit, pause_kl}` → sets `PAUSE_KL_ENABLED=true` **and force-enables `format_only`** — the right coupling, since the teacher-equals-base argument (§3.4) collapses without rows-only training. This coupling is the single most important line of plumbing in the packet and it is correct.
- Path defaults are self-consistent: 1.5B config resolves prepared root to `{data_root}/pause_sft/stage2_trusted_cot_18k_intra_cot3`; the 8B config explicitly pins `..._intra_cot4`. Both match what the builder writes.
- Config inheritance verified end-to-end for both KL configs (custom shallow-include loader in `cot_safety/config.py`, `defaults:` merged in order then file overrides; `${VAR:-default}` env expansion works as used). 1.5B chain: kl_emit → intra_pause_sft base → 1.5B model + 18k data + a6000_4x runtime. 8B chain: kl_emit_8b → format_only_8b_cot4 → sft_8b_4xa100 → base + 8B model + a100_4x (this is where `paged_adamw_8bit`, batch 1, GA 8 come from). Coherent.
- `build_trainer_config` pops `early_stopping`/`format_only` from trainer kwargs but deliberately forwards `pause_kl` to the trainer ctor — matches `PauseKLSFTTrainer.__init__`. Correct.
- Hot-checkpoint sync wiring and `--skip_*` flags: consistent with prior proven runs.
- **Environment assumption (flag, not bug):** `trl_train.py` uses the old-style SFTTrainer surface (`tokenizer=`, `max_seq_length=`, `formatting_func=`, `dataset_batch_size=`) and configs use `evaluation_strategy`. Newer TRL (≥0.13) / transformers renamed or removed these. Prior successful Stage2 runs prove the pinned pod image works — **run on that same image**. Any "let's upgrade the env while we're at it" torpedoes the launch.

### 2.3 Trainer (Q3) — **PASS on correctness** — proof in §3; perf issues P1/P2 above.

### 2.4 Shell / Hydra integration (Q1 part) — **FAIL until B1 is fixed.**

Everything else in `run_4gpu_intra_pause_sft.sh` follows the proven format-only pattern verbatim (torchrun 4-proc, `trainer._target_=src.utils.pause_kl_trainer.PauseKLSFTTrainer` is a valid dotted path, the other `+trainer.pause_kl.*` overrides are bare floats/bools/ints — all lexable). Only the pause_token line is new and unlexable.

### 2.5 The 1.5B and 8B configs (Q6) — **GO with adjustments**

**1.5B cot3, 4×A6000** (`stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`):

- Effective batch 2×2×4 = 16; max_steps 400 → 6400 examples ≈ 0.38 epoch of the 17k train split. Right size for a first pilot.
- LR 1e-3 on a single randomly-initialized embedding row (`init_from_text: ""` skips mean-init; `mean_resizing=False` at resize): consistent with the prior format-only runs that worked. Fine.
- `pause_kl` weights (continuation 1.0, pre 0.0, suppression 1.0, emit 0.3, T 1.0, max_kl_tokens 256, `require_pause_before_continuation_kl: true`, both assertions on): sensible starting point. `pre_weight 0.0` is right — pre-pause tokens are teacher-forced identical anyway.
- `save/eval 25` with `save_total_limit 20` keeps all 16 checkpoints — good, because `load_best_model_at_end` selects on the *composite* eval loss and the transparency-optimal checkpoint may not be the composite-loss-optimal one; you want them all for the eval battery.
- Adjust: `eval_steps` 25→50 or subsample val (P3). Early stopping (patience 2, threshold 0.001) on a composite loss whose components move at different speeds (emit CE collapses fast; KL can rise before falling) risks premature stop — set patience 4 for the pilot, or disable and let max_steps rule.

**8B cot4, 4×A100** (`..._8b_cot4_save50_max400_4xa100.yaml`):

- Batch 1×8 GA×4 = 32 effective; max_steps 400 ≈ 0.75 epoch. Fine.
- Inherited `paged_adamw_8bit` is unnecessary for embeddings-only training but harmless and matches the proven 8B env (the `runpod_stage2_env.sh` bitsandbytes native-lib gate covers it). Keep.
- **Fix: `save_total_limit: null`** → unbounded checkpoints: ~8 saves × (~16 GB model + ~8 GB optimizer states for the two untied 128256×4096 matrices) ≈ 200 GB. Set a limit (e.g. 8) or confirm hot-sync-then-delete covers it before launch.
- Do not launch 8B until the 1.5B pilot's eval battery is analyzed (see run order, §7). **Q6 answer: yes, these are good first GPU jobs after the adjustments above — 1.5B first, 8B strictly second.**

### 2.6 Model-comparison eval (Q5) — **structurally blind to self-emission** — full analysis in §4.

### 2.7 Stage3 handoff (Q7) — **COMPATIBLE, three operational notes**

Verified against `run_stage3_intra_pause_probe.py` and `stage3_intra_pause_probe*.yaml`:

- **Checkpoint layout**: the trainer saves a normal HF checkpoint (`save_pretrained`) at `raw/`, step checkpoints, and `final/`. Stage3's `model_path()` (run_stage3_intra_pause_probe.py:53-63) consumes `MODEL` env or `model.sft_checkpoint` — no format change. ✔
- **Tokenizer**: `<|pause|>` added identically to format-only runs (AddedToken + `resize_token_embeddings(len(tokenizer))`); tokenizer saved alongside. Stage3 loads ckpt+tokenizer together, so even if Qwen's padded vocab shrank at resize, the pair is self-consistent. ✔
- **cot3/cot4 convention and position naming**: data prep is unchanged (§2.1), so `pause_0..2`, `pre/post_pause_k`, `cot_3/cot_4` and `cot_offsets [3,4,7,8]` in the Stage3 configs remain valid, including the 8B config's `pause.cot_offset: 4`. ✔
- Notes: (1) `model.sft_checkpoint` in both model YAMLs still points at the **old** format-only runs — must be updated (or `MODEL` env set) at handoff; (2) `final/` = best-*composite*-loss checkpoint due to `load_best_model_at_end` — pick the Stage3 checkpoint from the eval battery, not by trusting `final/`; (3) `save_before_train`'s `raw/` doubles as the base+pause-token control for probes and as the reference for the teacher-equals-base invariant check.

**Q7 answer: yes, compatible without touching Stage3 code.**

---

## 3. Loss / Indexing Proof (Q3, Q4)

Notation: one row, `input_ids[0..L-1]`, `labels` = −100 on prompt/pad, token id on completion (DataCollatorForCompletionOnlyLM with the DeepSeek assistant template, `padding_side="right"`). Pause id `p`.

### 3.1 Teacher input construction — `_pause_stripped_batch`

For each row, iterate `src_idx ∈ [0, valid_len)`; copy non-pause tokens to the teacher sequence, recording `row_mapping[src_idx] = teacher_idx`; skip pause tokens (no mapping entry); right-pad with pad_token_id. Therefore the teacher sequence is exactly the base-model view of the same text — **identical to what the frozen base model would see, since the pause never enters the teacher input**. Mapping is injective and order-preserving on non-pause positions. Correct.

### 3.2 KL pair selection — `_select_kl_pairs`

For each labeled position `t` (label ≠ −100): if `input_ids[t] == p`, increment `pause_seen`, skip (pause targets get CE, not KL). Otherwise, if `pause_seen > 0` (when `require_pause_before_continuation_kl`), emit the pair

```
(batch_idx, student_pos = t − 1, teacher_pos = row_mapping[t] − 1)
```

Alignment argument: causal-LM logits at position `i` predict token `i+1`. The student must predict token `input_ids[t]` from its logits at `t−1`. In the stripped sequence, that same token sits at `row_mapping[t]`, so the teacher predicts it from stripped-logits at `row_mapping[t] − 1`. Both sides predict **the same next token from the same preceding text** — the student's context additionally contains the pause run, the teacher's doesn't. That is precisely the transparency objective.

The crucial edge case — the **first content token after the pause run**: student position `t−1` is the *last pause token* (the student predicts continuation while consuming pauses); teacher position `row_mapping[t]−1` is the last pre-pause content token. The pair is generated and correctly aligned. This is the position where a naïve implementation would be off by `n_pauses`; this one isn't. `_cap_pairs` linspace-subsamples to `max_kl_tokens_per_example=256` — uniform coverage, fine.

**Counterexample search result: none found.** Degenerate cases (pause at completion start; multiple pause runs; rows with no pauses → no pairs → `_zero` fallback; pairs where `t−1` lands on the prompt boundary — fine, logits exist at every position ≥ 0 and `t ≥ 1` because label positions follow the response template) all behave correctly.

### 3.3 KL computation — `_kl_loss`

Both student and teacher selected logits get `[:, pause_token_id] = finfo.min` before softmax; `F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction="batchmean") * T²`. Masking on **both** sides is what makes the objective well-posed: the student is not punished for putting mass on the pause token here (suppression handles that separately, and *only where it should stop*), and the teacher distribution becomes the base distribution restricted to non-pause vocab (see 3.4). One flagged unknown: masked slot → target prob exactly 0 → relies on `xlogy(0,·)=0` inside `kl_div`. Unit-test it (T-3 below).

### 3.4 Same-model teacher validity (Q4) — **valid, with the exact conditions stated**

Teacher forward = same DDP-wrapped model, `torch.no_grad()`, `model.eval()` toggled (config `teacher_eval_mode: true`), on the stripped inputs. Claim: teacher logits over non-pause vocab **exactly equal the frozen base model's logits** throughout training. Proof chain:

1. Rows-only training (`configure_format_only_training` + `mask_embedding_gradients` hooks) ⇒ only the pause row of the (tied or untied) embedding matrices ever receives gradient.
2. `weight_decay = 0.0` ⇒ no gradient-free parameter drift (this is the unguarded link — B3).
3. Teacher input contains no pause token ⇒ the trainable **input** row is never read in the teacher forward.
4. The trainable **output** row affects teacher logits only at the pause slot and via the softmax normalizer — and `_kl_loss` masks the pause slot to −inf on both sides before softmax, which renormalizes over non-pause vocab. The masked-renormalized teacher distribution is therefore invariant to the pause row. ∎
5. **Tied embeddings (Qwen 1.5B)**: input and output matrices are one tensor; the pause row is trainable in both roles. (3) covers the input role, (4) covers the output role — no leak. Untied 8B: both matrices hook-masked independently. **Q4 tied-embeddings answer: handled correctly, by the mask, not by luck.**
6. **Is eval mode enough?** For these architectures (dropout 0.0 throughout DeepSeek-R1-Distill Qwen/Llama configs), train/eval forward are numerically identical anyway; `teacher_eval_mode` is belt-and-braces. Fine.
7. **Is the rows-only assertion enough?** No — see B3. It checks the matrix-level `requires_grad` pattern, not the row-level hooks nor weight decay. It would pass even if `mask_embedding_gradients` were silently never registered — in which case full embedding matrices train, the teacher drifts, and the KL becomes self-referential (the model chases its own moving distribution). That is the one failure mode that breaks the whole method while every existing check stays green. The post-step row-identity test closes it.

### 3.5 CE + suppression — `_pause_losses`

`shift_logits = logits[:, :-1]`, `shift_targets = input_ids[:, 1:]`, `shift_labels = labels[:, 1:]` — standard next-token shift, correct. Emit CE over positions where the shifted target is the pause token (and labeled): trains the model to *start and continue* the pause run at cot3/cot4. Suppression `−log1p(−p_pause)` over labeled non-pause targets (clamped at 1−1e−6): includes the position right after the third pause, so it teaches *stopping* after the run, and covers every other completion position, teaching "don't emit pauses elsewhere". Together with the KL term this is exactly the stated `kl_transparent_emit` objective. Padding/label masking: right padding + label −100 on prompt/pad, all three losses restricted to labeled positions — correct.

### 3.6 DDP / gradient behavior

- Teacher under `no_grad` — no DDP reducer involvement. ✔
- `_zero(logits) = logits.sum() * 0.0` as the empty-mask fallback keeps every loss component graph-connected, so ranks whose micro-batch happens to contain no pause targets or no KL pairs still produce a (zero) gradient for the embedding params and **all-reduce doesn't hang**. This is the classic DDP deadlock for masked losses, and it's handled. ✔
- Gradient checkpointing with only-embeddings trainable: inherited from proven format-only runs. ✔
- `num_items_in_batch` (newer transformers loss-normalization kwarg) is ignored by the custom `compute_loss` — acceptable given per-batch mean reductions; just don't compare absolute loss values across GA settings.

---

## 4. Eval Gaps (Q5) — the blunt version

**Q5.1 — Does the current eval still force-insert pauses? Yes, unconditionally for pause conditions.** `legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py:306-320`: for `kind ∈ {sft, steer}` with `insert_pause_after_cot_tokens ≥ 0`, it generates a 3-token prefix, then `strip_pause_tokens(prefix) + PAUSE_TOKEN * n_insert_pauses`. Read that twice: **it deletes any pause the model emitted naturally, then pastes forced ones in.**

**Q5.2 — Does it measure natural self-emission at all? No.** Nothing in the generation or in `summarize_model_comparison_eval.py` measures unforced emission. `pause3_rate` (pause_count ≥ 3) counts the pauses the harness itself inserted — that's why the prior 8B summary shows pause3_rate = 1.0 everywhere; it's a tautology, not a finding. The only no-insertion pattern in the packet is the *base* condition of `stage2_model_comparison_eval_8b_4xa100.yaml` (`insert -1 / n 0`) — the mechanism exists, it's just never pointed at the trained model.

**Q5.3 — Exact changes needed before any self-emission claim:**

1. Add a `natural` condition per KL checkpoint in the eval config: `insert_pause_after_cot_tokens: -1`, `n_insert_pauses: 0` (config-only change; the generation script already supports it).
2. Extend `summarize_model_comparison_eval.py` with, per response: natural pause count; first-pause token index inside `<think>` (histogram — the claim is mass at cot3/cot4); run-length distribution (the claim is exactly 3, i.e. suppression works at the stop position); off-target pause rate (pauses outside the intended region, incl. after `</think>` — suppression working globally).
3. Keep the forced-insertion condition as a *separate* sanity check (does the model continue coherently after forced pauses — this is what Stage3/Stage4 actually need).

**Q5.4 — Missing metrics for transparency / KL / behavior drift:**

- **Uncapped KL drift**: training only ever sees post-pause KL capped at 256 tokens/example. Post-hoc: mean/p95/max per-token KL(base ‖ ckpt) over full held-out no-pause completions, plus the same on pause-containing inputs at aligned positions. This is the transparency claim's primary evidence.
- **Behavior equivalence vs base**: gsm8k/math500 and safety-judge rates for the KL checkpoint under **no insertion**, deltaed against base (the current eval compares conditions, but only under forcing).
- Qualitative: side-by-side CoT on fixed prompts, base vs checkpoint, no insertion.

---

## 5. Required Code Changes (exact)

| # | File | Function/site | Change | Class |
|---|------|----------------|--------|-------|
| C1 | `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh` | PAUSE_KL `HYDRA_ARGS` block | Delete the `pause_token` override (default already correct), or quote as `"...pause_token='$PAUSE_KL_PAUSE_TOKEN'"`; verify with the §1 one-liner | **Blocker (B1)** |
| C2 | `legacy/COTPauseToken/src/utils/pause_kl_trainer.py` | `_assert_rows_only_training` (+ first-step callback) | Assert `weight_decay == 0.0`; snapshot non-pause rows, assert bit-identity after step 1 | **Blocker (B3)** |
| C3 | same | `_pause_stripped_batch`, `_select_kl_pairs` | One `.cpu().tolist()` per tensor per batch; drop per-token `.item()` | Strongly rec. (P1) |
| C4 | same | `_pause_losses` suppression branch | Pause-column logprob via chunked `logsumexp` instead of full-vocab fp32 `log_softmax` | Strongly rec. (P2) |
| C5 | `configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_...yaml` | `eval_steps`, `early_stopping` | 25→50 (or subsample val); patience 2→4 | Recommended (P3) |
| C6 | `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_...yaml` | `save_total_limit` | `null` → 8 (or confirm hot-sync deletion) | Recommended |
| C7 | eval configs + `summarize_model_comparison_eval.py` | new `natural` condition + emission/KL metrics per §4.3–4.4 | Required before claims, not before training | **Claim-blocker (E1)** |
| C8 | packet only | add `template_deepseek_r1_distill.yaml`, `runpod_base_env.sh` | Packet completeness | Hygiene (P4) |

No changes required to: data prep, validator, `run_stage2_sft.py`, config loader, Stage3 runner/configs. `trl_train.py` needs no change (its `build_trainer_config`/format-only/pause-row plumbing is correct as-is).

---

## 6. Required Tests (Q8)

### 6.1 Local unit tests (CPU, tiny model — e.g. 2-layer random Qwen2 config + added pause token) — **before any GPU**

- **T-1** `_pause_stripped_batch`: synthetic rows with known pause positions (start/middle/multiple runs/none) → assert stripped ids, mapping, padding.
- **T-2** `_select_kl_pairs`: assert for every pair `input_ids[b, s+1] == stripped_ids[b, t+1]` (same predicted token); assert the last-pause→first-content pair exists; assert no pairs before the first pause when `require_pause_before_continuation_kl`.
- **T-3** `_kl_loss` with masked pause slot: loss finite (the `xlogy` zero-target check); identical student/teacher logits → loss ≈ 0.
- **T-4** `_pause_losses` on one hand-computed example: emit CE and suppression match manual values; shift off-by-one would fail this immediately.
- **T-5** rows-only invariant: one real optimizer step; non-pause rows bit-identical, pause row changed; repeat with `weight_decay=0.01` and assert the guard **fails** (proves C2 works).
- **T-6** teacher-equals-base: fresh resized model vs a frozen copy — teacher non-pause logits equal on stripped input; repeat after 5 train steps.
- **T-7** end-to-end `compute_loss` on a tiny batch incl. a row with no pauses: finite loss, gradients only on embedding params, no NaN.

### 6.2 Single-GPU tiny smoke (1.5B, ~15 min)

Real config with `max_steps 20`, ~50 train / 32 eval rows: Hydra override line parses (B1 verification in situ); loss parts logged and finite; checkpoint saves and reloads (model+tokenizer roundtrip, pause id stable); peak memory recorded; post-run non-pause rows identical to `raw/`.

### 6.3 4-GPU smoke (torchrun, ~20 min)

20 steps on 4×A6000: no DDP hang (specifically survive micro-batches where a rank has zero pause targets — the `_zero` path); NCCL clean; rank0-only save; resume-from-checkpoint works; throughput number to sanity-check the 400-step ETA (and to quantify P1 if unfixed).

### 6.4 Post-training invariants (after the pilot, before Stage3)

Non-pause embedding rows == `raw/` exactly; `len(tokenizer) == model vocab`; forced-pause generation coherent; Stage3 hidden extraction dry-run on one shard finds `pause_0..2` positions.

---

## 7. Minimum Post-Checkpoint Eval (Q9) + First Run Order

**Eval battery (per candidate checkpoint, in this order):**

1. **Natural emission** (new condition, §4.3): emission rate, first-pause position vs cot3/cot4, run-length==3 rate, off-target rate — on ~200 held-out prompts, no insertion.
2. **Uncapped KL drift** vs base: mean/p95/max per-token KL on ~100 held-out no-pause completions.
3. **Behavior equivalence**: gsm8k/math500 subset + safety judges, no insertion, delta vs base.
4. **Forced-insertion sanity**: existing eval path unchanged.
5. **Stage3 probe readiness**: small-shard hidden extraction + a single-layer probe fit on the chosen checkpoint.

**Recommended first run order:**

1. C1 fix + Hydra parse one-liner on pod (§1) — 10 min.
2. C2 (+ ideally C3/C4) and the unit tests T-1…T-7 — ~half day.
3. Single-GPU tiny smoke (§6.2), then 4-GPU smoke (§6.3).
4. **1.5B cot3 full pilot** (400 steps, 4×A6000) with C5 cadence.
5. Eval battery on the 3–4 most promising checkpoints (by eval-loss parts, not composite) + post-training invariants.
6. Decide checkpoint → update `model.sft_checkpoint` → **Stage3 probe run** on 1.5B.
7. Only after 4–6 are analyzed: **8B cot4 pilot** (with C6), then its eval + Stage3.

## 8. What Not To Claim Yet

- That the model **self-emits** pauses — the eval literally cannot see this today (§4).
- That **transparency is preserved** — training KL is capped/post-pause-only; the uncapped drift number doesn't exist yet.
- That **cot3 vs cot4 conclusions transfer** — the prior 8B summary (cot4 > cot3, full-SFT drifts) is format-only/full-SFT evidence; it motivates this method but does not predict its behavior.
- That the **8B recipe is validated** — nothing 8B-specific has run under this trainer.
- Anything about **Stage4 steering** — explicitly out of packet scope.

## 9. Final Go/No-Go

| Milestone | Verdict | Gate |
|---|---|---|
| Code review packet (experimental branch) | **GO (conditional)** | Method + plumbing sound; land C1, C2 (and C8 for the packet); Q10: acceptable with exactly those blockers fixed |
| 1.5B smoke (single-GPU then 4-GPU) | **NO-GO → GO** | After C1 verified + T-1…T-7 pass |
| 1.5B full Stage2 pilot (400 steps) | **NO-GO → GO** | After both smokes pass; C3/C4/C5 strongly advised first |
| 8B pilot | **NO-GO** | Until 1.5B pilot eval battery (§7) is analyzed; then GO with C6 |
| Stage3 handoff | **NO-GO** | Until post-training invariants (§6.4) + probe-readiness check pass and `model.sft_checkpoint` is repointed; interface itself already compatible (§2.7) |

**Bottom line:** the hard part — the loss and its indexing — is right, and the same-model-teacher construction is sound *under the exact rows-only + mask + zero-weight-decay regime this packet configures*. What stands between you and GPU spend is one shell-quoting bug that only manifests at Hydra parse time, one missing invariant guard, and a half-day of unit tests. What stands between you and any scientific claim is an eval that currently deletes the very behavior it needs to measure.