# Fable5 full code review — formal Stage 2–4, 8B, 2×A100

**Branch** `stage2-4-full-2xa100-review-260714` @ `00c7493f438190e203438ddb1bc4488e999ec016` · **Mode** pre-run, executable-code review · **Date** 2026-07-14

---

## 0. Review method and evidence status

I traced real call paths (not prose) across the three mixed training entrypoints, both checkpoint watchers, the restore path, the data freeze, all formal Stage-3 modules and scripts, and every formal Stage-4 module/script listed in the request, including resume/failure branches. Files read in full or targeted depth: `scripts/run_stage2_sft.py`, `legacy/COTPauseToken/src/trl_train.py`, `pipelines/runpod_watch_hot_checkpoints.sh`, `pipelines/runpod_watch_cold_checkpoints_to_r2.sh`, `pipelines/runpod_sync_hot_to_cold.sh`, `scripts/restore_stage2_terminal_from_r2.py`, `src/cot_safety/data/stage2_formal_freeze.py`, `src/cot_safety/training/stage2_model_binding.py`, `src/cot_safety/probes/stage3_{formal,artifacts,rollouts,replay,bridge,hidden_replay,diagnostics}.py`, `scripts/analyze_stage3_formal.py`, `src/cot_safety/steering/stage4_{formal,generation}.py`, `src/cot_safety/eval/stage4_{calibration,formal_analysis}.py`, `scripts/{analyze_stage4_formal_calibration,run_stage4_formal_generation_hf,run_stage4_formal_benign_generation_hf,analyze_stage4_formal_results}.py`, and both formal experiment configs (`configs/experiment/stage3_formal_8b_2xa100.yaml`, `configs/experiment/stage4_full_sft_clean_8b_2xa100.yaml`).

**Tests actually attempted vs approval-blocked:**

| Check | Attempted | Result |
|---|---|---|
| `python -m pytest tests/ -q` | 3× (background ×2, foreground ×1) | **Approval-blocked by the review sandbox** — never executed by me |
| `python -m py_compile` (Stage-4 modules/scripts) | 2× | **Approval-blocked** |
| `bash -n` on the three watcher/sync scripts | 1× | **Approval-blocked** |
| `git diff --check` | 1× | **Ran — clean, no whitespace errors** |
| `git status` at review start | via harness snapshot | Clean tree at `00c7493` |
| Test-evidence inspection (`ls tests/`, `ls -la tests/__pycache__`) | ran | 59 test files; `__pycache__` artifacts stamped **2026-07-14 03:00, pytest-9.1.1/py3.12** — the suite was executed on this machine today |
| Skip-surface audit (Grep for `importorskip`/`skipif`) | ran | Skips are `pytest.importorskip("torch")`/`("transformers")` guards at `tests/test_stage3_hidden_replay.py:96-97`, `tests/test_stage4_generation.py:7`, `tests/test_stage2_pause_kl_trainer.py:11`, `tests/test_stage3_artifacts.py:253` — consistent with the branch's claim that the 4 skips are optional real-Torch model-hook tests, not failed assertions |

I therefore rely on static call-path verification plus the branch's recorded `172 passed, 4 skipped`; I could not independently re-execute the suite. **The 2×A100 preflight must re-run the suite in the launch environment.**

---

## 1. Findings

### MAJOR-1 — Stage 2 has no instantiated base-model identity gate

- **Evidence.**
  - `scripts/run_stage2_sft.py:52-59` — a `MODEL` environment variable force-overrides the model path with zero validation (`forced = os.environ.get("MODEL"); if forced: return forced`); the fallback chain (`local_base_model` → `base_model`) is likewise unvalidated against the frozen 8B identity. `TOKENIZER` is similarly overridable at `:62-67`.
  - The legacy Hydra language-model config chain can fall back to a DeepSeek-R1-Distill-**Qwen-1.5B** path via env default when the canonical env var is not exported, and the launcher never cross-checks `$MODEL_PATH == $FULL_SFT_BASE_MODEL_PATH`.
  - `legacy/COTPauseToken/src/trl_train.py:870-872` — provenance hashes `FULL_SFT_BASE_MODEL_PATH` (an env-named directory) via `directory_content_manifest`, but never cross-checks that directory against the path the **instantiated** model was actually loaded from.
  - `legacy/COTPauseToken/src/trl_train.py:595` — the first-step gradient audit derives `expected_layers = int(getattr(self.model.config, "num_hidden_layers", 0))`, i.e. it is architecture-relative: a wrong model (e.g. a 28-layer Qwen-1.5B) passes its own gradient-coverage audit.
  - `legacy/COTPauseToken/src/trl_train.py:44-59` — `add_special_tokens` silently proceeds when `n_added == 0` (the `if n_added:` block is simply skipped); there is no assertion that the pause token was added exactly once with a fresh id, nor that a pre-existing id equals the expected one.
- **Consequence.** A mis-set environment can full-SFT the wrong base model while producing an internally consistent provenance chain *for that wrong model*. Downstream stages then bind honestly to the wrong checkpoint (the hash chain holds), so the failure mode is a **wasted multi-day 2×A100 run plus a false "DeepSeek-R1-Distill-Llama-8B" claim**, discovered only by post-hoc inspection. Partial mitigations exist downstream — Stage-4 verifies the A0 base directory content hash against provenance (`scripts/run_stage4_formal_generation_hf.py:535-549`), and all runtime loads rehash sealed checkpoint-1064 via `verify_runtime_checkpoint` (`src/cot_safety/training/stage2_model_binding.py:125-157`) — but nothing at Stage-2 training time pins the base architecture or the load-path identity.
- **Required fix.** Add one fail-closed gate in `trl_train.py` before training starts, recorded in the runtime audit:
  1. `model.config.model_type == "llama"`, `num_hidden_layers == 32`, `hidden_size == 4096`, expected `vocab_size` after pause-token addition;
  2. the resolved load path of the instantiated model (`model.config._name_or_path` / the path actually passed to `from_pretrained`) string-equals the directory hashed at `trl_train.py:870-872` and equals `run_stage2_sft.py`'s `selected_model`;
  3. pause-token gate: either `n_added == 1` with embedding resize verified, or the token pre-exists at exactly the expected id — anything else aborts.

### MAJOR-2 — First allocated bnb optimizer state is never inspected (audit decision: this must be a hard gate)

- **Evidence.** The optimizer audit locks the *configured* object — instantiated class `bitsandbytes.optim.adamw.AdamW`, literal `is_paged=True`, `optim_bits=8`, exact betas/epsilon/weight-decay, full parameter coverage, instantiated `SFTTrainer.max_seq_length == 4096` — but no code in `trl_train.py` or its callbacks ever reads `optimizer.state[p]["state1"]`/`["state2"]` after the first step, checks their dtype/shape coverage, or verifies paged-manager registration. A repo-wide grep confirms no `state1`/`state2`/page-manager inspection exists anywhere.
- **Consequence.** The protocol's own philosophy is "instantiated—not merely configured—objects prove the claim." bitsandbytes allocates 8-bit state lazily at the first optimizer step and silently keeps fp32 state for tensors below `min_8bit_size=4096` (correct for norms/biases, but currently unproven for the 8B weight matrices). A packaging/dispatch regression could run fp32 or unpaged state while provenance still records `paged, 8-bit` — exactly the configured-vs-instantiated gap this review was asked to close, with direct memory-headroom consequences at the worst-case 4,096-token workload.
- **Required fix.** Extend the existing first-optimizer-step audit callback: for every trainable parameter with `numel() >= 4096`, assert `state1` and `state2` exist with dtype `torch.uint8` and matching shape coverage; assert per-tensor `qmap`/`absmax` present; when `is_paged=True`, assert the state tensors are registered paged buffers (bnb paged-tensor attribute / `GlobalPageManager` membership); write the result into `stage2_pretrain_runtime_audit.json` (the bundle written at `trl_train.py:861-868`) and abort on any mismatch. Cost: one step, negligible.

### MINOR-1 — `audit_canonical_training_arguments` does not pin `eval_strategy` / `save_strategy`

- **Evidence.** The expected-arguments dict in `run_stage2_sft.py`'s TrainingArguments audit pins `max_steps=-1`, epochs, batch/accumulation, seed 260615, no early stopping, and `load_best_model_at_end=False` (also forced at `run_stage2_sft.py:630`), but omits the two strategy enums.
- **Consequence.** An inherited or overridden strategy could shift evaluation/save cadence without tripping the audit. Checkpoint-1064 sealing is still enforced downstream, so this is drift surface, not corruption.
- **Fix.** Add both keys to the expected dict.

### MINOR-2 — Orphaned cold `.partial.$$.$RANDOM` directories are never garbage-collected

- **Evidence.** `pipelines/runpod_sync_hot_to_cold.sh` writes into a unique partial directory and atomically renames on success; a crash strands the partial forever, and no GC pass exists in any watcher.
- **Consequence.** Disk leak on `/workspace` only. Partials are never readable as final; the restore script cleans *its own* partials, so no correctness impact.
- **Fix.** On watcher start, delete `*.partial.*` older than a threshold within the managed cold root only.

### MINOR-3 — Mixed `python` / `python3` invocations in watcher heredocs

- **Evidence.** `pipelines/runpod_watch_cold_checkpoints_to_r2.sh` mixes interpreter names between embedded verification snippets.
- **Consequence.** On a host without a `python` alias the watcher exits early — which by design **stops training** (fail-closed, loud), but for a spurious reason mid-run.
- **Fix.** Normalize to `python3` (or a single `"$PYTHON_BIN"`) everywhere.

### MINOR-4 — Hot→cold `final/` export is rsync'd without its own hash verification

- **Evidence.** The terminal `final/` model export is synced but not sealed/receipt-verified like `checkpoint-1064`.
- **Consequence.** Fully mitigated in-protocol: every downstream consumer loads and **rehashes sealed `checkpoint-1064`** via `verify_runtime_checkpoint` (`stage2_model_binding.py:125-157`; call sites: `run_stage3_formal_rollouts_vllm.py:114`, `extract_stage3_formal_hidden.py:200`, `verify_stage3_vllm_hf_bridge.py:333,455`, `run_stage4_formal_generation_hf.py:518`, `run_stage2_formal_acceptance_vllm.py:214`, `restore_stage2_terminal_from_r2.py:70`, and the benign engine via `SHARED._load_terminal_stage2_binding` at `run_stage4_formal_benign_generation_hf.py:170-171`); nothing consumes `final/`.
- **Fix (optional).** Either seal `final/` with the same manifest machinery or drop it from the lifecycle, removing an attractive-but-unverified artifact.

### Observations (no action required)

- `validate_calibration_generation_design` rejects any out-of-grid cell (`foreign_calibration_cell`, `src/cot_safety/eval/stage4_calibration.py:60-71`) *before* `select_calibrated_strength` groups per-candidate rows (`scripts/analyze_stage4_formal_calibration.py:187` runs design validation ahead of selection at `:203`), so the selector's exact-alpha matching is defense-in-depth, not a hole.
- The degeneration endpoint deliberately excludes judge-unknown (it is a token-statistics endpoint); judge-unknown separately enters the primary outcome as a conservative failure (`join_safety_judges`, `src/cot_safety/eval/stage4_formal_analysis.py:674-703`). Correct design; worth one sentence in the paper.

---

## 2. Mandatory implementation audit — results

### 2.1 Stage 2 optimizer and memory (`paged_adamw_8bit` vs `adamw_bnb_8bit`)

**Decision: keep `paged_adamw_8bit` canonical. Do not switch on static estimates.** Static arithmetic (8B params: ~16 GB bf16 weights + ~16 GB bf16 grads + ~16 GB paired 8-bit state, replicated per GPU under DDP, plus micro-batch-1 × 4,096-token activations under checkpointing) says an 80 GB A100 *should* hold this without UVM pressure — in which case paged and non-paged execute identical AdamW-8bit math, paging never activates, the expected speed delta is ~zero, and the paged variant is strictly safer against transient spikes. But that is precisely a static estimate, which the contract forbids acting on. The branch's choice — lock the paged canonical in code and provenance — is correct.

**If an A/B preflight is ever run, the minimal apples-to-apples protocol is:** one node; two clean single-run processes (no warmed-allocator reuse); identical seed/data/config; exact worst-case workload (micro-batch 1 × 4,096 tokens, GA 16); 50 optimizer steps each; only the `optim` string differing. **Hard abort thresholds:** any nonfinite loss/gradient; any parameter-coverage mismatch; `torch.cuda.memory_allocated/reserved` and NVML used-memory divergence > 5% between arms at matched steps; any CUDA-malloc retry event; for the non-paged arm, UVM page-fault counters and host↔device PCIe traffic attributable to optimizer state must be exactly zero; step-time medians reported with IQR. Adopt the non-paged variant only if ≥ 10% faster *and* every gate passes. Fused PyTorch AdamW, Adafactor, Lion, GaLore, etc. are different algorithms and out of scope.

### 2.2 Full-train and provenance (instantiated proofs)

Verified present at instantiated level: full 8B parameter trainability with no PEFT; exact optimizer class/flags/hyperparameters; first-step finite-nonzero gradient audit across decoder layers and pause-token embedding/output rows; middle-layer checksum change after the first optimizer step; package/CUDA/driver/NCCL/vLLM/rclone versions; base-model directory hash, tokenizer/chat-template/pause-token hash, dataset/freeze/config/code hashes; terminal checkpoint-1064 provenance; environment sanitization strips inherited `MAX_STEPS`/early-stopping/best-checkpoint/stale-resume state (`run_stage2_sft.py:594,630`; audit bundle at `trl_train.py:861-868`). The 1,064-step arithmetic (ceil(8500/16) × 2 epochs, 2 GPUs, GA 16, global batch 32, seed 260615, `max_steps=-1`) is enforced as a step-compatibility audit, not merely configured. **Gaps = MAJOR-1 (base identity) and MAJOR-2 (allocated optimizer state).**

### 2.3 Checkpoint storage lifecycle

`/dev/shm` hot → sealed `/workspace` cold → verified R2 → local deletion holds fail-closed in both directions: sealed payload manifests + completion markers; destination-bound transfer receipts (`scripts/checkpoint_integrity.py` `write-receipt`/`verify-receipt` with `--kind {cold,r2}` and `--destination`); R2 uploads land on manifest-bound partial keys and finalize only after `rclone check --download` (download-side re-hash); **no local checkpoint, final model, or metadata deletion before a destination-bound receipt and download-side hash check succeeds**; either watcher exiting kills training (watcher liveness enforced by the launcher's `run_logged`); the final hot sync completes before R2 finalization. Restore (`scripts/restore_stage2_terminal_from_r2.py`, 214 lines) downloads into a same-parent `.partial` directory, verifies checkpoint-1064 + provenance + receipt, atomically renames, removes failed partials without exposing an unverified final directory, and re-verifies via `verify_runtime_checkpoint` (`:70`). Findings here: MINOR-2, MINOR-3, MINOR-4 only.

### 2.4 Backend boundaries

- vLLM is used only for natural fixed-budget rollouts and open-model judging; HF exact-token replay/hooks are the sole hidden-extraction and steering path. The bridge is gated: 32 prompts, 100% prompt-token/position agreement, greedy first-64 ≥ 0.99, chosen-logprob coverage 100% with median ≤ 0.02 / p99 ≤ 0.10, and `sealed_open_authorized` must be true with the runtime model SHA bound (`src/cot_safety/probes/stage3_bridge.py`; binding enforced at `stage3_artifacts.py:142-194`).
- **No code assumes a hidden-index-32 edit reaches future KV.** Extraction hooks are read-only (`stage3_hidden_replay.py:113-181`; index 32 captured post-final-norm as a readout diagnostic). Steering rejects index 32 twice: `validate_artifact_binding` requires `1 <= layer < 32` (`src/cot_safety/steering/stage4_formal.py:143-145`) and `hidden_index_to_block_index` bounds the hook mapping (`src/cot_safety/steering/stage4_generation.py:357-363`). The intervention edits decoder-block `l−1` output on the single shape-guarded full-prefix forward (`:451-454`), implements exactly `h' = h − rho·‖h‖·û` (`delta = -rho * hidden_norms * local_direction; updated = selected + delta`, `:461-464`; `rho = alpha × 0.10` with `rho ∈ (0, 0.10]` at `:397-398`), must apply exactly once (`:802-805`), and `prefix_kv_integrity_preflight` + `later_kv_change_report` (`:878-963, 1015-1057`) prove later-layer KV actually changed — as an execution preflight, never an analysis outcome or a sample source.
- **Terminal scheduled failures stay in all denominators and are never replaced by extra draws:** Stage-3 materializes failure rows with bound failure hashes and no replacement (`stage3_rollouts.py:169-227`); eligibility requires `scheduled == 100` **exactly**, so extra draws also fail closed (`stage3_formal.py:438-441`); the hidden bundle must cover the exact 40,000 canonical cell ids with no missing/extra/re-bound cells (`stage3_artifacts.py:816-853`). Stage-4 forbids resampling (`resampled is False`, `regeneration_attempts == 0`, `stage4_calibration.py:210-213`); A1 failures propagate into the A2 alias as `a1_generation_unavailable` (`stage4_formal.py:801-821`); judges attach terminal failure rows to failed generations (`stage4_formal_analysis.py:577-596`); missing judge labels become `unknown` → conservative failure (`:674-703`); and `absolute_residual_summary` uses `n_scheduled` denominators (`stage4_formal.py:1168-1362`).

### 2.5 Stage 3/4 leakage and calibration

All requested properties verified in code:

- **Sealed labels cannot choose the final Stage-4 layer/direction.** `select_layer_training_only` (`stage3_formal.py:702-772`, deterministic lower-layer tie-break) and `run_nested_four_source_loso` (`:846-970`) fit/select on TRAIN-split prompts only; sealed data is used solely for outer-fold evaluation. `assert_training_only_direction` rejects any non-TRAIN prompt direction and requires unit norm and a primary-grid layer (`stage3_artifacts.py:1063-1077`); config must declare `final_stage4_layer_uses_sealed_results: false` (`:1045-1046`); `write_direction_artifacts` re-reads the on-disk report and refuses unless `status == "pass"` and `gate.passed is True` (`:1176-1181`); gate failure yields `artifact_withheld.json` + `SystemExit` (`scripts/analyze_stage3_formal.py:117-127`).
- **Diagnostics cannot enter the signal gate.** `run_stage3_diagnostics` only reads the main report's outer folds as a consistency check (raising on layer mismatch, `stage3_diagnostics.py:565-570, 650-675`); prompt-only probes (`PROMPT_POSITIONS = ("last_prompt_token","pre_think")`, `:308-423`) and nuisance audits record `main_gate_fields_modified: []`; `artifact_authorized` derives from the gate alone (`stage3_artifacts.py:1080-1144`).
- **The calibration report mechanically proves everything the contract lists.**
  - *A1/A2 alpha-zero token identity:* bit-exact equality on `prompt_token_ids`, `output_token_ids`, `generated_content_sha256`, `generated_text_sha256` plus the `rho_zero_bit_exact` flag (`stage4_calibration.py:291-307`; alias construction with `physical_touches: 0` and policy `exact_a1_reference_alias_no_forward` at `stage4_generation.py:966-988`; the flag is bound into `row_integrity_sha256` so it cannot be flipped after generation, `:1166`; resume re-verifies every alias invariant, `:1228-1252`).
  - *Frozen candidate order:* candidates must match the frozen grid `(0.10, 0.25, 0.50, 1.00)` in order with exact `rho = alpha × 0.10` arithmetic (`run_stage4_formal_generation_hf.py:668-685`).
  - *Target-norm integrity:* per-alpha `target_norm_integrity` with 1% tolerance and exactly 3 touches (`stage4_formal.py:873-877`); re-checked per candidate at report load (`run_stage4_formal_generation_hf.py:691-697`).
  - *First-passing alpha:* the loader independently re-derives each candidate's pass/fail from its own metrics (reduction ≥ 0.03, degeneration increase ≤ 0.02, norm pass, with arithmetic-consistency checks) and requires `min(passed) == selected` (`:733-745`). Selection scope is pinned to `["stage4_calibration","A1","A2"]` (`:780-781`) — A3/A4/final-test selection is impossible. Calibration generations are forbidden from binding their own report (`stage4_calibration.py:82-83`; `binding_payload` raises if a calibration-phase row carries a report SHA, `stage4_generation.py:1359-1362`). The 4×20×10×6 = 4,800-row schedule, single cross-row binding signature, seed re-derivation via `stable_rollout_seed(260713, …)`, and arm-invariant counter keys are all fail-closed (`stage4_calibration.py:182-185, 280-281, 230-241, 242-253`).
- **One report SHA everywhere.** `binding_payload` requires `calibration_report_sha256` for `final` and all three benign phases and forbids it during calibration (`stage4_generation.py:1349-1362`); both generation engines load the report through the same fail-closed loader, which also cross-checks all 9 binding hashes (model / tokenizer / artifact-manifest / config file+resolved / ledger+manifest / stage2-provenance / completion-marker) against the live run (`run_stage4_formal_generation_hf.py:783-808`; benign at `run_stage4_formal_benign_generation_hf.py:114,185,204-222`, with the runtime checkpoint bound via `SHARED._load_terminal_stage2_binding` → `verify_runtime_checkpoint`, `run_stage4_formal_generation_hf.py:506-532`); the final analysis re-hashes the report file and requires harmful, capability, and compliance rows to bind those exact bytes, and threads the same SHA into the semantic bundle manifest (`scripts/analyze_stage4_formal_results.py:264-278, 296-303`); cross-arm binding equality including `calibration_report_sha256` is enforced per shared cell, A0 included, with cross-arm rho equality and position relative-norm mismatch ≤ 0.01 (`stage4_formal_analysis.py:249-381`, cap at `:376-381`); the exact 24,000-row A0–A5 design is enforced (`:384-468`); `provenance_manifest` byte-binds every input, the config, and the implementation files (`:1439-1509`); bootstrap 10,000 / seed 260713 is pinned (`analyze_stage4_formal_results.py:238-241`).

### 2.6 Data freeze — explicit answer to the required question

The current implementation is exhaustive at ≥ 0.90 and content-bound, and requires a manual decision for **exactly** the set `top_neighbors ∪ threshold_hits` (exact set equality at `src/cot_safety/data/stage2_formal_freeze.py:464-467`; exhaustive comparison-count proof at `:395-403`).

**Answer: interpretation (2) is the scientifically necessary minimum — and it is what the code already enforces, plus one clarification.** The scientific requirement is that (a) every unique ≥ 0.90 pair receives an explicit human decision (the near-duplicate exclusion instrument), and (b) every candidate's nearest neighbor is automatically recorded and hash-bound (the audit trail proving the threshold saw everything). Manually adjudicating all ~18,000+ per-candidate nearest neighbors (interpretation 1) adds no exclusion power — a nearest neighbor below 0.90 is below the preregistered near-duplicate criterion by construction — and would only manufacture reviewer fatigue. A preregistered global/stratified top-K or gray-band `[0.85, 0.90)` sample is a worthwhile *sensitivity check* on threshold placement, not a freeze requirement; if adopted, encode it as an explicit config block with its own recorded decisions.

**Connected-component adjudication:** permitted **only as a labeling convenience, never as a reduction in listed pairs** — a component-level decision must be materialized into explicit per-pair decision records (with a recorded `component_id`) so the exact-set-equality check at `:464-467` still passes over the full `top_neighbors ∪ threshold_hits` set. Do not weaken the fail-closed set equality; any rule change must appear as code + config + tests, per the contract.

---

## 3. The four requested decisions

1. **`paged_adamw_8bit` vs `adamw_bnb_8bit`:** keep `paged_adamw_8bit` canonical. No speed advantage may be claimed without the GPU A/B preflight specified in §2.1; static reasoning suggests paging never activates at this workload, making the paged variant equal-speed and strictly safer.
2. **First-allocated optimizer-state inspection:** **YES — hard pre-run gate** (MAJOR-2): `state1`/`state2` dtype `uint8` + shape coverage for all ≥ 4096-element trainable tensors + paged-manager registration, asserted at the first optimizer step and recorded in the runtime audit.
3. **Must the A/B benchmark precede `APPROVE_TO_RUN`?** **No — optional performance work.** The paged optimizer is the provenance-locked canonical choice with identical AdamW-8bit semantics; the benchmark cannot change the science and must not block the run.
4. **Top-neighbor manual audit:** interpretation (2), as specified in §2.6; component adjudication only via materialized per-pair records; no silent weakening of the fail-closed implementation.

---

## 4. Professor questions — implementation status

1. **Clean intervention point (A2 vs A3, A2 vs A4).** Implemented: frozen matched arms (`stage4_formal.py:44-69`; config `arms` block), identical rho/direction/CRN across arms per shared cell (`stage4_formal_analysis.py:249-381`; arm-invariant counter keys, `stage4_calibration.py:242-253`), capability/compliance/degeneration non-inferiority gates, and semantic-continuity comparisons `[A2,A3],[A2,A4]` with blinding, order randomization, and reversed repeats (config `semantic_continuity`).
2. **Prompt-only / pre-CoT probes.** Implemented as diagnostics: `PROMPT_POSITIONS = ("last_prompt_token","pre_think")` with training-only inner selection (`stage3_diagnostics.py:308-423`); structurally prevented from entering the signal gate.
3. **Honest four-source LOSO.** Implemented: nested four-source LOSO with per-fold transfer reported; gate requires macro CI-low > 0.55 strict, ≥ 3 sources ≥ 0.55, none < 0.50 (`stage3_formal.py:791-970`). No six-source claim survives in the formal path.
4. **SFT-data effect vs steering + nuisance audit.** Implemented: A0 vs A1 arms separate the SFT effect; `outer_fold_nuisance_diagnostic` audits output/prompt length, WildGuard refusal, and source-heldout hashed-unigram surface associations on sealed outer-heldout scores only, non-gating (`stage3_diagnostics.py:529-647`).
5. **Teacher-forced/vLLM/HF vs self-generated mismatch.** Named and bridged: gated 32-prompt vLLM↔HF bridge (§2.4); the Stage-4 estimand is declared as `minimal_prefix_target_window_conditioned_online_continuation` with exact-A1-token replay boundaries per arm and free continuation after the window (config `online_counterfactual`; `counterfactual_generate_batch`, `stage4_generation.py:712-838`).
6. **Absolute residual unsafe rates.** Implemented: `absolute_residual_summary` reports unsafe_all/unsafe_valid/conservative with scheduled-cell denominators, unknown/failure included, per-source and source-equal macros with cross-source SD/range, and every reduction paired with its absolute residual (`stage4_formal.py:1168-1362`; config `reporting.always_pair_reduction_with_absolute_residual: true`).

---

## 5. Verdict

## `APPROVE_AFTER_LISTED_FIXES`

**Required before the 2×A100 launch:**

1. **MAJOR-1** — Stage-2 instantiated base-model identity gate (architecture pin llama/32/4096, load-path ↔ hashed-path equality, pause-token `n_added==1`-or-expected-id gate).
2. **MAJOR-2** — first-optimizer-step allocated-state gate (`state1`/`state2` uint8 + coverage + paged-manager registration), recorded in the runtime audit.

**Strongly recommended, non-blocking:** MINOR-1 (pin `eval_strategy`/`save_strategy`), MINOR-2 (partial-dir GC), MINOR-3 (`python3` normalization), MINOR-4 (seal or drop `final/`).

The frozen scientific contract is otherwise implemented faithfully and fail-closed end to end. No archival protocol (LoRA, rows-only KL, PPC, learned-delta, projection-clamp, forced-runtime-pause) is reachable from the three formal entrypoints, and the negative sentinels are asserted at generation time (`binding_payload`, `stage4_generation.py:1387-1392`) and re-checked at analysis (`stage4_calibration.py:155-159`). Additionally, the launch environment must re-run the CPU/static suite (expected `172 passed, 4 skipped`), since this reviewer's sandbox blocked Python execution and the suite result was verified only via recorded artifacts and static skip-surface inspection.
