# Fable5 Round-2 full executable code review — Stage2–4 full-SFT 2×A100

- Date: 2026-07-14
- Reviewer: fresh independent Round-2 reviewer (no shared state with Round-1)
- Branch: `stage2-4-full-2xa100-review-260714`, HEAD `6c1171c`
- Frozen code candidate: `aeaf4e5` — verified: HEAD differs from `aeaf4e5` only by the 140-line Round-2 request document; working tree clean
- Round-1 baseline: `00c7493`; normative spec: `IMPLEMENTATION_SPEC.md`
- Environment caveat honored: **no 2×A100, CUDA, NCCL, or UVM evidence exists**; nothing below treats local mocks or static tests as GPU evidence. All bitsandbytes/torch runtime behavior was verified statically against the pinned versions (bnb 0.46.1, transformers 4.52.4, TRL 0.8.1).

## 1. Review scope and commands actually executed

**Code read in full (candidate `aeaf4e5` content):** `src/cot_safety/training/full_sft_contract.py` (2,221 lines), `full_sft_runtime.py` (1,224), `legacy/COTPauseToken/src/trl_train.py` (2,172), `scripts/run_stage2_sft.py` (1,558), `src/cot_safety/training/cold_partial_gc.py` (384), `scripts/gc_stage2_cold_partials.py` (69), `src/cot_safety/training/stage2_model_binding.py` (166), `configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml`, `configs/provenance/deepseek_r1_distill_llama_8b_6a6f4aa_runtime_files.json`, the `checkpoint_integrity.py` delta, the full `git diff 00c7493..aeaf4e5` for the shell contract (`run_4gpu_intra_pause_sft.sh`), both watcher scripts, and `IMPLEMENTATION_SPEC.md`; the R2 watcher's verify→upload→check→receipt→delete core and hot watcher's seal/copy/GC path were re-inspected line-level; Stage3 grid/adequacy constants inspected in `configs/experiment/stage3_formal_8b_2xa100.yaml`, `src/cot_safety/probes/stage3_artifacts.py`, `stage3_formal.py`. `IMPLEMENTATION_SPEC.md`, `FABLE5_FULL_CODE_REVIEW_ROUND1.md`, and the Round-2 request read in full.

**Commands executed (local macOS sandbox, Python 3.12, no PyTorch, no GPU):**

| Command | Result |
|---|---|
| `git status` / `git log` / `git diff 6c1171c aeaf4e5` / `git diff --check aeaf4e5` | clean; HEAD = candidate + request doc only; no whitespace errors |
| `python3 -m pytest tests/ -q -o pythonpath=src --ignore=tests/test_stage4_targeting.py` | **326 passed, 13 skipped, 1 failed** — reproduces the branch claim exactly; sole failure is the pre-existing macOS `/tmp`→`/private/tmp` path-resolution assertion in `tests/test_export_normalized_pairs_for_stage1.py` (Stage1, local-env artifact, not shipped behavior) |
| `python3 -m pytest tests/ -q` (raw, no pythonpath) | 3 collection errors (`ModuleNotFoundError: cot_safety`) — local sandbox artifact, resolved by `-o pythonpath=src` |
| `python3 -m pytest tests/test_stage4_targeting.py --collect-only -o pythonpath=src` | collection error (no local torch) — matches the branch claim; real-Torch hook tests remain pod-mandatory |
| Targeted subsets (5 / 9 / 11 changed+adjacent test modules) | 77 passed/1 skipped; 111/1; 120/1 — all green |
| `python3 -m py_compile` on all 9 changed/new Python modules | OK |
| JSON parse + SHA-256 of the committed model manifest | file's own SHA-256 = `2edaed7855d8bc5274283ff5e73f3852ce1613ba927234664e26afae3308f4e1` — **exactly equals** `CANONICAL_APPROVED_MODEL_MANIFEST_SHA256` pinned in `full_sft_contract.py` |

**Approval-blocked in this sandbox (not executed; must be re-run in the launch environment):** `bash -n` / `sh -n` on the four shell scripts; `PYTHONPATH=`/`env`-prefixed invocations (replaced by `-o pythonpath=src`); anything requiring torch/CUDA/NCCL/bnb — including the real first-step optimizer audit, UVM rehydration, and the 2×A100 preflight.

**Not reproduced:** the branch's "formal targeted suite: 130 passed, 2 skipped" — the file composition is not recorded anywhere on the branch, so I could not reconstruct it; all compositions I ran pass, and the broad-suite figure reproduces exactly, so I treat the broad suite as authoritative.

## 2. Findings

### Blockers

None.

### Major nonblockers

**M-1. The unexpected-file rejection omits `.jinja` chat-template artifacts and top-level directories.** `_is_top_level_loadable_model_file` (`full_sft_runtime.py:55-63`) rejects weight suffixes, any `.json`, `merges.txt`, `vocab.txt`; the scan loop (`full_sft_runtime.py:323-329`) flags symlinks and loadable *files* but silently skips *directories*. Transformers 4.52.4 will load a snapshot-local `chat_template.jinja` (and `additional_chat_templates/`), so a preplanted template passes the "rejects extra top-level loadable files" gate — the Round-2 request's claim is slightly over-broad. Why not a blocker: (a) the threat requires local write access to the snapshot, the same trust domain that could edit the code itself; (b) the SFT trainer never applies a chat template (the dataset is pre-formatted text); (c) the loaded template's hash is recorded (`tokenizer_provenance`, `full_sft_runtime.py:408-413`), is a resume-lineage-compared key (`full_sft_contract.py:107,1960`), and is a required exact SHA in the downstream Stage3/4 binding (`stage2_model_binding.py:114-116`), so drift is visible and consistently bound. Exact fix (recommended before launch, ~3 lines): add `".jinja"` to `_LOADABLE_MODEL_SUFFIXES`, and in the `verify_approved_model_snapshot` scan treat any non-approved top-level *directory* (at minimum `additional_chat_templates`) as unexpected.

### Minor nonblockers

**m-1.** The "130 passed, 2 skipped" targeted-suite composition is unrecorded; record the exact file list in the runbook so the figure is reproducible.

**m-2.** `MODEL`/`TOKENIZER` env force-overrides survive in `model_path()`/`tokenizer_path()` (`run_stage2_sft.py:91-106`) — Round-1 MAJOR-1's original vector — but are now inert: any override must still pass the byte-exact 7-file manifest rehash **before model construction** on both ranks (`trl_train.py:2057` precedes `instantiate_model` at `:2075`), two-rank approval-hash agreement (`trl_train.py:1810-1826`), `FULL_SFT_MODEL_ID` equality, and the six-path instantiated identity audit. Identity is content-addressed; acceptable as-is.

**m-3.** `build_r2_checkpoint_watcher` overwrites `CHECKPOINT_INTEGRITY_STRICT` from config (`run_stage2_sft.py:1343`); triple-guarded fail-closed (`assert_full_sft_contract` requires `strict==1` at `full_sft_contract.py:932`, shell contract requires `"1"`, provenance construction raises at `trl_train.py:2003`), but the failure would be late. Cosmetic.

**m-4.** `bash -n` on the four shell scripts is recorded passing on the branch but was approval-blocked here; re-run on the pod (trivial).

### Fix verification (Round-1 findings)

- **MAJOR-1 — closed.** Pre-instantiation manifest gate: committed 7-file size/SHA-256 manifest, self-hash pinned in code and independently re-verified; per-file rehash, symlink rejection, unexpected-loadable-file rejection (modulo M-1); pre-instantiation call ordering confirmed; cross-rank agreement; instantiated-model audit pins architecture (llama/32/4096/14336/32/8 KV, untied, 291 bf16 tensors), exact base/resized parameter counts (8,030,261,248 / 8,030,269,440), pause token (mode `added_exactly_one` | `preexisting_exact_id`, ID 128256, resized vocab 128257, `encode==[128256]`, must-be-special), `_commit_hash` vs `6a6f4aa…`; gradient audit now pins canonical layer/tensor/param counts (closes the architecture-relative gap).
- **MAJOR-2 — closed (statically).** `audit_first_optimizer_step_state` (`full_sft_contract.py:1231-1808`) runs on every rank at the first real update via `on_optimizer_step`: exact `index2config`/`pid2config` whitelist == only `model.embed_tokens.weight → {"optim_bits": 32}`; single module override; effective `get_config()` recheck (lr/betas/eps/wd/bits/min-size/max_unorm 0.0/skip_zeros False); state1/state2 presence, distinctness, shape/numel, dtype by threshold (uint8 ≥4096 else fp32; fp32 for the sole override), paged + exactly-once `GlobalPageManager` registration + `page_deviceid` for ≥100,000-numel moments; qmap/absmax metadata; exact registered-set equality; state step; `page_mng` identity; `initialized is True`; raw-optimizer identity == preflight; symmetric gather + abort; results written into audits and provenance. Correct against bnb 0.46.1 semantics as pinned. **GPU truth of this audit is exactly what the mandatory on-pod preflight must establish.**
- **MINOR-1** `eval_strategy`/`save_strategy` pinned to `"steps"` (`full_sft_contract.py:350-351`) — closed. **MINOR-2** cold-partial GC — closed and conservative (exact binding grammar+record, ≥3,600s floor / 86,400s default age, owner-PID proven dead twice with inode-identity TOCTOU rechecks, complete partials promoted via `os.replace` + receipt, unknown/ambiguous retained). **MINOR-3** `python3` normalization in both watchers — closed. **MINOR-4** `final/` — declared non-authoritative; downstream authority is the sealed step-1064 manifest (`stage2_model_binding.py`) — closed as resolved.

## 3. Optimizer / embedding-override decision

**`APPROVE_EMBEDDING_FP32_PAGED_OVERRIDE`.** Reasons: (1) it preserves the pinned Transformers 4.52.4 stock behavior for bnb 8-bit optimizers with embeddings — deviating would be a bespoke, unaudited departure from the pinned framework; (2) `model.embed_tokens.weight` contains the freshly initialized pause row, whose Adam moments start at zero and receive sparse, large updates — the single highest quantization-noise-risk state in the model; FP32 moments there is the standard stability guardrail; (3) the ~2.94 GiB/rank cost is arithmetically correct (128,257×4,096 params; 8 B/param FP32 vs ~2.03 B/param blockwise-8-bit for two moments) and both moments remain UVM-paged, so worst-case pressure spills rather than OOMs — ample headroom on 80 GB with batch 1 + gradient checkpointing; (4) the first-step gate enforces the override as an *exact whitelist*, so it cannot silently widen.

**Keeping `paged_adamw_8bit` canonical: approved.** It is pinned (bnb ==0.46.1 in both `pyproject.toml`s, hard startup gate, shell-contract check, provenance validation) and first-step-audited. A non-paged A/B benchmark is optional systems work only; **the optimizer must not change mid-run** — a change would invalidate resume lineage, provenance, and the frozen contract, and the code correctly forbids it.

## 4. Resume / storage / provenance verdict

**PASS.** Versions: 13-key exact record, fail-closed collection including CUDA driver and NCCL; bnb/transformers/TRL are hard startup gates in Python, shell, and provenance validation. Provenance: approved-manifest block (7 files, exact order, revision, manifest digest, `unexpected_top_level_loadable_files == []`), resolved + semantic config hashes, code commit/diff over a 23-file pinned list, exact parameter/tensor counts, gradient + optimizer audits, seed/world-size/batch, data manifests (17,000/500/500 with formal-freeze triple binding), checkpoint lineage with per-file hashes. Resume: full sealed rehash before restore → path-portable semantic lineage projection with stable run-ID/R2-root binding → pinned restore-API observers → per-rank UVM rehydration in ≤16 MiB chunks with SHA-256 before/after and exact final manager-set equality → second full sealed rehash after RNG restore → rank-0 atomic nonce-bound readiness record outside the managed tree, preplant rejected by launcher preflight (`run_stage2_sft.py:189-221`), 32-hex nonce from `secrets.token_hex(16)`. Watcher ordering: neither watcher spawns until readiness validates (`run_logged`: readiness wait at :344 precedes watcher spawn at :354/:371); premature watcher death terminates training (rc 70/71); crash paths preserve terminate→hot-drain→R2-drain ordering. Storage: `/dev/shm` seal → atomic `/workspace` cold with receipts and bound partials → R2 via manifest-bound partial key, `rclone check --download` rehash, receipt object committed last, historical receipts re-verified remotely before any retry may delete — deletion only after destination-bound verification. Capacity preflight is fail-closed (distinct hot/cold filesystems, ≥2 concurrent payloads + terminal export + reserve). DDP error symmetry: `distributed_rank_zero_call` + `_gather_rank_audit` broadcast result-or-error so all ranks raise together. Resume GPU peak: old moments are freed per-moment after each digest-verified swap and UVM buffers are spillable; with weights ~16 GiB + moments ~17.6 GiB and no gradients yet allocated, the transient is well inside 80 GB. Residual risk is exactly the declared one: none of this has run on real CUDA/UVM — pod preflight is mandatory.

## 5. Layer-grid and sample-adequacy verdict

**Layer grid: scientifically consistent with Stage1.** The primary grid `[4,6,7,8,10,12,14,16,17,18,20,21,22,24,25,26,28,30]` is exactly the Stage-1 four-source grid with only terminal readout index 32 removed; index 32 is diagnostic-only. Removing 32 from steering-eligible layers is correct: hidden-state index 32 is the final-layer readout, where "steering" degenerates into a logit-space edit rather than a mid-computation intervention. Enforced in code, not just config: `configs/experiment/stage3_formal_8b_2xa100.yaml:22-23,69-71`; `stage3_artifacts.py:1064` rejects any direction artifact at a diagnostic layer, so 32 cannot reach Stage3 selection, the confirmatory gate, a direction artifact, or Stage4 steering.

**Sealed adequacy: sufficient for the confirmatory paper claim as specified.** Code defaults match the spec exactly (`stage3_formal.py:797-799`): ≥30 eligible SEALED prompts/source and ≥120 total as the confirmatory floor, with ≥10 TRAIN/source strictly a liveness floor that the spec now states is "never the paper test sample size". The frozen ledger supplies 70 sealed candidates/source (280) at 100 draws each within the fixed 40,000-generation budget (`draws_per_prompt: 100`, `expected_scheduled_cells: 40000`), giving realistic slack above the floor; ineligible prompts are retained in coverage reporting; selection is training-only nested four-source LOSO (`final_stage4_layer_uses_sealed_results: false`), and the gate (macro AUROC CI-low > 0.55, ≥3/4 ≥ 0.55, none < 0.50, 10,000 bootstrap over prompt-level units) is withheld — reduced-power/exploratory status only — if the floor is missed. No alternative rule is required.

## 6. Status against the six professor questions

All six remain implemented; `git diff 00c7493..aeaf4e5` touches no Stage3/4 source module (only `tests/test_stage3_artifacts.py`, +135 test lines), so Round-1's file:line evidence stands, and I independently re-verified the load-bearing constants today:

1. **Clean intervention point (A2 vs A3/A4):** implemented — frozen matched arms, arm-invariant rho/direction/CRN, non-inferiority gates, blinded semantic continuity (`stage4_formal.py:44-69`, `stage4_formal_analysis.py:249-381`, `stage4_calibration.py:242-253`). Unchanged since baseline.
2. **Prompt-only / pre-CoT probes:** diagnostics only, structurally excluded from the signal gate (`stage3_diagnostics.py:308-423`). Unchanged.
3. **Honest four-source LOSO:** re-verified — nested training-only LOSO, gate constants confirmed in `stage3_formal.py:791-970`; no six-source claim in the formal path.
4. **SFT-data effect vs steering + nuisance audit:** A0 vs A1 separation plus non-gating `outer_fold_nuisance_diagnostic` (`stage3_diagnostics.py:529-647`). Unchanged.
5. **Teacher-forced/vLLM/HF mismatch:** named and bridged (gated 32-prompt vLLM↔HF bridge; declared online-continuation estimand; `stage4_generation.py:712-838`). Unchanged; backend boundary (vLLM natural rollouts/judging, HF hidden replay/steering) re-confirmed in the formal config.
6. **Absolute residual unsafe rates:** `absolute_residual_summary` with scheduled-cell denominators and reduction-paired residuals (`stage4_formal.py:1168-1362`). Unchanged.

## 7. Final verdict

Both Round-1 MAJORs are correctly and thoroughly closed, all four MINORs are closed, the spec and code agree, the local evidence reproduces, and no new blocker was found. The single code-level gap (M-1, `.jinja`/directory rejection) is a recommended pre-launch hardening within the local trust domain, mitigated by provenance recording and downstream binding. Approval is conditional in the operational sense already mandated by the request and spec — before `trainer.train` proceeds on the pod: run the real-Torch hook tests and full test suite in the launch image, `bash -n` the four shell scripts, and let the built-in fail-closed gates (version pins, manifest rehash, identity audit, first-step gradient + optimizer-state audits, storage preflight) arbitrate on real hardware; any failure there aborts the run by construction.

APPROVE_TO_RUN
