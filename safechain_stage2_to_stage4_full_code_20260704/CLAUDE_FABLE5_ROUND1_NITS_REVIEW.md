# Round 1 Nit-Fix Review: SafeChain Stage2 → Stage4

- Reviewer: Claude Fable 5 (round-1 nit remediation pass)
- Snapshot: packet working tree at `30a5194 "Address Fable Round 1 nits"` (code diff vs `10b9466` = 5 files: trainer, extractor, prep, 8B GPRS config, pipeline; +96/−28 across code)
- Prior reviews: `CLAUDE_FABLE5_FULL_CODE_TREE_REVIEW.md`, `CLAUDE_FABLE5_ROUND1_FIX_REVIEW.md` (PASS with minor nits; open items R1-1..R1-10)
- Date: 2026-07-04
- Method: read the full `30a5194` diff line-by-line, then re-read every changed file in full context plus the files the changes interact with (`run_intra_pause_probe_full.py` shard/control wiring, `tests/test_stage2_pause_kl_trainer.py`, `run_stage4_steering.py` gate, `cli.py` plan writer). Re-executed the pure-Python gates in this environment (no PyYAML / numpy / torch, matching the author's constraints). No code was edited; working tree left git-clean (dry-run artifacts land only in gitignored `runs/`).

---

## 1. Top-Level Verdict

**All four claimed nits (R1-1, R1-2, R1-3, R1-5) are genuinely closed, every claim in the fix request reproduced under execution here, and the nit fixes introduce no blocker, high, or medium issue.** Residual items are two lows and two infos, none launch-relevant. The one pre-launch condition from the prior review — R1-1 before Stage2 GPU launch — is now satisfied.

| Item | Prior severity | Status | One-line evidence |
|---|---|---|---|
| R1-1 diagnostics every microstep + full-vocab softmax, no `no_grad` | medium (fix before GPU launch) | **CLOSED** | Gated to log steps via `_should_log_loss_parts()`, wrapped in `torch.no_grad()`, logsumexp + suppression-pattern chunking; loss path byte-identical (§2.1) |
| R1-2 silent prompt-key fallback, no label check on matched controls | medium | **CLOSED** | Fallback deleted on both build and lookup sides; duplicate IDs raise; label equality enforced with named drop counter; prep asserts unique/present IDs before writing anything (§2.2) |
| R1-3 two-edit 8B unblock undocumented | low | **CLOSED** | Config comment names both fields; loads under the fallback parser; blocked status re-confirmed by execution with and without the env var (§2.3) |
| R1-5 ack flag baked into archival plan | low | **CLOSED** | Flag removed; plans are serialize-only (no executor exists); runner refusal is instructive and the acknowledged path stays reachable (§2.4) |
| Stage2 1.5B kl_transparent_emit | — | **GO, launch-ready** | R1-1 was the only pre-launch condition; dry-run still exports `PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS=50` |
| Stage4 gates (learned-delta refusal, GPRS fail-closed, 8B liveness block) | — | unchanged, re-verified | rc=1 / rc=1 / `blocked_missing_positive_control`; 1.5B liveness still `planned` (no regression) |

**Land it into main `cot-safety`.** Nothing needs to happen before the Stage2 1.5B launch.

---

## 2. Fix Verification (per item)

### 2.1 R1-1 — Stage2 pause-logit diagnostics (`pause_kl_trainer.py`) — CLOSED

The prior finding: diagnostics built on every microstep, materializing an N×V fp32 copy plus a full-vocab softmax over all labeled non-pause positions, autograd-tracked (~10 GB transient/microstep worst case) with `.cpu()` syncs ~800×/run for numbers logged 16×.

All three requested elements are present and correct:

1. **Log-step gating** (`pause_kl_trainer.py:476-484`). Diagnostics are computed only when `_should_log_loss_parts()` (`:250-253`) is true. The predicate `step != self._last_pause_kl_log_step and step % logging_steps == 0` is the exact logical negation of the old early-return in `_maybe_log_loss_parts`, now shared by both call sites. Consistency check: between the diagnostics computation and the log emission inside one `compute_loss` invocation, neither `global_step` nor `_last_pause_kl_log_step` can change, so the two evaluations always agree — diagnostics are never computed-and-dropped nor logged-but-missing. Consumed-once semantics under gradient accumulation are preserved: the first microstep of a logging global step computes + logs, subsequent microsteps of the same step skip (`_last_pause_kl_log_step` set at `:510`). Telemetry values are therefore identical to the old code's logged values (the old code also logged the first microstep's diagnostics; it merely wasted the other microsteps' computation).
2. **`torch.no_grad()`** (`:478`). The entire diagnostics dict construction — logit diagnostics, pause-row norms/cosines, pair counts — is inside `no_grad`. No autograd graph retention; the removed `.detach()` calls are correctly redundant.
3. **No full-vocab materialization.** Pause branch (`:222-230`): boolean-mask select of only the pause slots (a few rows × V, MBs) then `logit − logsumexp` — mathematically identical to `softmax(...)[:, pause_id]` without the N×V softmax output. Non-pause branch (`:231-247`): flat indices + `index_select` in ≤`suppression_chunk_size` (1024) row chunks, exactly mirroring the already-blessed `_pause_losses` suppression pattern (`:402-416`); per-chunk transient is chunk×V fp32 (~0.6 GB at V≈152k), freed each iteration under `no_grad`. Mean/rate computed as chunk-sum ÷ count ≡ the old `.mean()`. The only whole-tensor operation is `shift_logits.reshape(-1, V)`, which may copy in model dtype for batch>1 — the identical, pre-existing operation `_pause_losses` performs **every microstep**; diagnostics add it only on log steps. `.cpu()` syncs now occur only on log steps (~16×/run).

Grep confirms no `F.softmax` remains anywhere in the diagnostics path; the only softmaxes left in the trainer are on pair-selected (≤256/example) logits inside the KL loss, pre-existing loss math.

**Loss/DDP untouched, re-verified:** nothing new contributes to `loss`; the diagnostics block runs after `loss` is formed and every output ends in a Python float; no collectives, no RNG. `test_compute_loss_is_finite_on_batch_with_and_without_pauses` (`tests/test_stage2_pause_kl_trainer.py:252-287`) still passes by inspection — with the test's `global_step=1, logging_steps=100` the diagnostics branch is skipped entirely and the grad assertions are unaffected.

Residual (new, minor — see §3): no test exercises the gated-on path (N-1); `_zero(logits)` full-reduction idiom now also runs in diagnostics (N-3, info).

### 2.2 R1-2 — Stage3 matched-control lookup + prep ID discipline — CLOSED

The prior finding: ID-lookup miss silently fell back to prompt-key matching (wrong-trajectory, possibly wrong-label controls on stale/hand-made inputs), no label-equality check, duplicate-ID last-wins/first-wins hazards.

Extractor (`extract_hidden_states.py`):

- **Prompt fallback deleted on both sides.** `build_matched_lookup` (`:169-178`) stores only `id:` keys and **raises** `ValueError` on a duplicate ID; `find_matched_row` (`:181-189`) returns the ID hit or `None`. The `prompt:` key path no longer exists (grep: `prompt_key` in this file now appears only in metadata/manifest emission at `:872,:915`, not in matching).
- **Label equality enforced** (`:691-694`). The matched row's label is derived by the *same* `label_from_row(…, args.label_field)` used for the pause row at `:637`; any mismatch drops the row with a dedicated `matched_control_label_mismatch` counter, manifest-visible. Fail-closed edges: a control missing the label field resolves to −1/`missing_label_field` (explicit field) or −1/`unlabeled` (default fields) → mismatch against any labeled pause row → dropped. Rows reaching the check are only label ≥ 0 or −1-with-`--allow_unlabeled` (filters at `:638-646`), and an unlabeled-pause/unlabeled-control pair passing equality is coherent (both −1, carried as unlabeled).
- **Id-less rows now fail loud, not wrong.** `row_id()`'s fallback hash (`:161-166`) includes the pause-bearing output and the row index, so cotpause/nopause hash IDs can never coincide — with the fallback gone, hand-made files without explicit IDs drop 100% of rows under `missing_matched_control_row` instead of silently pairing across trajectories. That is the intended failure direction; the wired path (prep-written shared IDs) is unaffected.

Prep (`prepare_intra_pause_probe_data.py`):

- **`map_rows_by_unique_id`** (`:514-527`) raises on a missing ID or any duplicate (with a sorted 10-sample) across all no-pause rows (trainable+heldout+partial), and is called at `:649` **before any output file is written** — a duplicate aborts the run with nothing on disk, so no partially-consistent data dir can exist.
- **Pause/no-pause lockstep** (`rewrite_rows`, `:612-644`): a row is appended to `rewritten` only if both the pause and no-pause builders succeed, both with the same `stable_probe_id`. Hence (a) the split-reference indexing `no_pause_by_id[row["id"]]` at `:678,:687,:693` cannot KeyError on the wired path, and (b) uniqueness of no-pause IDs implies uniqueness of pause-side IDs — the prior R1-8 hazard (distinct rows sharing an upstream `id` silently cross-matched) is now a hard stop in prep, with the extractor's duplicate raise as an independent second line of defense.
- **`stable_probe_id` fingerprint** (`:314-318`, source‖prompt‖reasoning‖final): distinct trajectories for the same prompt get distinct fallback IDs — necessary because `--dedupe_strategy` defaults to `none` ("keeps multiple trajectories for the same prompt", `:545-550`), which is precisely the corpus shape that made the old prompt fallback dangerous. Note this function is Round-1 code, unchanged in `30a5194`; the fix request bundles it as if new (§3, N-4 — accuracy note only, the property is verified present).

Interaction with the shard path (`run_intra_pause_probe_full.py:441-474`, unchanged but re-checked against the new semantics): shard buckets are ID-joined, keeping cotpause shard *k* and nopause shard *k* aligned; a missing nopause file for a shard makes the extractor `SystemExit` (`extract_hidden_states.py:598-599`); a pause row absent from the control map simply produces no control row for its bucket and drops at extraction with a named counter. All failure modes are fail-stop or counted drops. The Stage3 dry-run executed here (§4) emits `nopause_shards/train/train.shard{0,1}.json` for train shards and `nopause/{val,test,source_heldout_reasoningshield_test}.json` otherwise — exactly as the request claims.

### 2.3 R1-3 — 8B liveness two-edit documentation — CLOSED

`stage4_pause_gprs_8b_4xa100.yaml:17-19` now says: *"Keep this missing status until the env path above is verified to be a genuine 8B full-SFT pause control; both fields must be changed to unblock the 8B liveness battery."* That is precisely the previously undocumented behavior. Executed here: the config loads under the no-PyYAML fallback parser with the mid-mapping comment (values resolve to `positive_control_model: ''`, `positive_control_status: missing_required_full_sft_pause_control`); the 8B liveness dry-run writes `status: blocked_missing_positive_control` **both** with `STAGE4_8B_FULL_SFT_CONTROL` unset and set; the 1.5B config still plans (`planned`) — no regression.

### 2.4 R1-5 — ack flag removed from archival plan — CLOSED

`pipeline.py` `generate_and_judge` command is now `run_stage4_steering.py --config <config> --phase eval` with no `--allow_learned_delta` (verified by executing `plan_for_config` on the learned-delta config: no step contains the flag). Plans remain serialize-only: `cli.py:69` converts steps `to_dict`; `run_full_pipeline.py` is a thin `config show` wrapper; grep finds no executor of plan commands anywhere, and the `<config>` placeholder makes them non-runnable as-is. A reader copy-pasting the plan command now hits the instructive refusal (executed: rc=1, message names both `--allow_learned_delta` and `steering.acknowledge_deprecated: true`), and the acknowledged archival path stays reachable (executed: rc=0 dry-run with the flag). Repo-wide grep confirms the only remaining occurrences of the ack are the argparse definition, the gate itself, the `ALLOW_LEARNED_DELTA=true`-guarded shell, and the default-`false` config key.

Not claimed and correctly untouched: R1-4 (invariant snapshots held for the run — acknowledged host-RAM cost), R1-6 (controls fixed to offsets 3/4), R1-7 (re-prep required; extraction fails fast on old dirs), R1-9 (control-forward peak memory), R1-10 (no DDP interaction). All remain as previously assessed; R1-8 is substantially improved as a side effect of the prep assertion (above).

---

## 3. New Findings Introduced by the Nit Fixes

Severity scale as before: blocker / high / medium / low / info. **No blocker, high, or medium.**

- **N-1 (low, test coverage).** With the gating in place, the trainer tests' `bare_trainer` (`global_step=1, logging_steps=100`) means no test executes `_pause_logit_diagnostics`'s new chunked/logsumexp math or the `_should_log_loss_parts` true-path. The formula itself is pinned indirectly — `test_pause_losses_match_manual_shifted_ce_and_suppression` asserts the identical `logit − logsumexp` pattern in the loss — and diagnostics are non-evidence-bearing telemetry, so this is cheap-insurance territory: one test at `global_step == logging_steps` capturing the `trainer.log` payload and checking `pause_emit/*` against a manual softmax would lock the refactor down.
- **N-2 (low, operational).** `map_rows_by_unique_id` makes exact-content duplicate raw rows (same source‖prompt‖reasoning‖final, no explicit `id`) a **fatal** prep error where they were previously silent last-wins. Fail-stop is the right direction, but the built-in remediation is coarse: `--dedupe_strategy prompt` also collapses genuinely distinct same-prompt trajectories. If the real corpora contain exact dupes, re-prep will hard-stop until they are removed upstream. Consider keep-first (or drop-with-counter) for *identical-fingerprint* duplicates only, while keeping the raise for same-ID-different-content.
- **N-3 (info, perf).** `_zero(logits)` builds its zero scalar via `logits.sum() * 0.0` — a full reduction over the B×T×V logits tensor, now invoked twice per log step in diagnostics. The idiom is inherited from the loss path (where the graph connection is load-bearing); under `no_grad` it is pointless but harmless at ~16 log events/run. Not worth changing on its own.
- **N-4 (info, request accuracy).** The fix request lists "Fallback stable IDs include source, prompt, reasoning, and final answer" under this round's prep changes; that property is Round-1 code (`stable_probe_id`, unchanged in `30a5194`). Verified true in the tree — noted only so the change log stays exact. Every other claim in the request matched the diff one-for-one.

---

## 4. What Was Executed vs Inspected

Executed in this environment (no PyYAML — the fallback mini-YAML parser path itself was exercised; no numpy/torch, so `smoke_test.py` proper cannot run and the torch tests are inspection-only, same constraints as prior rounds):

| Gate | Result |
|---|---|
| `python3 -m py_compile` over the four changed Python files | OK |
| `load_config` on all 9 KL/GPRS/steering configs incl. the commented 8B GPRS | all load; 8B control fields resolve correctly |
| 8B liveness `--dry_run`, env unset / env set | rc=0 / rc=0; `blocked_missing_positive_control` both times |
| 1.5B liveness `--dry_run` | rc=0; `planned` (regression check) |
| learned-delta `--phase eval --dry_run` (no ack) | rc=1, instructive refusal |
| learned-delta `--phase eval --dry_run --allow_learned_delta` | rc=0 (archival path reachable) |
| GPRS 8B `--phase eval --dry_run` | rc=1, fail-closed |
| bare `run_stage4_steering.py --dry_run` | rc=0, validate scope |
| Stage2 1.5B `--dry_run` | rc=0; `PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS=50` exported |
| Stage3 legacy `--dry_run --extract_train_shards 2` | rc=0; `nopause_shards/.../train.shard{0,1}.json` for train, `nopause/{split}.json` for val/test/heldout |
| `plan_for_config` on learned-delta config | no `--allow_learned_delta` in any step |
| repo-wide grep for ack flags / prompt-fallback remnants | clean |
| `git status` after all runs | clean (artifacts only under gitignored `runs/`) |

Verified by inspection only (torch-dependent): the diagnostics math equivalence (logsumexp ≡ softmax-column; chunk-sum÷count ≡ mean), no_grad scoping, gradient-accumulation log semantics, test-suite compatibility, and the shard/control ID-join behavior on real data. GPU memory behavior is no longer a concern by construction (the offending allocations are gone from the per-microstep path).

---

## 5. Direct Answers to the Three Review Questions

1. **Are R1-1, R1-2, R1-3, and R1-5 closed?** Yes, all four — each verified against the actual diff and, where the environment allows, by execution. R1-1's closure also discharges the one pre-launch condition attached to the prior PASS.
2. **Is the code clean enough to land into main `cot-safety` before Stage2 1.5B launch?** Yes. The loss path is untouched (diagnostics are no-grad, log-step-gated, read-only), all Stage3/Stage4 failure modes introduced or touched by this diff are fail-stop or counted fail-closed drops, the gates re-verify identically, and the tree is internally consistent (tests, shards, configs, plans). Nothing remains that should block landing or the 1.5B GPU launch.
3. **Any blocker/high/medium issues introduced by these nit fixes?** No. Two lows (N-1 test coverage for the gated diagnostics path; N-2 fatal-duplicate prep behavior lacking a content-level dedupe escape hatch) and two infos (N-3, N-4). Both lows are next-pass material, not landing conditions.

---

## 6. Remaining TODO (carried, unchanged — not landing blockers)

Before **trusting Stage3 numbers**: re-run prep so `nopause/` exists everywhere (R1-7; old dirs fail fast); report pause-AUROC side-by-side with prompt-baseline AUROC; implement `within_prompt_auroc` (on-policy sampling + CoT-segment judge) before any confirmatory claim; document legacy-vs-src `cot_k` coordinate divergence (S3-3).

Before **any Stage4 evidence**: implement the liveness kernels behind the plan schema and wire `liveness_decision`+`status` into the GPRS unlock; train a genuine 8B full-SFT positive control, then flip both `STAGE4_8B_FULL_SFT_CONTROL` and `positive_control_status` (now documented in-config); implement GPRS artifact producers + existence checks (`mean_diff_direction.pt`, `safe_centroid.pt`, `probe.pt` still have no writers); apply `gate_threshold` (validated-but-unused); judge endpoint separation per S4-5 — still the largest single open item in the repo.

Small carries: `num_items_in_batch` note (S2-3); checkpoint-selection rule vs the checkpoint-400 default (B7); NOT-IMPLEMENTED comments on declared-not-consumed `stage4_pause_gprs.yaml` keys (F5 residue); new N-1/N-2 from this round.

---

## 7. Headline Verdict

**PASS to land** — merge into the main `cot-safety` repo and proceed to the Stage2 1.5B launch. All four nits are closed as claimed, the fixes introduce no blocker/high/medium issue, and the previously attached pre-launch condition (R1-1) is discharged. Queue N-1 and N-2 as low-priority items for the next pass.

— End of review —
