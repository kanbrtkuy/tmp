# Fable Code Review — SafeChain Stage2→Stage4 Framework Commit

Date: 2026-07-04. Reviewer: Claude (claude-fable-5). Read-only; no code or config edited.

Scope: commit `88559e0` ("Add KL-transparent Stage2 to Stage4 framework") vs base
`c372bab`, in `/Users/baby/Documents/SafeChain/cot-safety`. Dirty-worktree Stage1
files ignored as instructed. Spec basis: the five prior Fable review artifacts in
this packet, primarily `CLAUDE_FABLE5_STAGE2_TO_STAGE4_FLOW_REVIEW.md` (which
re-verified the earlier three against packet source). Blocker/TODO IDs (B1–B7,
T1–T16, NEW-F1, NEW-B2) carry their prior meanings.

## How this review was verified (not just read)

- Full `git diff c372bab..88559e0` read file-by-file (26 files, +1753/−13).
- **Byte-diffed the committed Stage2 launch chain against the packet artifacts
  the flow review verified line-by-line**: `pause_kl_trainer.py`,
  `tests/test_stage2_pause_kl_trainer.py`, `run_4gpu_intra_pause_sft.sh`,
  `run_stage2_sft.py`, and the 1.5B KL config are **byte-identical** to the
  reviewed packet versions; the 8B KL config differs from the packet by
  **exactly the two NEW-F1 lines** and nothing else. Trainer anchors
  re-confirmed in the committed file (weight-decay assert `pause_kl_trainer.py:143–148`,
  chunked suppression logsumexp `:313–326`).
- Executed `load_config` on all five new experiment configs (env interpolation
  `${VAR:-default}` and deep-merge verified by running, not by reading).
- Ran the 12 torch-free tests locally: `tests/test_steering_scope.py` +
  `tests/test_stage4_gprs_liveness.py` → **12 passed** (with `PYTHONPATH=src`;
  the suite assumes the installed src-layout package, same as before — the
  7 trainer tests need torch and remain pod-gated).
- Ran `cot-safety steer validate-scope --config stage4_pause_gprs.yaml` and
  `run_stage4_liveness.py --dry_run` (output to /tmp) successfully.
- Hand-verified `projection_rejection_update` numerically (rejection,
  norm-cap, safe-side no-op cases; no torch available locally).
- Checked imports for all new helper code in the modified legacy scripts
  (`re`, `Any`, `defaultdict` all present).

---

## 0. Executive Verdict

**GO for Stage2 1.5B launch. GO for keeping this commit on main.** The launch
path is bit-frozen to the Round-3-verified artifacts, NEW-F1 is applied
surgically, the T2 prompt-baseline plumbing is correct end-to-end, and the
Stage4 scaffolding is honest — it hard-fails instead of pretending to run.
Stage2.5 is correctly absent from the training path.

Three findings should be fixed in an immediate follow-up commit (all are
one-to-few-line edits, none blocks the 1.5B launch): **F1** the 8B Stage3/Stage4
default checkpoint paths point at a directory the 8B Stage2 config will never
create (`_4xa100` suffix mismatch); **F2** the liveness runner and pipeline
planner text encode the *wrong yellow semantics* ("yellow/red ⇒ Stage2.5"),
regressing the flow review's amendment #1 (yellow proceeds on live layers; only
red branches); **F3** the 1.5B GPRS `probe_checkpoint` points at the *old*
non-KL Stage3 run directory — the same provenance-trap class as T14.

Stage3 and Stage4 *runs* remain NO-GO on the carried gates (battery, T1, B7,
kernels, hooks) — as this framework commit itself correctly enforces.

---

## 1. Findings

Severity: **H** = fix before the affected milestone can be trusted;
**M** = fix in the next commit (cheap, trap-class); **L** = polish.

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| F1 | **M** (H before any 8B Stage3/4) | `configs/experiment/stage3_intra_pause_probe_kl_transparent_8b_cot4_4xa100.yaml:9`, `stage4_pause_gprs_8b_4xa100.yaml:11` | Default `STAGE2_KL_CHECKPOINT` fallback is `/workspace/outputs/deepseek_8b_intra_pause_cot4_kl_transparent_emit_trusted_cot_18k_save50_max400_4xa100/checkpoint-400`, but the 8B Stage2 config writes to `..._save50_max400` (no `_4xa100`) — verified by loading both configs (`stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml:6`). The default silently references a path that will never exist. The 8B *eval* config uses the correct un-suffixed path, confirming which side is wrong. | Drop `_4xa100` from the two defaults (or add it to the Stage2 `run.output_dir`; pick one and keep the eval config consistent). |
| F2 | **M** | `scripts/run_stage4_liveness.py:52–55` (`next_step` string), `src/cot_safety/pipeline.py` liveness-step `notes` | Both say "yellow/red ⇒ Stage2.5 branch". The agreed decision table (flow review §0 amendment 1, §7a) is: **yellow proceeds restricted to live layers, Stage2.5-A merely queued for the next train; only red stops Stage4**. No branching logic exists yet, so this is text-only — but this text is the plan-of-record written into every `liveness_plan.json` and every pipeline printout, and it re-creates the exact over-trigger the flow review corrected. The request's own intended sequencing (items 5–6) states the correct semantics. | Reword both strings: "green ⇒ fixed Stage3 then GPRS; yellow ⇒ proceed on live layers only, queue Stage2.5-A for next Stage2 train; red (positive control green) ⇒ stop Stage4, Stage2.5-A/B branch." |
| F3 | **M** | `configs/experiment/stage4_pause_gprs.yaml:47` | 1.5B `steering.gprs.probe_checkpoint: .../runs/stage3_intra_pause_probe_deepseek_1p5b/probe.pt` — the **old full-SFT Stage3 run name**, not the new `stage3_intra_pause_probe_kl_transparent_deepseek_1p5b_cot3` run this chain produces. The 8B variant (`stage4_pause_gprs_8b_4xa100.yaml:27`) correctly points at its KL Stage3 run, proving the 1.5B line is stale. A probe gate trained on the old SFT checkpoint's activations applied to the KL checkpoint is precisely the T14 provenance-break class. Additionally, no current Stage3 script writes a `probe.pt` at run root — the path is a placeholder until T7 lands (acceptable, but the run dir should at least be the right one). | Repoint to `runs/stage3_intra_pause_probe_kl_transparent_deepseek_1p5b_cot3/...`. |
| F4 | **M** (scoped to 8B battery) | `configs/experiment/stage4_pause_gprs_8b_4xa100.yaml:15` | 8B liveness `positive_control_model` is `deepseek_8b_intra_pause_cot4_format_only_.../checkpoint-250` — a **rows-only (format-only) checkpoint, i.e. the same training class as the model under test**. A positive control must be a known-live anchor (the 1.5B config correctly uses the full-SFT cot3 ckpt250). If the format-only 8B ports are themselves dead, `require_positive_control_green` fails and the battery is uninterpretable by construction. | Before the 8B battery: locate/train a full-SFT 8B control, or pre-register the substitution and its interpretive limits. Not urgent — 8B is gated behind the full 1.5B chain anyway. |
| F5 | **L** | `configs/experiment/stage4_pause_gprs.yaml:51–89` (`eval:` block), `:48` (`random_direction_control`) | Aspirational/unconsumed keys: `eval.model_conditions: [base, sft, sft_gprs, sft_random_direction]` is a list of *strings* — not even the dict shape `run_model_comparison_eval.py:160` consumes; `pause_modes`, `report.*`, and `steering.gprs.random_direction_control` are read by nothing. This is the "advertised, unconsumed" pathology the main review dinged, though materially defanged here because GPRS generation/judge/eval is hard-blocked (see §3). `steering.pause_mode`/`insert_pause_after_cot_tokens` are likewise not yet plumbed to the legacy eval shell (T12 still open; `run_intra_pause_full_steering_eval.sh:297` hardcode still stands). | Mark the block `# planned — consumed when T5/T10/T12 land` or trim it; wire `random_direction_control` into `validate_gprs_config` output when the hook lands. |
| F6 | **L** | `tests/test_stage4_gprs_liveness.py` | No test exercises `projection_rejection_update` (the file is deliberately torch-free). I hand-verified the math (see §4), but the pod suite has torch — the one function that will touch hidden states in the decode loop deserves the three obvious cases (rejection amount, norm-cap clamp, safe-side no-op). | Add a torch-marked test on the pod suite. |
| F7 | **L** | `src/cot_safety/steering/liveness.py:56–58` | `liveness_decision` passes any explicit `report["decision"]` through unvalidated (`"GREEN"`, `"gren"` propagate verbatim to downstream string comparisons). | Normalize/validate against `{green, yellow, red, not_run}`; raise or return `"unknown"` otherwise. |
| F8 | **L** | `src/cot_safety/cli.py:33–37`; `scripts/run_stage4_steering.py:build_env` | `_target_specs_from_config` raises bare `KeyError` (not `ValueError`) if a dict spec lacks `name`; duplicate spec names are not rejected; and a `TARGET_SPECS` env override bypasses CLI validation entirely (`env.setdefault` keeps the env value; validate-scope only reads the config). The last is a pre-existing T16 residue, now narrower. | Cheap hardening when convenient. |
| F9 | **L** | `scripts/smoke_test.py:18–35` | The config load-check list was not extended with the five new experiment configs. I load-checked all five manually (they pass), but the smoke test is where that should live. | Append them to the list. |
| F10 | **L** | `scripts/run_stage2_sft.py:352–354`, `run_logged` allowlist | `PAUSE_KL_PAUSE_TOKEN` is exported but (a) not in the logged-env allowlist and (b) not forwarded to Hydra by the shell (that *is* the B1 fix). Decorative, as the flow review already noted — the trainer's built-in `<|pause|>` default matches. | Leave; known. |
| F11 | **L** | `src/cot_safety/pipeline.py` (`build_gprs_artifacts` step) | The "build GPRS artifacts" plan step's command is `run_stage4_steering.py --phase validate` — the notes honestly say "Placeholder", but the command itself does validation, not artifact building. | Swap in the real T7 command when it exists. |
| F12 | **L** | `configs/experiment/stage4_pause_gprs.yaml:4` | The 1.5B GPRS config includes `../runtime/a100_4x.yaml` while the rest of the 1.5B chain is the A6000 line. Runtime block is env-level (hf_home/devices), so harmless, but inconsistent. | Cosmetic. |

Two measurement caveats worth recording, not defects: (a) `pause_spans`
(`run_model_comparison_generation.py:59–74`) counts only *contiguous* pause-token
runs — a natural emission of `<|pause|>\n<|pause|>` counts as two runs, which
slightly deflates `has_single_pause_run_of_3`-style stats; (b) the
`skip_special_tokens=False` fix was applied to the vLLM path only — the HF
(`transformers`) generation path would still hide naturals. Both eval configs
pin `generation_backend: vllm`, so this is latent, not live.

---

## 2. Q1 — Stage2 `pause_kl` blockers for the 1.5B launch?

**None found.** The critical property of this commit is that it did not write
new Stage2 code — it *copied the already-triple-reviewed artifacts verbatim*:

- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`,
  `tests/test_stage2_pause_kl_trainer.py`,
  `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`, and
  `scripts/run_stage2_sft.py` are **byte-identical** (`diff -q`) to the packet
  versions the flow review verified in full. B1 (no `pause_token` Hydra
  override — the shell's PAUSE_KL block passes only bare scalars,
  `run_4gpu_intra_pause_sft.sh:103–119`), B3/C2 (weight-decay assert + rows-only
  invariant callback), C3/P1 (single `.cpu().tolist()`), C4/P2 (chunked
  suppression logsumexp, re-confirmed at `:313–326`) all therefore remain landed.
- The 1.5B KL config is byte-identical to the packet version: `load_best: false`,
  `early_stopping.enabled: false`, save 25 / eval 50 / limit 20,
  `weight_decay: 0.0`, `pre_weight: 0.0` explicit.
- Env plumbing verified at both ends: `run_stage2_sft.py:340–372` maps
  `method: kl_transparent_emit` → `PAUSE_KL_ENABLED=true` (and forces
  `FORMAT_ONLY=true`, which the rows-only recipe requires), exports every
  `PAUSE_KL_*` knob the shell consumes; the shell forwards them as
  `+trainer.pause_kl.*` plus `trainer._target_=src.utils.pause_kl_trainer.PauseKLSFTTrainer`.
  Config-explicit values (e.g. `pre_weight: 0.0`) win over the 0.1 shell default
  because the runner always exports.
- Config mechanics verified by execution: deep merge (`config.py:50–61`) keeps
  parent keys under partial `sft:` overrides — the merged 1.5B config resolves
  to `method=kl_transparent_emit, cot_offset=3, max_steps=400, save=25`.
- Stage2.5 is correctly absent: no near-pause bucket, no hinge, nothing merged
  into the default path.

The launch gate is unchanged and unmet by anything I can run here: **pod pytest
(the 7 torch trainer tests have still never executed anywhere) → single-GPU
smoke → 4-GPU smoke.** I executed the 12 torch-free tests (all pass); the torch
suite is the pod's job.

## 3. Q2 — NEW-F1 on 8B?

**Yes, exactly and minimally.** The committed 8B KL config differs from the
packet version by precisely two lines: `load_best_model_at_end: true → false`
(:19) and `early_stopping.enabled: true → false` (:23). Verified by direct diff
against the packet file, and by loading the merged config over its
`format_only_8b_cot4` parent (which still carries `load_best: true`/ES-on — the
child override wins under deep merge; merged result confirmed
`load_best=False, es=False, cot_offset=4, intra_dir=intra_pause_cot4`). The
vestigial `patience: 2` is inert with `enabled: false`. `final/` now means
last-step on both lines — the checkpoint-selection trap is closed.

## 4. Q3 — Stage3 prompt-baseline plumbing end-to-end?

**Correct, verified at every hop:**

1. `configs/experiment/stage3_intra_pause_probe.yaml:13` defines
   `hidden.positions.prompt_baselines: [last_prompt_token, pre_think]` (8B KL
   variant repeats it).
2. `scripts/run_stage3_intra_pause_probe.py:95–96` folds `prompt_baselines`
   into `--positions` (so probes get fitted at those positions) and
   `:101–109/:219–220` independently emits `--prompt_positions` from the same
   list — the two flags are consistent by construction.
3. `run_intra_pause_probe_full.py:187–194` accepts `--prompt_positions` and
   `extraction_cmd` (`:405–406`) forwards it to `extract_hidden_states.py`.
4. `extract_hidden_states.py:486` allows exactly
   `{last_prompt_token, assistant_start, assistant_last, pre_think}` (config
   values covered), `:587` appends them to `position_names` so they land in the
   NPZ under the same names the probe scan will look up, `:824` records them in
   the manifest. Row-level guards (`:247`, `:300`) drop the position when the
   anchor isn't found, matching the existing valid-mask semantics.

One footgun, not a bug: a prompt-position name added to `diagnostics` but not
to `prompt_baselines` would reach `--positions` without being extracted → probe
fitting fails late on a missing NPZ key. The current configs cannot hit it.

Also correct: the base Stage3 config now carries the honest annotation
`probe.true_content_controls.current_control_cot_aliases_valid: false`
(:29–31). Note this is an *annotation*, not the fix — `control_cot_3/4` are
still aliased to `post_pause_1/2` in the extractor and still listed in the
diagnostics of both new KL configs. T1 remains an open pre-extraction gate.

## 5. Q4 — Are the new Stage3 KL configs safe once the checkpoint exists?

**1.5B: yes, mechanically.** Paths are self-consistent (training output dir ==
Stage3 default prefix), `STAGE2_KL_CHECKPOINT` env override supported,
cot-offsets `[3,4,7,8]` inherited (depth caveat covered), fresh
`legacy.*` dirs prevent NPZ collisions with old runs, and the on-policy block
is present but `enabled: false` (fields-now-data-later, as specced).

**8B: no — F1 first.** The default checkpoint path has the spurious `_4xa100`
suffix and will not exist (§1). Also `hidden.positions.diagnostics` correctly
switches to `cot_4, cot_5, cot_8, cot_9` with `cot_offsets: [4,5,8,9]` — good
cot4-line consistency.

**Both:** "safe to use" ≠ "cleared to run." The run gates carried from the flow
review are untouched by this commit and still open: battery green/yellow on the
*chosen* checkpoint (defaults pin `checkpoint-400`; B7 repoint after the eval
battery, never trusting last-step), T1 de-aliased controls landed, pod pytest
green. The commit does not create any pressure to violate that ordering.

## 6. Q5 — Honest scaffolding or "looks implemented"?

**Honest, with one text regression and one pocket of aspiration.** The three
mechanisms that make it honest, verified by execution:

- `run_stage4_liveness.py` without `--dry_run` writes the plan with
  `status: not_run` and then **raises SystemExit** with an explicit "GPU
  liveness metrics are not implemented" message (`:57–62`). `--dry_run` writes
  `status: planned`. Nothing pretends to produce metrics.
- `run_stage4_steering.py:277–281` **hard-blocks** every generation-side phase
  (`generation`/`judge`/`summary`/`eval`/`all`) for `method: gprs|projection`
  with an explicit "not wired into the legacy generation shell yet" error. Only
  `validate` and `liveness` are reachable. The legacy `learned_delta` path is
  untouched (its plan note now correctly demotes it to "baseline/control only").
- The pipeline plan labels `build_gprs_artifacts` a placeholder in its notes.

The regression: the honest text encodes the *wrong decision rule* for yellow
(F2). The aspiration pocket: the `eval:` block and a few steering keys nobody
consumes (F5) — materially harmless while eval is hard-blocked, but exactly the
key-advertising habit prior reviews flagged, so trim or mark it.

Battery plan fidelity vs the T13 spec: layers supersets of {7,14,21}/{8,16,20,24} ✓,
ε∈{1,2,4}·σ ✓, next-16-token window ✓, injection-gain + KV-ablation plus
attention-mass and patching ✓, gate thresholds 25%-of-content and 5×-BOS match
§7a ✓, `require_positive_control_green` ✓. Missing relative to spec:
`calibrate_on: positive_control` (thresholds are fixed numbers, not
control-calibrated), per-seed random directions are a single `random` entry,
and `layer_source: liveness_report` does not exist — `steering.layer: 14` is
pinned (fine as a placeholder; must be sourced from the battery's live-layer
set before the micro-pilot, per §8 below).

## 7. Q6 — Bugs in steering phases, GPRS validation, scope validation?

- **`projection_rejection_update` (`gprs.py:36–61`): correct.** Implements
  `h ← h − λ·((h−μ_safe)·û)₊·û` with cap ‖δ‖ ≤ ρ‖h‖. Hand-checked: h=[3,4],
  u∝[0,1], μ=[0,1] → [3,1] (rejects exactly the positive component); with
  ρ=0.1 → [3,3.5] (δ clamped to 0.5=0.1·‖h‖); safe-side h=[3,0.5] unchanged
  (clamp at 0). Direction is re-normalized defensively; ε-guards on both norms.
  Not covered by any test (F6).
- **`validate_gprs_config`: correct and correctly strict.** `learned_delta`
  passes through untouched; `gprs|projection` requires the three artifact keys,
  `norm_cap > 0`, `gate_threshold ∈ [0,1]`. It intentionally does not check
  file existence (the artifacts cannot exist yet). It runs unconditionally in
  `main()` — desirable: a malformed GPRS config fails even on `--phase liveness`.
- **Phase ordering: no bugs found.** `validate` → prints gprs meta and returns;
  `liveness` → subprocess to the liveness runner (cwd/env correct); everything
  else blocked for gprs. `--phase all` with gprs runs validate then stops with
  the explicit error — acceptable.
- **`target_specs` scope validation: correct.** `validate_target_specs` accepts
  both the newline string format and lists; every group's positions go through
  `validate_no_pre_post_or_cot_targets`, which (via `validate_pause_only_targets`,
  `scope.py:6–16`) also **rejects empty groups** ("At least one steering target
  position is required") — so `"name|"` cannot silently steer nothing. The env
  producer (`run_stage4_steering.py:target_specs`) emits the same
  `name|p1,p2` format from the same config node the CLI validates, closing the
  T16 config-path gap. Residues: env-var override bypass and KeyError polish
  (F8). Functional check on the shipped config passes and prints validated
  groups.
- **`liveness_decision`:** precedence red > yellow > all-green > unknown is
  right; mixed green+unknown safely degrades to "unknown"; explicit-decision
  passthrough unvalidated (F7).

## 8. Q7 / Q8 — What must be fixed now vs what waits

**Fix now (next commit; none requires GPU or blocks the 1.5B launch):**

1. F1 — two default-path one-liners (8B chain).
2. F2 — two strings (yellow semantics). Cheapest possible fix for a
   decision-table regression that would otherwise be copied into the kernel
   implementation.
3. F3 — one path (1.5B probe provenance).
4. (Recommended, minutes each): F9 smoke-test list, F7 decision normalization,
   F8 KeyError→ValueError.

**Waits until after the Stage2 checkpoint / next implementation pass:**

- T13 GPU kernels behind `run_stage4_liveness.py` (the plan schema is in place).
- T5/T6 GPRS hook wiring + persistent pause-ordinal counter +
  `--steer_natural_pauses`; random-direction control arm consumption (F5).
- T12 offset/mode env plumbing into the eval shell (the `:297` hardcode
  stands; currently unreachable for GPRS, formally 8B-blocking).
- T1 de-alias + pause-free-forward control (pre-Stage3-extraction gate).
- T8/T9/T10 judge/summarizer integrity; on-policy labels + within-prompt AUROC
  analysis (config fields exist, `enabled: false`).
- `layer_source: liveness_report` + `gate.threshold_source` (probe-FPR) swap;
  `calibrate_on: positive_control` in the liveness gate block.
- F4 8B positive-control decision; F6 torch test for the GPRS update.
- B7 checkpoint repoint everywhere after the eval battery chooses.

## 9. Q9 — Verdicts

| Question | Verdict |
|---|---|
| **GO for Stage2 1.5B launch?** | **GO.** Launch chain byte-identical to the triple-reviewed artifacts; config byte-identical to the reviewed pattern; env plumbing verified at both ends. Gates unchanged: pod pytest (torch suite still never executed) → single-GPU smoke → 4-GPU smoke. Nothing in this commit adds risk to, or is allowed to delay, the launch. |
| **GO for keeping this commit on main?** | **GO.** It is what it claims to be: an honest framework commit. Apply F1–F3 as an immediate follow-up commit (no history rewrite; they are default-path/text traps, not behavior). |
| **NO-GO items before the Stage3 run** | Battery green/yellow on the battery-chosen checkpoint (kernels not yet implemented — T13); T1 de-aliased controls landed (this commit only annotates the aliasing); B7 repoint of `STAGE2_KL_CHECKPOINT` away from the `checkpoint-400` default; pod pytest green. For the 8B line additionally F1. T2 is **done** as of this commit. |
| **NO-GO items before the Stage4 run** | T13 kernels + battery green/yellow (yellow ⇒ live-layer restriction — after F2's wording fix); T7 direction artifacts at the F3-corrected path with QC gates; T5/T6 GPRS hook + random-direction norm-matched control; T12 + T8/T9/T10 eval integrity; steering layer sourced from the liveness report instead of the pinned `layer: 14`; micro-pilot before any 3-seed pilot. 8B additionally: F1, F4, own battery, T12/T14 (T14's fix is inherited here only if the GPRS 8B config, not the old `stage4_pause_steering_8b` one, is what actually runs). |

## 10. Bottom Line

This commit does the two things the flow review demanded of the framework PR
and avoids the one thing it forbade: the measurement scaffolding now exists in
the tree (battery plan, gprs/liveness/scope helpers, prompt-baseline plumbing,
emission eval fixed at the decode boundary with `skip_special_tokens=False`),
the Stage2 launch path was copied frozen rather than re-implemented, and
Stage2.5 stayed out. The scaffolding fails loudly everywhere it is incomplete,
which is the correct failure mode. The defects that remain are the small,
familiar kind this project keeps getting bitten by — default paths and decision
text that quietly disagree with the plan of record (F1–F3) — and they are all
sub-five-minute fixes. Fix those three, launch the 1.5B run, and let the
battery decide everything after step 400.
