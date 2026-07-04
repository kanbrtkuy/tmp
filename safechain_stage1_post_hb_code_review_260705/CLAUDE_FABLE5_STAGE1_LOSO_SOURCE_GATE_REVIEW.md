# Claude Fable 5 Review — Stage 1 LOSO Source Gate (Narrow Final Review)

Date: 2026-07-05
Scope: `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh` (LOSO source gate:
`REQUIRED_LOSO_SOURCES` / `MIN_LOSO_SOURCE_PAIRS` / `EXTRA_LOSO_PAIR_JSONL`,
`verify_loso_sources`) and the "LOSO Source Gate Addendum" in
`POST_FABLE_FIXES.md`. Code-only/content-quiet static review; the packet has no
`src/cot_safety`, so tests were not executed locally.

## Methodological verdict

The fail-closed design is correct and genuinely needed. Without this gate,
`build_stage1_loso_freeze.py` would happily build a `holdout_reasoningshield`
fold with an **empty test set** (`test_ids = set(source_to_pairs.get(holdout, []))`
at build_stage1_loso_freeze.py:304 never errors on an absent source), so a
three-source input would silently masquerade as the four-source LOSO plan.
Failing before audits/freeze, requiring an explicit env override to relax the
floor ("declared pilot"), and making the ReasoningShield supplement an explicit
`EXTRA_LOSO_PAIR_JSONL` input rather than an implicit assumption are all the
right calls. Gate placement is also correct: after fixed-budget selection,
before freeze audit / dedup / fold build / QA sampling, so all downstream
artifacts consume the combined inputs.

## Blocking issue (1)

**B1 — Floor is enforced pre-freeze only; freeze drops can silently erode a
source below the floor after the gate passes.**
`verify_loso_sources` counts unique `pair_id`s in the *raw* combined inputs.
`build_stage1_loso_freeze.py` then drops pairs in `group_pairs`
(build_stage1_loso_freeze.py:189-214) for: missing arm
(`requires_exactly_one_safe_and_one_unsafe`), `ambiguous_source_family`,
`unregistered_source_family`, and — importantly — **duplicate `pair_id`s across
input files are dropped entirely, not deduplicated** (two safe + two unsafe
rows fail the exactly-one check). Nothing re-checks `keep_pairs_by_source` in
`stage1_loso_freeze_summary.json` against `MIN_LOSO_SOURCE_PAIRS`. So the gate
can pass on e.g. 200 raw ReasoningShield pairs while the freeze keeps fewer
than the floor, recreating exactly the failure mode the addendum says the gate
prevents ("believing it is the four-source LOSO plan").

Fix is cheap: after `build_loso_freeze_${fixed_tag}`, add a small check that
reads `stage1_loso_freeze_summary.json` and asserts
`keep_pairs_by_source[s] >= MIN_LOSO_SOURCE_PAIRS` for every required source
(and optionally that `n_dropped_pairs` is reported). Alternatively add a
`--min-pairs-per-source` flag to `build_stage1_loso_freeze.py` and pass it from
the orchestrator.

Classification: blocks the four-source LOSO claim and GPU Stage1 reliance on
the gate. Syncing to RunPod and running CPU steps would not corrupt anything,
but since the fix is a few lines in the same file being synced, fix it before
sync.

## Non-blocking issues / nits

1. **Lossy substring canonicalization (shared with the freeze builder).**
   `canonical()` maps any string containing `wildjailbreak` to
   `wildjailbreak_vanilla_harmful` and any containing `harmbench` to
   `harmbench_standard`. A file of e.g. adversarial-WJB or non-standard-HB
   variants would satisfy the gate (and be registered by the freeze) under the
   wrong sub-source name. Prefer an exact alias map plus fail-on-unknown for
   values that match a family substring but not a registered variant. Same
   pattern exists in `build_stage1_loso_freeze.py:canonical_source`, so at
   least gate and freeze agree today.
2. **Gate logic is a diverging copy of the freeze builder's.** The heredoc
   `source_family` omits the freeze's `pair_id`-prefix fallback
   (build_stage1_loso_freeze.py:98-99) and the exact-registered early return.
   Today the drift direction is fail-closed (gate stricter than freeze), which
   is acceptable, but future edits can invert that. Recommend extracting the
   checker into `scripts/data/` sharing `canonical_source`/`source_family` with
   the freeze builder.
3. **No test coverage for the gate.** Because it lives in a bash heredoc, the
   pytest suite cannot exercise it (missing source, below-floor source, alias
   forms, extra-file merge, duplicate pair_ids). Extraction per nit 2 makes
   this testable; the addendum's `19 passed` does not cover the gate.
4. **Gate output is not persisted.** `verify_loso_sources` is not run via
   `run_step`, so the per-source count JSON goes only to the console. For
   provenance, tee it into `${STAGE1_OUT_ROOT}` or `${LOG_DIR}` next to the
   other step logs.
5. **Unquoted word-splitting of `EXTRA_LOSO_PAIR_JSONL`** (line 259) breaks on
   paths with spaces and is glob-expanded. Documented behavior and fine on
   RunPod; noting for completeness.
6. **Verify default coherence: `MIN_LOSO_SOURCE_PAIRS=150` vs the `[0,100)`
   fixed-budget window.** `select_fixed_budget_gen_gen_pairs.py` is not in the
   packet, so I cannot confirm the default budget window yields ≥150 unique
   pairs per source. If it cannot, the defaults are jointly unsatisfiable and
   every run fails the gate until env overrides are set — fail-closed, but a
   footgun. Please confirm on your side (no private counts needed in this
   packet; a yes/no suffices).

## Methodological caution (not a code defect)

The gate verifies presence and count, not provenance. For the LOSO transfer
claim to be clean, the ReasoningShield file passed via `EXTRA_LOSO_PAIR_JSONL`
must come from the same generation/judging protocol and budget policy as the
k300 primary run; a holdout source generated under a different protocol
confounds the leave-one-source-out estimate. The freeze audit's
`--snapshot-inputs` helps; additionally recording the extra file's sha256 and
its generation-config identifier in the freeze summary would make this
auditable later.

## Verdict

- Fail-closed design and placement: methodologically correct, endorse.
- One blocking gap (B1): enforce the per-source floor on **post-freeze kept
  pairs**, not only raw inputs. Fix before syncing this gate to GitHub/RunPod
  (few lines), and certainly before any GPU Stage1 launch that relies on the
  four-source claim.
- Nits 1–5 are non-blocking; nit 6 is a configuration question to confirm.
- With B1 fixed, this gate is acceptable to sync and run through the CPU
  pipeline after HB generation completes.
