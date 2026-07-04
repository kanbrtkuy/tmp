# Claude Fable 5 Review: Stage 1 Post-HB Code

I've read the full packet (brief, report, 6 scripts, orchestrator, 2 test files). Review below.

## Verdict on the Seven Questions

1. **LOSO split policy: yes, defensible.** `build_stage1_loso_freeze.py` removes HB from trainval for every non-HB fold, and for the HB fold HB is only the test set. Split overlap is hard-checked, splits are seed-deterministic per `(seed, fold, source)`, and the test asserts the HB-exclusion invariant. Pairs with wrong label multiplicity or ambiguous source are conservatively dropped with reasons.
2. **`[0, 100)` contract: correctly encoded** via `FIXED_BUDGET_SAMPLE_START=0` / `MAX_SAMPLE_IDX=100` in the orchestrator; the freeze, audits, QA sample, and folds all derive from that one selection. `[100, 300)` is not touched here, which is consistent with "sensitivity only".
3. **Human-QA strictness: mostly, with two gaps**. Per-source minimum + unsafe-agreement bar (0.90) are implemented and fail-closed.
4. **Grouped bootstrap: yes, defensible.** `match_family -> pair_id -> id` resamples the pair or prompt family as the unit, which is the correct exchangeable unit for safe/unsafe pairs. The rank-AUC with tie-averaged ranks is correct, and deltas use paired resampling over shared group keys.
5. **Optional inputs: yes, explicit.** Both S-to-S diagnostics and quarantine are env-gated with loud SKIP messages; nothing is silently assumed complete.
6. **GPU gating: yes.** `set -euo pipefail` plus `exit 20` before GPU Stage1 unless a `passes: true` summary exists; CPU audits/baselines run before the gate.

## Blocking

1. **The "blinded" human-QA sheet is not blinded, and by default has no text.** `sample_stage1_human_qa.py` writes `judge_label` directly into the annotation TSV, so annotators see the label they're supposed to independently reproduce. This inflates agreement and invalidates the gate as an agreement check. Compounding it, the orchestrator calls the sampler without `--include-text`, so the sheet it advertises for annotation contains hashes but no prompt/reasoning text. Fix: drop `judge_label` from the sheet; keep it in `stage1_human_qa_manifest.jsonl`; have `summarize_stage1_human_qa.py` join by `qa_id`; and pass `--include-text` in the orchestrator. This must land before annotation, but it does not affect freeze/CPU steps.

## Non-Blocking Nits

- No safe-side agreement bar. Judge-safe rows that humans call unsafe only move ungated overall agreement. Recommend symmetric safe-agreement or overall-agreement bar.
- Gate is not bound to the freeze. `HUMAN_QA_SUMMARY_JSON` can point at any passing summary; recommend checking the sheet/manifest hash recorded in the sample summary.
- `min_labeled_per_source` equals `rows_per_source`, so a single unclear row fails a source permanently. Sample about 60/source or set the minimum below sample size.
- `summarize_rows` names its safe-row-only counter `sources`; it is really pairs-by-source.
- Orchestrator always passes `--force` to freeze builder; overwrite protection rests on timestamped `STAGE1_OUT_ROOT`.
- `wait_for_hb` passes immediately if the HB wrapper has not started yet; a sentinel done file would be more robust.
- Unquoted word splitting of `SAFE_PROMPT_INPUTS` / `EXTERNAL_PROMPT_JSONL` breaks paths with spaces.
- Safe-prompt loader keeps rows with empty labels when a label filter is set; document or require labels.
- Quarantine builds a dense ext-by-ref cosine matrix and double-counts prompts from normalized safe/unsafe rows.
- Confirm on pod that text baseline prediction naming matches bootstrap's expectations.

## Bottom Line

No blocking bugs in the freeze/LOSO/bootstrap/orchestration path itself. OK to sync to RunPod and run CPU gates after HB generation completes, on the condition that the QA-sheet blinding fix lands before any human annotation. GPU Stage1 remains correctly gated either way.
