# Fable Review Response: Stage1 Post-Run Audit

Date: 2026-07-05

Input packet:

- GitHub tmp repo: `kanbrtkuy/tmp`
- Tmp commit reviewed: `0cea8a9`
- Packet path: `stage1_fable_review_260705/`

Claude/Fable access note:

- Direct handoff was blocked by platform policy.
- GitHub `tmp` packet handoff was used.
- The Fable environment could authenticate but could not fetch GitHub due to network egress restrictions, so it reviewed the local clone of the exact tmp repo packet.

## Verdict

Fable verdict: **provisional, with a conditional path to formal acceptance**.

The current GPU Stage1 sequence is usable only as provisional. Formal Stage1 claims require:

1. Apply the word-budget gate to the final LOSO freeze and confirm the too-long row is removed.
2. Complete human QA on the exact gated frozen packet.
3. Rerun affected folds with the gated freeze.
4. Certify baseline/CI artifacts.

## Must-Fix Blockers

### Blocker A: Extraction Row Coverage

Fable agrees this is a formal blocker.

Evidence from the packet:

- A WildJailbreak unsafe row with hash `1bedd82f59c0f070` has 6305 reasoning words.
- It exceeds the extractor's `extract_max_length=4096` path and was dropped as `too_long`.
- It is missing from StrongReject-fold validation predictions and WildJailbreak-fold test predictions.

Fable accepts the fail-closed word-budget gate as the right remediation, rather than extractor-side truncation.

Required rerun scope after final gated freeze:

- `stage1` and `stage1b` for StrongReject fold.
- `stage1` and `stage1b` for WildJailbreak fold.
- ReasoningShield and HarmBench folds may be carried forward only if their extraction coverage remains certified.

### Blocker B: Human QA

Human QA remains a formal blocker.

Current state:

- QA packet exists.
- Human annotations and agreement summary do not exist.
- The previous GPU run has a bypass marker.

Fable says retrospective QA is acceptable, but it must be performed on the final gated frozen packet.

### Blocker C: Baseline/CI Certification

Fable says the existing surface and bootstrap artifacts are not yet formally certified.

Required checks:

- Baselines and CIs must match the final gated freeze counts.
- Reporting must be validation-selected, not test-max.
- Delta CIs must compare probe versus the correct surface baseline, not only report marginal probe CIs.
- Truncation curves must include per-k sample sizes and avoid claims for low-n windows.

## Methodologically Acceptable Items

Fable judged the following as acceptable in principle:

- Fixed-budget N=100 re-selection.
- Embedding/TF-IDF dedup near-band check with zero cross-source near-band hits.
- Source-stratified QA sampling protocol.
- Token-window and length-caliper reporting.
- WJB train/val cap.
- Batch increase from 16 to 24 on the A100.

## LOSO Concerns

Fable accepts the LOSO layout if its asymmetry is disclosed:

- HarmBench is held-out-only and has no train/val signal.
- WJB appears in train/val for every non-WJB fold, with cap.
- Claims should be framed as held-out source generalization, not source-specific transfer.

Post-word-budget-gate counts from the probe:

| Source | Keep Pairs |
|---|---:|
| HarmBench | 151 |
| ReasoningShield | 304 |
| StrongReject | 271 |
| WildJailbreak | 1953 |

Fable says these counts appear acceptable if they still satisfy the predeclared source-readiness thresholds. HarmBench remains tight and must be disclosed.

## Row-Count Mismatch Guidance

Fable distinguishes two categories:

- High-CoT-offset coverage gaps are expected and should be reported with per-position `n`.
- Extraction-level full-row drops are blockers.

Success criterion after rerun:

- The confirmed too-long row is no longer in the frozen data.
- Any remaining mismatches are high-offset coverage gaps only.
- No extraction-level blockers remain.

## Minimal Remediation Order

Fable recommended:

1. Finalize a gated freeze using the word-budget gate.
2. Certify surface baselines and bootstrap CIs against that gated freeze.
3. Regenerate or validate the human QA packet against that gated freeze.
4. Rerun affected GPU folds: StrongReject and WildJailbreak, both Stage1 and Stage1b.
5. Re-run prediction row coverage audit.
6. Complete human QA.
7. Formally accept Stage1 only after all four conditions pass.

## Notes For Execution

Fable's example CLI flag names in its prose used `--stage1_max_*`; the implemented script uses:

- `--max-prompt-words`
- `--max-reasoning-words`
- `--max-final-words`

Use the implemented flag names in actual runs.

