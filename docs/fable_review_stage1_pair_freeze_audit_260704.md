# Fable Review Request: Stage 1 Pair Freeze Audit Code

This packet contains code only. It does not include raw prompts, raw CoTs, or
private JSONL data.

## Context

We are preparing natural safe/unsafe CoT pair datasets for a four-stage CoT
safety project:

1. Stage 1: hidden-state separability probes with prompt-only baselines,
   leave-one-source-out evaluation, surface baselines, token-matched controls,
   and bootstrap confidence intervals.
2. Stage 2: pause-token SFT on general instruction data, not on the safety
   pairs.
3. Stage 3: pause-position separability on on-policy pause-model rollouts.
4. Stage 4: steering at pause hidden states with unsafe reduction,
   over-refusal, and capability evaluations.

The current concern is whether the pair datasets are ready to freeze for the
main Stage 1 run. We need a CPU-only audit that can run while GPU generation
continues. The audit should not interrupt or mutate generation outputs.

## Files To Review

- `scripts/data/audit_stage1_pair_freeze.py`
- `tests/test_stage1_pair_freeze_audit.py`

## Intended Behavior

The script reads one or more pair JSONL files in either format:

- combined rows with `safe_reasoning` and `unsafe_reasoning`;
- normalized two-row pairs with `trajectory_safety_label` and `reasoning`.

It emits:

- source-family counts and readiness against target floors;
- ambiguous-row drops;
- exact and near-duplicate duplicate clusters;
- cross-source duplicate quarantine;
- same-source duplicate canonical retention;
- token length statistics for safe/unsafe reasoning;
- length-caliper retention counts;
- token-window availability counts for token-matched truncation experiments;
- JSONL manifests for all audited pairs, main-kept pairs, and dropped pairs.

It should not print prompts or trajectories.

## Review Questions

1. Does the source-family extraction logic look robust enough for the known
   pair formats?
2. Is the duplicate/quarantine behavior conservative and appropriate for LOSO?
3. Is the `main_keep` definition too strict, too loose, or well scoped?
4. Is length-caliper retention computed in the right direction for our later
   matched-length controls?
5. Are the emitted outputs sufficient to decide whether HT/RS and other source
   families meet final data-prep floors after filtering?
6. Are there code bugs or design issues that could silently miscount, leak
   source duplicates across folds, or drop too much data?
7. Should this script be changed before we run it on the live RunPod data?

Please review as a critical ML experiment/data-integrity reviewer. Focus on
construct-validity risks, silent failure modes, and whether this CPU audit
answers the current data-freeze question.
