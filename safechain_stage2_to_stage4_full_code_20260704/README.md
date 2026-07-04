# SafeChain Stage2 to Stage4 Full Code Review Packet

This packet is for a full code-tree review by Claude Fable 5 Pro, not a diff-only review.

## Snapshot

- Source repo: `cot-safety`
- Snapshot commit: `8d5dd4d Address Fable Stage2 to Stage4 code review`
- Snapshot method: `git archive HEAD`, so the `cot-safety/` directory contains only committed files from that commit.
- Purpose: review the current Stage2 -> Stage3 -> Stage4 framework as complete code, including cross-file assumptions that are easy to miss in a diff.

## Contents

- `cot-safety/`: complete committed code/document/result snapshot from the current SafeChain repo.
- `review_context/`: prior Fable reviews, professor critique, project context, Stage2 flow reviews, and Stage3/Stage4 design discussions.
- `FABLE_FULL_CODE_REVIEW_REQUEST.md`: the review brief Fable should read first.

## Review Scope

The primary scope is the current revised pipeline:

1. Stage2: KL-transparent pause emission training.
2. Stage3: prompt baseline, pause/control probing, on-policy label requirements.
3. Stage4: liveness battery and gated projection steering framework.

The central research question is whether we can train the model to emit pause tokens while preserving continuation/capability, and still keep the pause positions useful as live steering ports rather than making them harmless but inert.
