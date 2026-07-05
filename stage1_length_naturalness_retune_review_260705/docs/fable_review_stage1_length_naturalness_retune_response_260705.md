# Fable Review Response: Stage1 Length Naturalness + Batch Retune

Date: 2026-07-05

Packet reviewed:
`/private/tmp/kanbrtkuy_tmp_review/stage1_length_naturalness_retune_review_260705`

GitHub tmp status:

- Local tmp repo commit: `173162e` (`Add stage1 length naturalness retune review packet`).
- Push to `https://github.com/kanbrtkuy/tmp.git` was attempted but blocked by
  the current local network proxy with HTTP 403.
- SSH push was also attempted and blocked by the proxy.
- Fable reviewed the local exact packet while the GitHub push remains pending.

## Verdict

Fable verdict: **PASS**.

No blockers were identified for this correction packet.

## Findings

Fable found that the revised Stage1 plan is aligned with the earlier guidance:
natural same-prompt, same-model generated/generated safe/unsafe CoT length and
style differences should be preserved in the primary freeze, not forcibly
matched by dropping pairs.

Fable confirmed:

- The post-HB LOSO pipeline no longer passes `--max-prompt-words`,
  `--max-reasoning-words`, or `--max-final-words`.
- The optional LOSO builder caps default to `0` and are documented as
  technical-only escape hatches, not primary length/style matching rules.
- The 8B A100 Stage1 and Stage1b configs set `model.max_length: 12288`, which
  covers the tokenizer audit maximum of 8576 rendered tokens.
- Preserving the single long row by increasing extractor length is preferable
  to dropping it for length/style matching.
- `batch_size_per_gpu: 20` with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a reasonable A100
  extraction setting: batch 20 completed with no drops, 78,363 MiB peak
  observed memory, and 100% peak GPU utilization; batch 22 OOMed.

## Required Follow-Up

Fable did not request code changes before a formal rerun.

Required checks after the formal rerun:

- Re-audit prediction row coverage and confirm the previously dropped long WJB
  row is no longer missing.
- Certify surface baselines and bootstrap CIs against the exact final freeze.
- Complete human QA on the exact frozen packet used for formal GPU results.

## Execution Implication

Proceed with the natural-preserving Stage1 rerun path:

- no primary word-budget pair dropping
- `max_length: 12288`
- A100 extraction batch 20 with expandable CUDA segments
- downstream length/style controls and probe-minus-surface delta reporting
