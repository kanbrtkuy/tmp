# Stage1 Length Naturalness + Batch Retune Review Packet

Date: 2026-07-05

This GitHub tmp packet asks Fable to review a narrow Stage1 correction and the
corresponding RunPod retune.

## Review Questions

1. Is the corrected Stage1 plan aligned with the earlier guidance that natural
   same-prompt, same-model safe/unsafe CoT length and style differences should
   not be forcibly matched by dropping primary LOSO pairs?
2. Is the implementation consistent with that plan: no primary word-budget caps
   in the post-HB LOSO pipeline, optional builder caps disabled by default and
   marked technical-only, and length/style handled by downstream controls?
3. Is the long-context extractor remediation reasonable: preserve the lone long
   row by using `model.max_length: 12288`, rather than dropping it as a
   length/style match?
4. Is the A100 extraction batch retune acceptable: `batch_size_per_gpu=20` with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, after `batch_size=20`
   completed and `batch_size=22` OOMed in the long-row sanity test?

## Included Files

- `configs/experiment/stage1_natural_pairs_8b_a100_1x.yaml`
- `configs/experiment/stage1b_natural_pairs_8b_a100_1x.yaml`
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
- `scripts/data/build_stage1_loso_freeze.py`
- `docs/fable_review_stage1_length_naturalness_correction_260705.md`
- `docs/fable_review_stage1_post_run_response_260705.md`
- `docs/fable_review_stage1_post_run_audit_260705.md`

## Aggregate Evidence Only

This packet contains no raw prompts, raw CoTs, raw completions, or unreleased
example text. It includes only code/config files and aggregate review/sanity
statistics.

Relevant aggregate sanity facts:

- Tokenizer audit checked 13,438 rendered rows.
- One unique row/pair exceeded 4096 and 8192 rendered tokens.
- Max rendered token length was 8576.
- Zero rows exceeded 12288 rendered tokens.
- `batch_size=20`, `max_length=12288`, expandable CUDA segments: completed,
  no drops, peak observed memory 78,363 MiB, peak GPU utilization 100%.
- `batch_size=22` with the same settings OOMed, so 20 is the highest validated
  A100 long-context extraction batch for this path.
