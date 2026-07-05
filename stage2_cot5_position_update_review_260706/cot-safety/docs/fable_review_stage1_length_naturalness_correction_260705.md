# Fable Review: Stage1 Length/Style Naturalness Correction

Date: 2026-07-05

## Question

We asked Fable to review a narrow correction:

- Do not use word-budget caps as the primary LOSO freeze rule.
- Preserve natural same-prompt, same-model generated/generated length/style variation.
- Treat length/style as measured surface confounds.
- Preserve the single too-long row by raising extractor `max_length` rather than dropping pairs by word cap.

Tokenizer-only audit evidence:

- Rendered rows checked: 13,438.
- Rows over 4096 tokens: 2 row appearances, 1 unique row/pair.
- Rows over 8192 tokens: 2 row appearances, 1 unique row/pair.
- Max rendered token length: 8576.
- Rows over 12288 tokens: 0.

Implementation reviewed:

- Main pipeline no longer passes LOSO word-budget caps.
- Optional LOSO builder caps remain disabled by default and are marked technical-only.
- 8B A100 Stage1/Stage1b configs set `model.max_length: 12288`.
- Extraction batch is set to 20 for the longer context path, with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

GPU sanity evidence:

- Test batch: the single long WJB unsafe row plus 11 shorter prepared rows.
- Extractor settings: `max_length=12288`, 8B hidden extractor, 4 layers, one
  CoT offset.
- Result: completed successfully.
- Conservative initial test: `batch_size=12`, feature shape `[12, 1, 4, 4096]`,
  metadata rows 12, peak observed GPU memory 53,405 MiB.
- Retuned final test: `batch_size=20` with expandable CUDA segments, feature
  shape `[20, 1, 4, 4096]`, metadata rows 20, peak observed GPU memory
  78,363 MiB, peak GPU utilization 100%.
- Boundary check: `batch_size=22` with expandable CUDA segments OOMed; use 20
  as the highest validated batch size for the current A100 path.
- Dropped rows: `{}`.

## Fable Verdict

Fable verdict: **methodologically preferable**.

Fable agreed that removing the word-budget drop and preserving natural length variation is the correct call. Artificially homogenizing CoT length by pair dropping can erase real information about how safety-relevant reasoning naturally unfolds and can confound the downstream interpretation.

## Required Follow-Up

Fable did not identify a methodological blocker, but requested:

1. Mark word-cap flags as disabled-by-default technical-only controls.
2. Document that the >4096-token long stratum has N=1 and should not be treated as a powered length stratum.
3. Run a small GPU sanity check for the retuned extraction batch and
   `max_length=12288`.

All three follow-ups are now addressed for the 8B A100 path. The long-token
stratum remains a technical coverage note, not a powered subgroup analysis.

## Retune Review Result

After retuning the extraction batch upward, Fable reviewed the code/config/docs
packet and returned **PASS**. See
`docs/fable_review_stage1_length_naturalness_retune_response_260705.md`.

## Execution Implication

Formal Stage1 should preserve the natural frozen dataset and handle length/style through:

- length-only baseline
- word/char TF-IDF and BoW baselines
- embedding baseline
- token-matched truncation curves
- length-stratified reporting or pre-registered caliper sensitivity
- paired probe-minus-surface delta CIs

The too-long row should be treated as an extractor-feasibility issue, not as a reason to length-match or style-match the dataset.
