# Stage2 formal decontamination artifacts

The canonical 2×A100 run does not accept the historical `trusted_cot_raw.jsonl`
row shuffle.  It requires an over-collected candidate pool, a complete external
prompt-vector audit, and an independently completed manual-decision file.

Build the over-collected pool with:

```bash
python legacy/COTPauseToken/scripts/data_generation/pause_sft/build_trusted_cot_sft.py \
  --output_dir /workspace/data/pause_sft/trusted_cot_18k \
  --emit_formal_candidate_pool
```

The candidate pool must retain `source_family_id` for every row.  The formal
freeze then applies NFKC/casefold/whitespace normalized SHA-256, source-family
grouping, and word-5-gram Jaccard at 0.80.  Exact or lexical matches to any
configured formal evaluation file are excluded before per-source quotas are
filled.

The external cosine JSON must use schema
`stage2_prompt_vector_cosine_audit_v1`, status `complete`, threshold `0.90`,
and bind the exact candidate and formal-evaluation SHA-256 values.  Its method
record must identify the embedding model, revision, content SHA-256, and state
`fallback_used: false`.  `comparison_scope` must mark candidate-candidate,
candidate-eval, top-neighbor, and threshold-hit enumeration complete and give
the exact row and pairwise-comparison counts.  Every record in the
union of `top_neighbors` and `threshold_hits` needs a stable `pair_id`, kind,
cosine, and content-bound row/prompt identifiers.

The human file uses schema `stage2_top_neighbor_manual_decisions_v1`, binds the
cosine-report SHA-256 and the same inputs, and must contain exactly one decision
for every reported pair.  Allowed decisions are `remove_candidate`,
`remove_other_candidate` (candidate-candidate only), and `keep_distinct`; each
requires non-empty `reviewer`, `decided_at`, and `rationale`.  A blank,
`pending`, missing, stale, or hash-mismatched file stops the freeze.  The code
never creates a synthetic manual pass.

Run the freeze with:

```bash
python scripts/build_stage2_formal_freeze.py \
  --config configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml
```

It writes the frozen 17,000/500/500 groupwise split, a Stage2 freeze manifest,
hash-bound lexical/exclusion evidence, and `decontamination_formal_eval.json`.
The canonical pause builder verifies those artifacts and preserves the assigned
split.  The Stage4 benign-ledger builder independently rehashes the exact
Stage2 manifest and every required formal evaluation file before accepting the
decontamination report.
