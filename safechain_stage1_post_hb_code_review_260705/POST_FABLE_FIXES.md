# Post-Fable Fixes

This note documents fixes made after the first Claude Fable 5 review.

## Blocking Fix

Fable found that the human-QA TSV was not truly blinded because it included
`judge_label`, and the RunPod orchestrator sampled a no-text sheet by default.

Changes:

- `scripts/data/sample_stage1_human_qa.py`
  - removed `judge_label` from `stage1_human_qa_sheet.tsv`
  - kept `judge_label` only in `stage1_human_qa_manifest.jsonl`
- `scripts/data/summarize_stage1_human_qa.py`
  - now joins the TSV to the manifest by `qa_id`
  - verifies prompt/reasoning hashes against the manifest
  - records `manifest_jsonl_sha256`
  - adds a safe-side agreement bar in addition to the unsafe-side bar
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
  - samples 60 rows/source by default, while the summarizer gate still defaults
    to at least 50 labeled rows/source
  - passes `--include-text` so the annotation sheet is usable
  - checks that the passing QA summary's manifest hash matches the current QA
    manifest before launching GPU Stage1
- `tests/test_stage1_aux_audits.py`
  - updated so tests cannot read `judge_label` from the TSV
  - fills simulated human labels from the manifest, matching the new blinded flow

## Verification

After the fix, the focused Stage1 test suite passed:

```bash
.venv-stage1-test/bin/python -m pytest \
  tests/test_fixed_budget_gen_gen_selection.py \
  tests/test_embedding_dedup_audit.py \
  tests/test_stage1_pair_freeze_audit.py \
  tests/test_stage1_text_baselines.py \
  tests/test_stage1_surface_audit.py \
  tests/test_stage1_loso_freeze_build.py \
  tests/test_stage1_aux_audits.py
```

Result: `19 passed`.

## LOSO Source Gate Addendum

After checking the live RunPod source-expansion manifest, we found that the
k300 source-expansion run covers SR/HB/WJB but does not itself include
ReasoningShield. The post-HB orchestrator now fails closed before audits/freeze
unless the required LOSO sources are all present at the configured floor:

- `REQUIRED_LOSO_SOURCES` defaults to
  `reasoningshield,strongreject_full,wildjailbreak_vanilla_harmful,harmbench_standard`
- `MIN_LOSO_SOURCE_PAIRS` defaults to `150`
- `EXTRA_LOSO_PAIR_JSONL` can be set to one or more extra fixed pair JSONLs,
  e.g. the ReasoningShield fixed gen/gen pair file
- freeze audit, embedding dedup, LOSO fold build, and QA sampling all consume
  the combined pair inputs

This is meant to prevent accidentally producing a three-source freeze while
believing it is the four-source LOSO plan.

## Second Re-Review Fix

Fable's re-review caught that `row_id` still leaked labels because freeze rows
use IDs like `pair_id::safe` and `pair_id::unsafe`.

Additional changes:

- `scripts/data/sample_stage1_human_qa.py`
  - the annotation TSV now contains only `qa_id`, hashes, text, and annotation
    fields
  - `source_family`, `pair_id`, `row_id`, and `judge_label` are kept only in the
    manifest
  - sampling avoids selecting both arms of the same `pair_id` when possible
- `scripts/data/summarize_stage1_human_qa.py`
  - requires a manifest; removed legacy unblinded TSV fallback
  - merges `source_family` from the manifest before computing per-source gates
  - treats blank/mismatched hash fields as failures
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
  - fails closed if the expected QA manifest is missing at GPU gate time
- `tests/test_stage1_aux_audits.py`
  - asserts the exact blinded TSV header
  - asserts non-text cells do not leak `safe` / `unsafe` before simulated
    annotation

Verification after this second fix:

```bash
.venv-stage1-test/bin/python -m pytest \
  tests/test_fixed_budget_gen_gen_selection.py \
  tests/test_embedding_dedup_audit.py \
  tests/test_stage1_pair_freeze_audit.py \
  tests/test_stage1_text_baselines.py \
  tests/test_stage1_surface_audit.py \
  tests/test_stage1_loso_freeze_build.py \
  tests/test_stage1_aux_audits.py
```

Result: `19 passed`.
