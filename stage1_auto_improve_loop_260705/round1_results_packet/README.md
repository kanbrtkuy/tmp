# Stage1 Auto-Improve Loop Round 1 Packet

Date: 2026-07-05

This packet is sanitized for Fable-5 review. It contains aggregate result files,
method notes, and code/result references only. It excludes raw prompts, raw
CoTs, hidden activations, and generated pair files.

## Current State

Code and prior review:

- `cot-safety` code commit: `1d30c40`
- `cot-safety` result-summary commit: `0db1f86`
- `tmp` prior Fable-5 OK commit: `47267cd`
- Prior Fable-5 verdict for Module T/M code: `OK_TO_RUN`

RunPod/R2 output:

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/

cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/
```

## Included Files

- `stage1_cpu_reanalysis_threshold_matched_horizon_260705.md`
  - Human-readable summary of the last CPU reanalysis.
- `stage1_threshold_matched_horizon_literature_260705.md`
  - External DB/literature verification notes.
- `results/threshold/stage1_threshold_reanalysis.tsv`
- `results/threshold/stage1_threshold_reanalysis.json`
- `results/matched_horizon/stage1_matched_horizon_summary.tsv`
- `results/matched_horizon/stage1_matched_horizon_residual.tsv`
- `results/matched_horizon/stage1_matched_horizon_summary.json`

## Key Observed Results

Module T:

- Hidden current-threshold mean test balanced accuracy: about `0.604`.
- Hidden Platt mean test balanced accuracy: about `0.710`.
- Hidden val-BA-max mean test balanced accuracy: about `0.708`.
- This improves thresholded accuracy, but it is only an operating-point fix.

Module M:

- Equal-horizon primary comparison is hidden `cot_k` vs text-at-k surface.
- Sources: `harmbench_standard`, `wildjailbreak_vanilla_harmful`.
- Global selected layer: `28`.
- Global selected surface family: `char_tfidf`.
- Pooled delta AUROC, hidden minus matched surface:
  - k=4: `+0.0584`
  - k=8: `-0.0046`
  - k=16: `-0.1580`
  - k=32: `-0.0964`
  - k=64: `-0.1499`

So Stage1 remains a negative/control result under the current probe/data setup:
hidden has a small early-prefix advantage at k=4 but loses to same-horizon text
baselines over the main k range.

## Request To Fable-5

Please review the included artifacts and propose the next iteration of the
Stage1 improvement loop.

Important constraints:

1. Do not propose methods that merely improve thresholded accuracy by using
   test labels or asymmetric thresholding across hidden/surface/length arms.
2. The main target is to improve the fair equal-horizon result:
   `delta AUROC(hidden@k - text@k)` and/or within-pair ranking, especially for
   k in `{8,16,32,64}`.
3. Separate suggestions into:
   - CPU-only reanalysis/diagnostics.
   - GPU-light reruns using already saved hidden arrays.
   - GPU-heavy reruns requiring new hidden extraction or new generation.
4. For each suggestion, state:
   - Why it might improve the real Stage1 result instead of just reframing it.
   - Whether it risks leakage/cherry-picking.
   - Minimal code change needed.
   - Minimal experiment command and expected cost.
   - Stop criterion for whether the iteration succeeded.
5. If you think Stage1 cannot be made positive without changing the scientific
   claim or data-generation design, say that explicitly and propose the most
   honest pivot.

Please return:

- Verdict: continue Stage1 improvement loop / stop and pivot / run diagnostics
  first.
- Ranked action items.
- The single best next code/experiment change to implement first.
- Any required code-review checks before running it.
