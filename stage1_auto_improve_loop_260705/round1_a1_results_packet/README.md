# Stage1 Auto-Improve Round 1 A1 Results Packet

Date: 2026-07-05

This packet is sanitized for Fable-5 result review. It contains aggregate A1
score-pooling result TSV/JSON files only. It excludes raw prompts, raw CoTs,
hidden activations, generated pair files, and prediction JSONL files.

## Code And Review

- `cot-safety` A1 code commit: `99583ed`
- `tmp` A1 code packet: `b0027fb`
- `tmp` A1 blocker patch packet: `03854ee`
- Fable-5 A1 code verdict: `OK_TO_RUN`

## Run

RunPod command used:

```bash
/workspace/venvs/cot-natural/bin/python scripts/data/run_stage1_score_pooling_reanalysis.py \
  --pred-dir /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/matched_horizon/predictions \
  --output-dir /workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/score_pooling_a1_260705_b500 \
  --sources harmbench_standard,wildjailbreak_vanilla_harmful \
  --k-grid 4,8,16,32,64 \
  --holm-ks 8,16,32,64 \
  --surface-family char_tfidf \
  --selected-layer 28 \
  --n-bootstrap 500 \
  --seed 260705 \
  --fail-on-error
```

Run result:

```text
n_summary_rows: 15
n_lead_rows: 75
n_errors: 0
a1_success: true
```

R2 backup:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/score_pooling_a1_260705_b500/
```

## Key Results

Pooled same-horizon comparison, hidden z-mean cumulative score minus text@k:

| k | pooled hidden AUROC | text AUROC | delta AUROC | 95% CI |
|---:|---:|---:|---:|---:|
| 4 | 0.7882 | 0.7323 | +0.0558 | [0.0452, 0.0668] |
| 8 | 0.7844 | 0.7570 | +0.0274 | [0.0183, 0.0366] |
| 16 | 0.7983 | 0.8018 | -0.0035 | [-0.0121, 0.0061] |
| 32 | 0.8138 | 0.8077 | +0.0061 | [-0.0035, 0.0149] |
| 64 | 0.8289 | 0.8495 | -0.0206 | [-0.0298, -0.0120] |

Script success preview:

- pooled hidden AUROC by k:
  - 4: 0.7882
  - 8: 0.7844
  - 16: 0.7983
  - 32: 0.8138
  - 64: 0.8289
- max adjacent AUROC drop: `0.0038`
- k=8 delta CI excludes negative territory: `true`
- a1_success: `true`

Lead-time matrix, pooled delta AUROC hidden@k - text@k':

| hidden \\ text | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|
| 4 | +0.056 | +0.031 | -0.014 | -0.020 | -0.061 |
| 8 | +0.052 | +0.027 | -0.017 | -0.023 | -0.065 |
| 16 | +0.066 | +0.041 | -0.003 | -0.009 | -0.051 |
| 32 | +0.081 | +0.057 | +0.012 | +0.006 | -0.036 |
| 64 | +0.097 | +0.072 | +0.027 | +0.021 | -0.021 |

Per-source same-horizon deltas:

- HarmBench:
  - k=8: +0.0021, CI [-0.0287, 0.0298]
  - k=16: -0.0392, CI [-0.0736, -0.0014]
  - k=32: +0.0029, CI [-0.0304, 0.0369]
  - k=64: -0.0199, CI [-0.0520, 0.0117]
- WildJailbreak:
  - k=8: +0.0295, CI [0.0196, 0.0393]
  - k=16: -0.0013, CI [-0.0112, 0.0085]
  - k=32: +0.0062, CI [-0.0034, 0.0152]
  - k=64: -0.0207, CI [-0.0295, -0.0109]

## Request To Fable-5

Please review these A1 results and decide the next loop step.

Questions:

1. Did A1 satisfy your predeclared stop criterion enough to justify A2
   feature-level cumulative pooling?
2. Does the result count as a real improvement in Stage1 fair equal-horizon
   evidence, or only as a diagnostic/reframing?
3. Should we implement A2 now? If yes, specify the exact minimal code path and
   review checks before running.
4. If A2 is not justified, should we pivot to the lead-time claim now?
5. Is the k=64 negative result fatal for the claim, or compatible with your
   proposed "hidden early warning, text catches up later" framing?

Please return:

- Verdict: implement A2 / pivot now / run another diagnostic first.
- Required code changes for the next step.
- Success/failure gate after the next run.
- Any reporting language constraints to prevent overclaiming.
