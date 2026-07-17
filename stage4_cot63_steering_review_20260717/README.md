# Stage4 cot63 Steering Review Bundle

Date: 2026-07-17

Purpose: provide a compact, reproducible bundle for fable5 to review the current Stage4 strict A1-replay steering design, related code, and latest diagnostic results.

This bundle intentionally excludes model checkpoints, raw generations, secrets, and large hidden arrays.

## Scope

Model/checkpoint:

- DeepSeek-R1-Distill-Llama-8B
- cot63 pause SFT checkpoint-500
- Stage4 target subset: cot63, target-resolved rows only

Protocol:

- Strict A1-replay counterfactual flow.
- Generate A1 baseline.
- Replay exact A1 tokens to `pause_2`.
- Intervene on pause hidden state(s).
- Continue with common-random continuation.
- Analyze only the 309 target-resolved rows.

Baseline on the 309 rows:

| label | count |
|---|---:|
| safe | 209 |
| unsafe | 99 |
| unknown | 1 |

Baseline rates:

| metric | value |
|---|---:|
| unsafe rate | 99/309 = 32.04% |
| degeneration rate | 70/309 = 22.65% |
| unsafe_or_deg_unknown | 138/309 = 44.66% |

## Files

### Code

- `code/diagnose_stage4_cot63_target309.py`
  - Current diagnostic/generation/analyze script.
  - Includes fixed-relative, score-gated, and GPRS replay paths.
- `code/run_stage4_judge.py`
  - Stage4 judge wrapper used for WildGuard normalized labels.

### Results

- `results/score_gate_existing_a2_summary.json`
  - Existing fixed-relative A2 composition/gating summary.
- `results/projection_scores.jsonl`
  - Per-row projection scores used for score-gating diagnostics.
- `results/gprs_replay_diagnostic_analysis_summary.json`
  - Raw analyzer output for GPRS replay.
  - Important caveat: the analyzer's generic `unsafe_rate=0` fields are not valid for WildGuard three-class labels (`unsafe_valid`, `safe_refusal`, `ambiguous_partial`). Use the normalized files and manual transition counts below.
- `results/gprs_wildguard_normalized/*.jsonl`
  - WildGuard normalized labels for GPRS replay at alpha 0.1, 0.25, and 2.0.

## Current key results

### Existing fixed-relative steering

Best observed strict A1-replay fixed-relative result:

| alpha | baseline unsafe → safe | still unsafe |
|---:|---:|---:|
| 0.1 | 30/99 = 30.3% | 69/99 = 69.7% |
| 0.25 | 23/99 = 23.2% | 76/99 = 76.8% |
| 0.5 | 24/99 = 24.2% | 75/99 = 75.8% |
| 1.0 | 22/99 = 22.2% | 77/99 = 77.8% |

Overall table from the same 309-row target-resolved subset:

| arm | alpha | unsafe rate | relative to A1 | degeneration |
|---|---:|---:|---:|---:|
| A1 baseline | 0 | 44.66% unsafe_or_deg_unknown | — | 22.65% |
| A2 fixed-relative | 0.1 | 41.42% unsafe_or_deg_unknown | -3.24 pp | 21.68% |
| A2 fixed-relative | 0.25 | 41.42% unsafe_or_deg_unknown | -3.24 pp | 20.06% |
| A2 fixed-relative | 0.5 | 43.04% unsafe_or_deg_unknown | -1.62 pp | 19.74% |
| A2 fixed-relative | 1.0 | 41.75% unsafe_or_deg_unknown | -2.91 pp | 19.74% |

Interpretation so far: steering can flip a meaningful fraction of originally unsafe valid rows, but the overall pp gain is small because the 309-row pool contains many baseline-safe rows and collateral/format effects dilute the net gain.

### Existing score-gated A2 composition

Best by unsafe rate:

| method | alpha | gate top | unsafe rate | delta unsafe | unsafe_or_deg_unknown | delta unsafe_or_deg_unknown |
|---|---:|---:|---:|---:|---:|---:|
| score_gate_existing_a2 | 0.1 | 60% | 86/309 = 27.83% | -4.21 pp | 41.75% | -2.91 pp |

Best by `unsafe_or_deg_unknown`:

| method | alpha | gate top | unsafe rate | delta unsafe | unsafe_or_deg_unknown | delta unsafe_or_deg_unknown |
|---|---:|---:|---:|---:|---:|---:|
| score_gate_existing_a2 | 0.25 | 60% | 94/309 = 30.42% | -1.62 pp | 41.42% | -3.24 pp |

### Legacy GPRS projection replay

GPRS replay formula was tested inside the same strict A1-replay framework. WildGuard normalized results:

| alpha | overall unsafe_valid | baseline unsafe → safe_refusal | still unsafe_valid | ambiguous_partial |
|---:|---:|---:|---:|---:|
| 0.1 | 95/309 = 30.74% | 9/99 = 9.09% | 76/99 = 76.77% | 14/99 = 14.14% |
| 0.25 | 107/309 = 34.63% | 6/99 = 6.06% | 82/99 = 82.83% | 11/99 = 11.11% |
| 2.0 | 103/309 = 33.33% | 5/99 = 5.05% | 77/99 = 77.78% | 17/99 = 17.17% |

Conclusion so far: legacy GPRS projection replay is worse than fixed-relative steering in this strict replay flow. Earlier norm audit showed `alpha=0.1` had median relative perturbation around 0.003 and about 29% of token interventions were zero because projection <= 0.

## Known caveats / review targets

Please review:

1. Whether the current fixed-relative steering implementation truly preserves the strict A1-replay counterfactual design.
2. Whether score-gating/adaptive-alpha can be made official without leaking A2 outcome or judge labels.
3. Whether multi-layer steering is scientifically clean:
   - use multi-layer scores only for gate/alpha?
   - steer multiple fixed layers with norm matching?
   - train a multi-layer direction?
4. Whether the current analysis code mishandles WildGuard's three-class labels anywhere else.
5. Whether observed patterns imply the bottleneck is algorithm, pause SFT contract, target position, label/data composition, or sample composition.

## SHA-256

```text
267c74776df4d842177b4c23c011832bcff948819a5431189d7c87d67252379e  code/diagnose_stage4_cot63_target309.py
2636bc44e4c9edc0f283d8a0b87ec59e232c33662aeaf4dd215cfa4f2516e935  code/run_stage4_judge.py
d47835e5a81128f6996d5af07b6bbb65b0ddbdae00285f75592dee109f4a8b83  results/gprs_replay_diagnostic_analysis_summary.json
18977765c75dbb37ace5acb54ecf39a6283a18e7df70231c8cbea629d02d5676  results/score_gate_existing_a2_summary.json
d09ae41fac72dfb1fb6531ddbb115c142e7f24fef2573b6ada0f63314076e4db  results/projection_scores.jsonl
d5d2cccc64312a1a5c14d3afa4df64cf341fea9cb42cc63986851f03cb41b34b  results/gprs_wildguard_normalized/alpha_0p1.open_judges_normalized.jsonl
29a33ecce0df25598976ae07fd394e33411201c831ecf172e2838e4194c46901  results/gprs_wildguard_normalized/alpha_0p25.open_judges_normalized.jsonl
fe7a4142212e1491a7dd752dd8da1d56f9116644d860de764f99ce3cba3d04b5  results/gprs_wildguard_normalized/alpha_2.open_judges_normalized.jsonl
```
