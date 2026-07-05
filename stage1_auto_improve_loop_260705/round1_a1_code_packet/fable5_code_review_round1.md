# Fable-5 A1 Code Review Round 1

Verdict: `BLOCKED`

True blocker:

- `source="pooled"` rows compared hidden scores that were already per-source
  z-normalized against raw surface scores.
- This could manufacture a positive pooled delta from a cross-source calibration
  offset in the surface arm.
- The primary endpoint and Holm family are based on pooled rows, so this was a
  real blocker.

Required fix:

- For pooled rows only, z-score both arms per source with validation-split
  statistics before cross-source concatenation:
  - hidden: val stats of pooled hidden scores for that source/k.
  - surface: val stats of surface scores for that source/k.
- Add a regression test where per-source hidden and surface are identical up to
  a per-source constant offset; pooled delta must be near zero.

Non-blocking caveats:

- Rank-accuracy bootstrap currently undercounts duplicated clusters.
- Position metadata fallback is enforced by path when explicit fields are
  absent; the audit dict is not separately reported.
- Surface metadata can be checked with the same expected-k mechanism.
- Diagonal hypotheses appear in both summary and lead matrices with different
  bootstrap seeds; summary is canonical.
- Bootstrap p-values can be exactly zero at B=500.

