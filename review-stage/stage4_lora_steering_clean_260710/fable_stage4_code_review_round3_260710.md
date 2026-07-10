# Fable Stage4 Code Review Round 3

Reviewer: `claude-fable-5`, high effort

## Verdict

Fable confirmed:

- N1 pre-pause crop dropping the pause run is fixed.
- N2 primary pause arm being marked diagnostic is fixed.
- Pivot artifact existence preflight is fixed enough for smoke.
- Matched-strength checker no longer crashes on `None`, but still needed to skip alpha-zero groups.

## Final Blocker From Round 3

`check_stage4_matched_strength.py` compared `alpha_0p0` groups, which have no hook-applied norms by construction, causing a guaranteed failure on full run roots.

## Action Taken

Added an explicit alpha-zero filter in `compare(...)` and CLI flag `--include_alpha0`. Default behavior now checks only nonzero steering alphas, which is the intended matched-strength gate.

## Fable Smoke Assertions

1. Dry-run matrix checks condition/direction/alpha expansion and diagnostic flags.
2. Real run preflight prints all artifact paths as ready.
3. Tiny real run checks hook fires once on three target tokens, norm cap holds, skip/judge sentinel consistency, diagnostic flags, content crop tail, pause suppression.
4. Matched-strength checker exits 0 for nonzero alpha arms when relative gaps are within tolerance and fails for deliberately empty nonzero arms.
