# Fable Stage4 Code Review Round 2

Reviewer: `claude-fable-5`, high effort

## Result

Fable judged the previous blockers mostly fixed:

- B1 row-set abort/asymmetry fixed at generation level with per-row `skip_judge` / `resolution_status`.
- B2 probe gate mismatch fixed by explicitly setting `gate_mode: none`.
- B3 condition/direction output collisions fixed by path names containing `condition_*` and `direction_*`.
- B4 target overlap fixed by replacing `cot_4,cot_5,cot_6` with pre-pause `cot_2,cot_3,cot_4`.
- B5 matched-strength mostly fixed by adding `check_stage4_matched_strength.py`.

## Remaining Issues Fable Flagged

- N1: pre-pause content crop could drop the pause run and make content arms produce zero usable rows.
- N2: global `diagnostic_targets: true` caused primary `pause_all3` rows to be labeled diagnostic.
- NB1: steering-first pivot skipped even artifact-existence preflight.
- NB3: matched-strength checker could crash on `None` norm means.

## Actions Taken After Round 2

- Changed pre-pause content crop to keep the full pause run in the cropped prefix.
- Added a regression test for pre-pause content crop preserving the pause run and resolving target masks.
- Changed runner so `--diagnostic_targets` is passed only for non-pause targets.
- Added steering-first artifact existence preflight for formal non-dry-run execution.
- Changed matched-strength checker to report `no_values` instead of crashing.
- Replaced `torch.flatnonzero` with `.nonzero(...).flatten()` in Stage4 target/liveness paths for compatibility with the local torch build.
