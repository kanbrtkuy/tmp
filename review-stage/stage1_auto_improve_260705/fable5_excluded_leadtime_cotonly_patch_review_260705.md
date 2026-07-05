# Fable-5 Narrow Patch Review: Cot-Only Extraction

Date: 2026-07-05

Context: the first RunPod extraction attempt showed that the legacy trajectory
extractor automatically saved `think_last` in addition to the preregistered
cot offsets. The run was stopped before interpretation. A narrow patch added
`--omit_think_last`, updated the helper to pass it, and moved official outputs
to `hidden_archives_excluded_leadtime_cotonly`.

```text
OK_TO_RUN. The patch is sufficient: with the helper's pinned flags, manifest
position_names will be exactly cot_4,cot_8,cot_16,cot_32,cot_64 with
omit_think_last: true, think-marker parsing and row filtering are unchanged
from the approved run, and the fresh cotonly output root prevents the
skip-existing logic from reusing contaminated artifacts. Only follow-ups: keep
the old hidden_archives partials quarantined, and have result review confirm
manifests originate from the cotonly root since HIDDEN_ARCHIVE_ROOT is
env-overridable.
```
