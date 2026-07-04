# SafeChain Stage 1 LOSO Data Readiness Review Packet

Date: 2026-07-04

This packet is for external Fable-style review of whether the current SafeChain
Stage 1 generated/generated paired data is sufficient to launch the planned
leave-one-source-out (LOSO) experiment.

Primary file:

- `FABLE_STAGE1_LOSO_DATA_READINESS_REQUEST.md`

Review focus:

- Whether the current source counts are enough for Stage 1 LOSO.
- Whether any source-specific data is still missing, especially for HarmBench,
  StrongReject, and HarmThoughts.
- Whether WildJailbreak should be capped, downsampled, or source-balanced.
- Whether the stale freeze audit must be rerun before launch.
- What final data-freeze checklist is required before claims are allowed.

Disclosure boundary:

- This packet contains aggregate counts, planned gates, audit summaries, and
  reviewer questions.
- It excludes raw prompts, raw trajectories, raw model outputs, credentials,
  cloud endpoint details, Jupyter tokens, SSH details, and private logs.

