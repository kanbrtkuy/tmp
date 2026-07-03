# SafeChain Stage 1 Review Packet

This public packet contains an aggregate-only summary for external review of a
Stage 1 ML safety experiment. It intentionally excludes raw prompts, raw
chain-of-thought trajectories, credentials, private logs, and machine-specific
paths.

## Review Goal

We want an external reviewer to assess whether the current Stage 1 results are
sufficient evidence for hidden-state separability between safe and unsafe
reasoning trajectories, and whether they are strong enough to justify moving to
a small Stage 2 pause-token SFT pilot.

The key distinction we want reviewed:

- **Separability exists**: hidden states contain a signal that distinguishes safe
  vs unsafe trajectories in a same-prompt natural-pair setting.
- **Safety-semantic separability**: the signal reflects a safety-relevant latent
  direction rather than surface trajectory properties such as length, refusal
  templates, lexical style, or generation artifacts.

## Files

- `fable_review_prompt.md`: full external-review prompt with aggregate metrics,
  professor concerns, current results, and specific review questions.

## Data Disclosure Boundary

This packet includes only:

- aggregate pair counts,
- aggregate train/validation/test sizes,
- aggregate surface-baseline metrics,
- aggregate hidden-probe metrics,
- high-level experiment design and open questions.

It does not include:

- raw prompts,
- raw CoT trajectories,
- raw unsafe samples,
- model outputs,
- credentials,
- cloud endpoint details,
- private local paths.
