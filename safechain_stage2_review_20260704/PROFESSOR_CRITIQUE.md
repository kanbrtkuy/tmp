# Professor Critique To Address

The reviewer should treat the following as the motivating critique.

## Stage 1 Construct Validity

Construct-validity question on Stage 1: is the token_3 signal about the
reasoning, or about the prompt? In a teacher-forced setup the prompt fully
determines those early tokens, so a probe on the last pre-`<think>` position
(prompt only, no CoT) might already hit about 0.97. If it does, we are mostly
doing prompt classification and the "trajectory monitoring" framing weakens.
Add a prompt-only / pre-CoT probe as the baseline and show the trajectory adds
signal beyond it.

## Source Generalization

Generalization is currently one held-out source (RS-Test). That is not enough to
rule out source, format, or length artifacts, and it is probably why the fixed
threshold drifts across datasets. Leave-one-source-out across all six sources
would give an actual transfer estimate, plus a variance-across-sources number we
can report directly.

## Capability And Steering Confounding

Handle the capability table carefully. Steering raising GSM8K and MATH is a
yellow flag rather than something to lean on; an unsafe-removal vector should
not improve math. It suggests the delta may be picking up a general
"structured / careful" direction confounded with the high-quality SFT data, not
a clean unsafe axis.

Two things help:

1. Cleanly separate the SFT-data effect from the steering effect.
2. Characterize what delta actually correlates with: refusal rate, length,
   topic, etc., so we can argue it is "unsafe" and not a proxy.

## Teacher-Forced Versus Self-Generated Distribution Mismatch

The probe is validated on curated teacher-forced traces, while steering runs on
self-generated traces at generation time. Different distributions. The Stage 4
end-to-end numbers cover this somewhat, but name the mismatch explicitly.

## Calibrated Safety Claims

On framing: "markedly lowers unsafe" is fair in relative terms, but residual
unsafe-valid is still about 33% in-domain and over 50% under LlamaGuard. Keep
the absolute residual visible so the safety claim stays calibrated.

