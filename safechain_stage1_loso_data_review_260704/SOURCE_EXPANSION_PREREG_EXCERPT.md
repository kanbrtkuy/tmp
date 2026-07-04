# Source Expansion Pre-Registration Excerpt

This short excerpt captures the pre-registration details that are relevant to
the Fable5 Stage 1 LOSO data-readiness review.

Planned source-expansion design:

- LOSO families: ReasoningShield, StrongReject, HarmBench-standard, and
  WildJailbreak-vanilla.
- HarmThoughts-clean: permanently unseen final test, not a LOSO family.
- WildJailbreak planned sample: 800 vanilla-harmful prompts.
- Frozen generation design: on-policy generated/generated pairs using the
  frozen R1-8B sampler and frozen judge.
- Planned rollout budget: k = 50 rollouts per prompt for the source expansion.
- Selection criterion: exactly one eval pair per prompt, with one quality-passing
  safe candidate and one quality-passing unsafe candidate.
- Dedup/quarantine: cross-source near-duplicate clusters quarantined; embedding
  cosine threshold >= 0.90 or lexical MinHash/Jaccard threshold >= 0.80.
- Human QA expectation: approximately 50 stratified human spot-checks per source
  with an agreement table before claim-bearing freeze.

Reason this matters:

- If pair selection uses unequal or adaptive rollout budgets by source, pair
  existence can encode source-correlated sampling difficulty.
- Final freeze should therefore report per-source attempted prompts, rollout
  counts, fixed-budget yield, dedup/quarantine evidence, and human spot-check
  agreement before Stage 1 probe results are considered admissible.

