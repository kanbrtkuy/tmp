# Stage 1 LOSO Data Readiness Review Request

Date: 2026-07-04

This brief is for a Fable-style external review of whether our current Stage 1 paired data is sufficient to run the next Stage 1 experiment, especially the leave-one-source-out (LOSO) portion.

Please answer as a skeptical senior ML experiment reviewer. Focus on data readiness, leakage risks, LOSO fold construction, and whether more data collection is needed before freezing Stage 1.

## Core Question

Given the current source-expanded generated/generated paired dataset below, do we have enough data to run Stage 1 LOSO under the planned protocol, or are we still missing data? If missing data, specify the minimum additional data needed by source family and why.

Please pay special attention to:

1. Whether the current source counts support the LOSO design.
2. Whether HarmBench with 154 pairs is enough as a held-out-only fold/test family.
3. Whether WildJailbreak with 2043 pairs should be downsampled/capped/weighted to avoid dominating training.
4. Whether HarmThoughts should be required before Stage 1, or reserved only as a final never-seen external test.
5. Whether the current stale freeze audit is acceptable, or whether final dedup/quarantine must be rerun before any Stage 1 claim.
6. Whether additional StrongReject or HarmBench data is needed to reduce confidence interval width or avoid fragile folds.
7. What exact "data freeze checklist" should be satisfied before launching Stage 1 probes/baselines.

## Prior Fable Advice We Are Trying To Satisfy

Fable's earlier Stage 1 advice was:

- Drop all old "test-max" numbers and do not use them as evidence. The old 0.926/0.927 LOSO-like numbers were invalid/leaky; the defensible range was closer to 0.70-0.79 and needed clean validation selection.
- Reframe prompt/pre-CoT controls as hygiene checks by construction, not as positive evidence.
- Retract the old cross-scale consistency claim until matched dense-grid validation-selected heatmaps exist.
- LOSO was blocked because the eligible data was effectively ReasoningShield-only. Transfer evidence was zero until source-family expansion was done.
- Highest priority before making any "beyond surface" claim: token-matched truncation curves, then length-matched evaluation and Gates 3/4.

## Planned Stage 1 LOSO Protocol

Dataset design:

- Use on-policy generated/generated pairs only.
- Frozen generator: R1-8B sampler config.
- Frozen judge.
- Source family assigned at generation time.
- Exactly one eval pair per prompt after selection.
- Do not train on HarmBench if it is designated as a test-only family.
- Do not use AdvBench or SafeChain/SafeCoT-derived sources because of provenance/template leakage risk.

Source families:

- ReasoningShield: existing old eligible generated/generated pairs.
- StrongReject: source-expanded generated/generated pairs.
- WildJailbreak vanilla harmful: primary training mass.
- HarmBench standard: intended held-out/test-fold-only source.
- HarmThoughts clean: intended permanently unseen final test, not necessarily part of LOSO.
- WildJailbreak adversarial: stress only.
- XSTest safe: safe diagnostic only.

Dedup/quarantine pre-registration:

- Cross-source duplicate/near-duplicate clusters are quarantined from train/test.
- Within-source near-duplicates collapse to one representative.
- Embedding cosine threshold: >= 0.90.
- Lexical MinHash/Jaccard threshold: >= 0.80.

LOSO folds currently planned:

- F1: hold out ReasoningShield; train on StrongReject + WildJailbreak.
- F2: hold out StrongReject; train on ReasoningShield + WildJailbreak.
- F3: hold out WildJailbreak; train on ReasoningShield + StrongReject.
- F4: hold out HarmBench; train on ReasoningShield + StrongReject + WildJailbreak.
- Validation is drawn only from training families.
- Minimum test pairs per held-out source: 80.

Required reporting:

- Hidden probe dense grid selected by validation only.
- At least 3 seeds.
- Paired bootstrap confidence intervals.
- Surface baselines: length, word TF-IDF, word BoW, char TF-IDF, first-sentence-removed TF-IDF, embedding logistic.
- Compare probe against test-max over all six surface baselines.
- Token-matched truncation curves at k = 4, 8, 16, 32, 64, 96, 128, 160.
- Controls: shuffled-label, last-prompt-token, prompt/pre-CoT hygiene.
- Encoder controls: matched base and random encoders; R1 probe must be at least 0.10 higher.

Pre-registered gates:

- Gate A: mean held-out validation-selected probe AUROC >= 0.68 across 4 folds.
- Gate B: probe AUROC minus max surface test AUROC has paired-bootstrap 95% CI excluding 0 on at least 3/4 folds.
- Gate 2: shuffled-label and last-prompt-token controls at chance.
- Gate 3: safety calibration checks, including S-to-S FPR around <= 10% and XSTest-safe FPR@TPR90.
- Gate 4: R1 probe beats matched base and random encoder by >= 0.10.
- Token truncation: at least one token budget k must beat token-matched surface with CI excluding 0; otherwise any "beyond surface" framing dies.

Allowed claim after passing:

- "Held-out prompt-source separability beyond strongest surface baselines under this protocol."

Forbidden claims:

- Do not claim "safety semantics" or "latent safety manifold."
- Do not use pooled-fold or test-max probe results as main evidence.
- Do not revive old 0.926/0.927 LOSO numbers.

## Current RunPod Data Snapshot

RunPod checked on 2026-07-04.

### Existing old Stage 1 generated/generated data

- Old 8B generated/generated export: 467 pairs.
- After dropping 132 ambiguous "harmthoughts+reasoningshield" pairs, only 335 ReasoningShield pairs are eligible for clean source-family LOSO.
- This was the reason prior LOSO evidence was judged blocked/zero-transfer.

### Mixed source expansion: committed current state

Path: `/workspace/cot-safety/runs/source_expansion_r1_8b_k300_v1`

- Selected generated/generated pairs: 2474.
- By source:
  - WildJailbreak vanilla harmful: 2043.
  - StrongReject full: 277.
  - HarmBench standard: 154.
- Dropped/remaining prompts without both safe and unsafe: 1039.
- Candidate rows generated/judged for model: 163210.
- Quality pass counts: 154341 pass, 8869 fail.
- Last committed sample range in generation summary: 105-109.
- No mixed-source generation process appeared active at the latest check; treat 2474 as the currently usable committed mixed-source count.

### HarmThoughts expansion: committed current state

Path: `/workspace/cot-safety/runs/source_expansion_harmthoughts_r1_8b_k300_v1`

- Selected generated/generated pairs: 315.
- Source: HarmThoughts clean.
- Dropped/remaining prompts without both safe and unsafe: 601.
- Candidate rows generated/judged for model: 15080.
- Quality pass counts: 12476 pass, 2604 fail.
- Last committed sample range in generation summary: 15-19.
- Active process at latest check: generating round 20-24, so HarmThoughts may increase later.

### Stale freeze audit from earlier in the day

Path: `/workspace/cot-safety/runs/stage1_pair_freeze_audit_260704_live/audit_summary.json`

Important: this audit is stale relative to the current mixed-source expansion. It used 2461 mixed-source pairs, while the current mixed-source selection has 2474 pairs.

Audit input:

- 2461 mixed-source pairs available at audit time.
- 467 old normalized 8B generated/generated pairs.
- Total input pairs: 2928.

Audit output:

- Main keep pairs: 2796.
- Dropped: 132, all ambiguous `harmthoughts+reasoningshield`.
- Duplicate clusters: 0.
- Cross-source duplicate clusters: 0.
- Duplicate edges: 0.

Main-keep pairs by source at audit time:

- ReasoningShield: 335.
- StrongReject full: 277.
- HarmBench standard: 154.
- WildJailbreak vanilla harmful: 2030.
- HarmThoughts clean: 0 in main keep for this audit.

Source-readiness conclusion at audit time:

- ReasoningShield: ideal met, 335 pairs.
- StrongReject: minimum met, 277 pairs, near the 280 ideal.
- HarmBench: minimum met, 154 pairs, above the 150 minimum but below the 190 ideal.
- WildJailbreak: ideal met, 2030 pairs.
- HarmThoughts: not included in main keep for audit; current separate expansion has 315 committed pairs and is still running.

Token-window availability at audit time:

- k = 4, 8, 16, 32, 64: all 2796 pairs available.
- k = 128: 2791 pairs.
- k = 256: 2704 pairs.
- k = 512: 966 pairs.
- k = 1024: 27 pairs.

Length-caliper availability at audit time:

- 0.9 caliper: 260 pairs.
- 0.8 caliper: 549 pairs.

## Current Interpretation Before Review

Our current provisional view is:

- The core four-source LOSO now appears possible for a Stage 1 run because ReasoningShield, StrongReject, HarmBench, and WildJailbreak all exceed the minimum 80 held-out test-pair requirement.
- The data is still not frozen because the audit is stale and HarmThoughts is still actively running.
- HarmThoughts may not be required for the four-fold LOSO if it is treated only as a permanently unseen final test, but this should be confirmed.
- WildJailbreak is much larger than all other sources, so training imbalance is a serious design choice; we need advice on downsampling/capping/weighting.
- HarmBench at 154 pairs is the thinnest LOSO/test-only source and may lead to wide confidence intervals; we need advice on whether this is acceptable or whether more HarmBench data is worth collecting.
- The final Stage 1 claim must wait for a rerun freeze audit over the final selected dataset, including cross-source dedup/quarantine, source-family accounting, token-window counts, and fold manifests.

## Requested Reviewer Output

Please provide:

1. A yes/no/conditional verdict: are we data-ready to launch Stage 1 LOSO?
2. A minimal missing-data checklist, if any, with source-specific target counts.
3. A recommended LOSO fold construction given the current imbalance.
4. Whether HarmBench should remain held-out-only or be included symmetrically as a LOSO family.
5. Whether to include HarmThoughts in LOSO, reserve it as a final external test, or wait for more HarmThoughts before freezing.
6. Whether WildJailbreak should be capped/downsampled in training and validation.
7. The exact final audit/freeze checks required before results are claimable.
8. Any changes to the prior gates or thresholds given the current counts.

