# SafeChain Stage2 Review Packet

Date: 2026-07-04

This packet contains the GitHub `origin/main` snapshot material needed to review
SafeChain Stage2 intra-pause SFT and downstream probe/steering concerns.

Source repositories:

- `kanbrtkuy/cot-safety` remote commit: `c372bab6e7dfe67286b34657bc75266f620a1577`
- `kanbrtkuy/SafeChain` remote commit: `0cd2eade7d502df424240efcb0d2b89a14e5f542`

Primary review files:

- `CLAUDE_REVIEW_REQUEST.md`: the full review brief and questions.
- `PROFESSOR_CRITIQUE.md`: professor's concerns to address.
- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`: main Stage2 result summary.
- `cot-safety/configs/experiment/stage2_*.yaml`: Stage2 SFT and model-comparison configs.
- `cot-safety/scripts/run_stage2_sft.py`: config-driven Stage2 runner.
- `cot-safety/legacy/COTPauseToken/src/trl_train.py`: legacy TRL trainer, including format-only training.
- `cot-safety/legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py`: pause-token SFT data builder.
- `cot-safety/res/stage1_stage1b_prompt_baseline_summary_20260630.md`: prompt-only / pre-CoT and LOSO Stage1 response.
- `cot-safety/res/stage1_natural_pair_experiment_results_260703.md`: natural-pair Stage1 controls and remaining confounds.
- `cot-safety/res/deepseek-8b/stage4_cot3_full250_hardsafe/stage4_cot3_full250_hardsafe_summary.md`: downstream steering summary for the cot3 full-SFT diagnostic candidate.

Reference paper:

- Goyal et al., "Think before you speak: Training Language Models With Pause Tokens", arXiv:2310.02226, ICLR 2024.
  https://arxiv.org/abs/2310.02226

The core question for review:

Can we design a Stage2 pause-token training method that reliably inserts pause
tokens at the target CoT position while minimizing capability drift and avoiding
the confound where high-quality SFT data improves general reasoning?

