# Fable Review Request: Stage 1 Post-HB Freeze and LOSO Code

Please review the code snapshot in this directory for correctness and research
methodology. This is a code-only/content-quiet review. Raw prompts, CoTs, model
outputs, bucket paths, and unpublished experiment counts are intentionally not
included.

Primary files to inspect:

- `analysis_reports/stage1_post_hb_code_review_packet_260705.md`
- `scripts/data/build_stage1_loso_freeze.py`
- `scripts/data/sample_stage1_human_qa.py`
- `scripts/data/summarize_stage1_human_qa.py`
- `scripts/data/run_stage1_bootstrap_ci.py`
- `scripts/data/build_stage1_safe_prompt_diagnostics.py`
- `scripts/data/quarantine_stage1_external_prompts.py`
- `pipelines/runpod_stage1_post_hb_freeze_then_loso.sh`
- `tests/test_stage1_loso_freeze_build.py`
- `tests/test_stage1_aux_audits.py`

Questions:

1. Does the LOSO freeze builder implement a defensible four-source Stage 1
   split policy, especially keeping HarmBench out of non-HB train/val and using
   HB only as its own held-out fold?
2. Is the `[0, 100)` fixed-budget primary freeze contract correctly encoded,
   with later HB samples treated as sensitivity/diagnostic data?
3. Are human-QA sampling and summarization strict enough for the manual
   agreement gate previously recommended?
4. Is grouped bootstrap by `match_family,pair_id,id` a defensible default for
   AUROC CIs and model deltas?
5. Are safe-prompt S->S diagnostics and external-prompt quarantine correctly
   separated as explicit optional inputs rather than silently assumed complete?
6. Does the RunPod post-HB orchestrator avoid launching expensive GPU Stage1
   before CPU audits and human-QA gates pass?
7. Please list any blocking bugs, then non-blocking nits. If no blockers, say
   whether the code is acceptable to sync to RunPod and run through the CPU
   gates after HB generation completes.
