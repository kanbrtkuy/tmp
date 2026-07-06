# Fable Review Brief: Stage3 Stage1-Paired WJB Screen

Date: 2026-07-06

Please do a full, critical review of the current Stage3 screen result and the updated Stage3 runner/config. The goal is not to defend the result. The goal is to decide, carefully and without overclaiming, what the current Stage3 evidence supports:

1. whether inserted pause tokens carry any safe/unsafe signal at all;
2. whether that signal is meaningfully stronger or more useful than nearby matched no-pause content-token controls;
3. whether the current evidence is enough to proceed to liveness / Stage4 pilots, or only enough to justify more Stage3 diagnostics.

## Project Logic

Stage1 found early CoT hidden-state separability in the base model. Based on the Stage1-only evidence, the recommended intervention point is after `cot_4` / before `cot_5`, because hidden@cot_4 has already seen token `cot_4`.

Stage2 trains a KL-transparent pause-emitting model:

- base: DeepSeek-R1-Distill-Qwen-1.5B
- method: full SFT-style pause emission, but with continuation preservation / KL-transparency logic
- pause insertion: `<think> t0 t1 t2 t3 t4 <pause><pause><pause> t5 ...`
- current pilot checkpoint: `/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`

Your earlier preflight verdict was conditional pass: Stage2 pilot is behavior-preserving enough to run a teacher-forced Stage3 screen, but Stage4 must stay paused until Stage3 signal and liveness gates pass.

## What Was Run

This Stage3 run uses the corrected Stage1 paired data, not the older external probe data.

Data provenance:

- R2 archive: `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/`
- fold used first: `wildjailbreak_vanilla_harmful`
- restored on RunPod at: `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/wildjailbreak_vanilla_harmful`

The run used preserve-input-splits passthrough mode:

- no source caps
- no resplitting
- no dedupe
- preserves Stage1 `train/val/test`
- rewrites rows only to insert pause after `cot_4`
- matched no-pause controls generated for `control_cot_5` and `control_cot_6`

Stage3 command config:

- `configs/experiment/stage3_intra_pause_probe_stage1_paired_wjb_1p5b_cot5_2xa6000.yaml`
- parent config: `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5_2xa6000.yaml`
- runner: `scripts/run_stage3_intra_pause_probe.py`
- legacy orchestrator: `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`

## Split Counts

Prepared rows:

- all: 5262 rows, 2631 unsafe / 2631 safe
- train: 1100 rows, 550 unsafe / 550 safe
- val: 124 rows, 62 unsafe / 62 safe
- test: 4038 rows, 2019 unsafe / 2019 safe
- prompt overlap train/val/test: 0

Hidden extraction caveat:

- final test hidden manifest has 4034 rows, not 4038
- label counts after extraction: 2015 unsafe / 2019 safe
- 4 rows were skipped by extractor, apparently length-related under `max_length=4096`
- please evaluate whether this small asymmetric drop could affect the interpretation

## Positions And Layers

Extracted layers:

- `[7, 14, 17, 21, 22, 28]`

Positions:

- prompt baselines: `last_prompt_token`, `pre_think`
- pause: `pause_0`, `pause_1`, `pause_2`
- post-pause diagnostics: `post_pause_1`, `post_pause_2`, `post_pause_3`
- true no-pause content controls: `control_cot_5`, `control_cot_6`
- extra diagnostics: `pre_pause_1`, `pre_pause_2`, `pre_pause_3`, `cot_4`, `cot_5`, `cot_6`, `cot_9`, `cot_10`

## Stage3 Success Criteria

There are two different levels of success. Please keep them separate.

Basic Stage3 signal:

- Does a probe trained on pause hidden states predict safe vs unsafe CoT labels above chance?
- This does not need to be extremely strong. The immediate Stage3 question is whether pause has any usable separability signal.

Pause-specific / steering-port evidence:

- Does the pause probe beat prompt baselines and true matched no-pause content-token controls by a meaningful margin?
- This stronger condition is needed before making a pause-specific claim, or before treating pause as a privileged steering port.

So, if pause AUROC is high but content-control AUROC is almost as high, the right interpretation may be:

- Stage3 basic signal: positive
- Stage3 independent pause advantage: weak / not yet established
- Stage4 readiness: unresolved, pending liveness and on-policy checks

## Primary Pause-Specific Gate

As you recommended, the primary Stage3 screen gate is:

`pause_only_margin = best_pause_test_AUROC - max(best_prompt_baseline_test_AUROC, best_true_content_control_test_AUROC)`

Required margin:

- `> 0.01`

This is only a forced-aligned teacher-forced screen. It is not an on-policy trajectory monitoring claim.

## Result

Evidence report:

- local: `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/probe_single/stage3_evidence_report.json`
- remote: `/workspace/cot-safety/legacy/PauseProbe/runs/probes/stage3_stage1_paired_wjb_kl_transparent_1p5b_cot5_2xa6000_single/stage3_evidence_report.json`

Core result:

```json
{
  "status": "fail_no_independent_pause_signal",
  "pause_minus_best_baseline": 0.008021193205490773,
  "pause_only_margin": 0.008021193205490773,
  "pause_only_status": "fail_no_independent_pause_signal"
}
```

Best positions:

| group | model | position | layer | val AUROC | test AUROC |
|---|---:|---:|---:|---:|---:|
| pause | linear | pause_2 | 21 | 0.7034 | 0.7871 |
| post_pause | linear | post_pause_3 | 17 | 0.7578 | 0.7605 |
| prompt_baseline | linear | last_prompt_token | 7 | 0.5000 | 0.5000 |
| true_content_control | linear | control_cot_6 | 28 | 0.7349 | 0.7791 |

Top test-AUROC rows include:

- `post_pause_2` layer 7: test AUROC 0.7953
- `cot_6` layer 17: test AUROC 0.7936
- `control_cot_6` layer 21: test AUROC 0.7929
- `pause_2` layer 22: test AUROC 0.7924
- selected-by-val primary `pause_2` layer 21: test AUROC 0.7871

Interpretation before review:

- Prompt baseline is near chance, so this does not look like pure prompt classification on this split.
- Pause positions have clear absolute signal: selected pause AUROC is `0.7871`.
- However, pause barely beats true content controls: best matched content-control AUROC is `0.7791`.
- The pause-specific margin is `+0.0080`, below the `0.01` gate.
- Therefore the current WJB fold should not be described as "no pause signal." A more precise interpretation is: pause has signal, but independent pause-specific advantage over nearby content controls is not yet established.

## Code Update Since Run

During the WJB run we discovered that the legacy runner only sharded `train`; `val/test` extraction was single-file and therefore could run on only one GPU. The result above remains valid, but test extraction was manually sharded and merged.

I patched the runner so future runs handle this automatically:

- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
  - added `--extract_eval_shards`
  - generalized shard/merge logic from train-only to any split
  - merged shard outputs back to the canonical split paths, so downstream probe input format is unchanged
- `scripts/run_stage3_intra_pause_probe.py`
  - passes `hidden.extraction.eval_shards`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5_2xa6000.yaml`
  - sets `train_shards: 4`
  - sets `eval_shards: 4`
  - sets 1.5B hidden extraction `batch_size_per_gpu: 32`

Verification:

- local py_compile passed
- remote py_compile passed
- local dry-run shows train/val/test shards assigned round-robin to `cuda:0,cuda:1`
- remote dry-run shows:
  - `--extract_jobs 2`
  - `--extract_train_shards 4`
  - `--extract_eval_shards 4`
  - `--extract_devices cuda:0,cuda:1`
  - `--extract_batch_size 32`

Please review whether this code update preserves experimental validity and whether any hidden sharding/merge ordering issue could invalidate metrics.

## Files To Inspect

Core code/config:

- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `scripts/run_stage3_intra_pause_probe.py`
- `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_wjb_1p5b_cot5_2xa6000.yaml`

Results:

- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/probe_single/stage3_evidence_report.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/probe_single/summary_grid.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/probe_single/summary_by_test_auroc.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/hidden_manifests/`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/logs/`

Do not read the raw data snapshot files under:

- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/data_prep/cotpause/*.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/data_prep/nopause/*.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/data_prep/cotpause_shards/**/*.json`
- `review-stage/stage3_stage1_paired_wjb_fable_review_260706/results/data_prep/nopause_shards/**/*.json`

Those files are large raw row snapshots. Use the split counts and hidden manifests in this brief instead.

Previous review:

- `review-stage/stage2_stage3_stage4_fable_review_260706/fable_stage2_pilot_stage3_preflight_review_260706.md`

## Questions For You

1. Is the current WJB Stage3 screen result better interpreted as "pause has signal, but independent pause-specific advantage is not established" rather than simply `fail_no_independent_pause_signal`?
2. Is the near-chance prompt baseline meaningful evidence that this is not prompt-only classification, despite the teacher-forced setup?
3. Does the true content control being almost as good as pause imply the pause token is mostly acting as a readout of nearby CoT content rather than an independent steering port?
4. Should we run the other Stage1 paired folds next, or stop and implement on-policy generation + CoT-segment judging first?
5. Does the updated eval/test sharding code preserve row ordering and matched no-pause control alignment well enough for future runs?
6. Based on this result, should Stage2 be changed to make pause tokens more "live", or should Stage2 remain fixed until we run more folds/on-policy confirmation?
7. What are the most important concrete next code changes before Stage4?

Please give:

- a verdict
- blocking issues
- non-blocking concerns
- recommended next run order
- any exact code/config changes you recommend
