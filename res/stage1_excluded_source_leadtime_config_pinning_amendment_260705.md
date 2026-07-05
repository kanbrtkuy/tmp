# Stage1 Excluded-Source Lead-Time Config-Pinning Amendment

Date: 2026-07-05

Status: required amendment to
`res/stage1_excluded_source_leadtime_confirmation_prereg_plan_260705.md`.
This file pins the exact recipes that the implementation must use before
Fable-5 code review. It does not change the preregistered estimands, sources,
k grid, gates, or disallowed actions.

## Provenance Anchors

Primary preregistration packet:

- tmp commit: `feefabb`
- packet:
  `tmp/stage1_auto_improve_loop_260705/round1_leadtime_confirmation_plan_packet/`

Plan-only review:

- reviewer: `claude-fable-5`
- outcome: `OK_TO_IMPLEMENT_PLAN_ONLY`
- required condition: this config-pinning amendment is checked during code
  review; code still needs a separate Fable-5 `OK_TO_RUN`.

Run artifacts to treat as fixed references:

- Stage1 post-HB run root:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso`
- Stage1 GPU archive root:
  `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20`
- Module M matched-horizon output:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/matched_horizon`
- Module M summary JSON in tmp packet:
  `tmp/stage1_auto_improve_loop_260705/round1_results_packet/results/matched_horizon/stage1_matched_horizon_summary.json`
- A1 summary JSON in tmp packet:
  `tmp/stage1_auto_improve_loop_260705/round1_a1_results_packet/results/stage1_score_pooling_summary.json`
- A2 summary JSON in tmp packet:
  `tmp/stage1_auto_improve_loop_260705/round1_a2_results_packet/results/stage1_feature_pooling_summary.json`

Resolved Stage1 fold configs are archived at:

- `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/stage1_natural_pairs_8b_a100_1x_loso_reasoningshield/stage1_natural_pairs_8b_a100_1x_loso_reasoningshield_resolved.yaml`
- `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/stage1_natural_pairs_8b_a100_1x_loso_strongreject_full/stage1_natural_pairs_8b_a100_1x_loso_strongreject_full_resolved.yaml`
- analogous resolved configs for `harmbench_standard` and
  `wildjailbreak_vanilla_harmful`

## Fixed Sources And Frozen Population

Sources:

- `strongreject_full`
- `reasoningshield`

The implementation must compute a frozen test population per source before
scoring: pair-complete rows present for both arms at every k in
`{4,8,16,32,64}`. Every reported A1 and A2 test comparison must use this same
per-source frozen population at every hidden/text k. If either source retains
fewer than 150 test pairs, the run halts before scoring.

## Module M Surface Recipe

The surface arm is fixed to the Module M selected family:

- family: `char_tfidf`
- tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- whitespace tokenizer fallback: disallowed
- k grid: `4,8,16,32,64`
- text horizon: prompt plus first k CoT tokens under the Module M truncation
  semantics
- vectorizer: `sklearn.feature_extraction.text.TfidfVectorizer`
  - `analyzer="char_wb"`
  - `lowercase=True`
  - `ngram_range=(3,5)`
  - `min_df=1`
  - `max_features=200000`
- estimator: `sklearn.linear_model.LogisticRegression`
  - `class_weight="balanced"`
  - `max_iter=2000`
  - `random_state=260705`
  - `solver="lbfgs"`
- surface-family reselection is forbidden

This recipe is pinned by:

- `scripts/data/run_stage1_matched_horizon_reanalysis.py`
- Module M JSON field `surface_selection.selected_family == "char_tfidf"`
- Module M JSON field `layer_selection.selected_layer == 28`

## A1-Compatible Hidden Score Recipe

A1 reuses the original Stage1 single-position hidden probe scores. It must not
retrain a different estimator for A1.

Existing excluded-source score artifacts are archived under:

- `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/stage1_natural_pairs_8b_a100_1x_loso_reasoningshield/runs/linear/linear_cot_{k}_l28/`
- `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/stage1_natural_pairs_8b_a100_1x_loso_strongreject_full/runs/linear/linear_cot_{k}_l28/`

for k in `4,8,16,32,64`, each containing:

- `probe.pt`
- `metrics.json`
- `predictions_val.jsonl`
- `predictions_test.jsonl`

The underlying original Stage1 single-scan recipe is:

- launcher: `legacy/PauseProbe/scripts/probe/run_position_scan_batched.py`,
  called through `legacy/PauseProbe/scripts/probe/run_position_scan_full.py`
  and `scripts/run_stage1_positionscan.py`
- model kind: `linear`
- selected layer for this confirmation: `28`
- positions: exactly `cot_4,cot_8,cot_16,cot_32,cot_64`
- hidden features: one 4096-d hidden vector per selected position/layer
- preprocessing: train-split standardization, stored in `probe.pt`
- classifier: linear sigmoid probe trained with Torch
  - weights shape: `(hidden_dim,)`, bias scalar
  - loss: `binary_cross_entropy_with_logits`
  - optimizer: `AdamW`
  - learning rate: `3e-4`
  - weight decay: `0.0`
  - class balancing: BCE `pos_weight = n_neg / n_pos`
  - sample weights: `source_label`
  - early-stop metric: validation AUROC
  - thresholding: validation threshold constrained by `threshold_max_fpr=0.05`
  - scan epochs/patience/batch sizes as stored in each archived `metrics.json`
    and resolved YAML

A1 cumulative scoring is fixed to the already reviewed A1 rule:

- for target hidden k, pool single-position hidden scores from all
  `j <= k` in the fixed k grid
- z-score each hidden position by that position's validation split mean/std
- use the unweighted mean of those validation-z-scored hidden scores
- for pooled cross-source rows, z-score both hidden and surface arms per
  source using validation-split stats before concatenation
- no learned weights, max pooling, layer search, hyperparameter search, or
  retraining

This recipe is pinned by:

- `scripts/data/run_stage1_score_pooling_reanalysis.py`
- A1 summary JSON `script_version == stage1_score_pooling_reanalysis_v1`

## A2 Feature-Pooling Recipe

A2 must use the already reviewed feature-level refit recipe:

- hidden input: layer 28 only
- positions available to target k: `cot_j` for fixed-grid j where `j <= k`
- feature pooling: unweighted mean of hidden vectors over those positions
- classifier: `StandardScaler + LogisticRegression(class_weight="balanced")`
- logistic regression `max_iter=2000`
- train on train split only
- validation used only for reporting and cross-source score normalization
- compare to the same frozen matched-horizon `char_tfidf` text@k scores

This recipe is pinned by:

- `scripts/data/run_stage1_feature_pooling_reanalysis.py`
- A2 summary JSON `script_version == stage1_feature_pooling_reanalysis_v1`

## Extract-Minimal Hidden Requirement

Dense hidden archives are already present only for
`harmbench_standard` and `wildjailbreak_vanilla_harmful`. For this excluded
confirmation, `reasoningshield` and `strongreject_full` may be extracted only
under the following minimal settings:

- frozen splits from
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/{source}/`
- model/tokenizer from the archived resolved Stage1 configs
- `pause_layout=none`
- `n_pause_tokens=0`
- `label_field=trajectory_safety_label`
- `layers=28`
- `cot_offsets=4,8,16,32,64`
- `max_length=12288`
- no additional layers, positions, sources, prompts, rollouts, or resampling

The extraction outputs must be stored as metadata/hidden artifacts locally and
backed up to R2 after completion. Review packets may include only manifests,
safe metadata ids/labels, counts, hashes, and aggregate metrics.

## Decision Rule Pins

The implementation must write these rules into its preregistration JSON before
any fit/score loop:

- primary A1 estimand: pooled A1 hidden@4 minus text@8 delta AUROC
- A1 primary gate: CI low >= 0
- per-source sanity: no source A1 hidden@4 minus text@8 point estimate < -0.02
- A2 robustness gate: pooled A2 hidden@4 minus text@8 CI high >= 0 and point
  estimate >= -0.01
- only if all three gates pass may the result be called "confirmed"
- A1 pass with A2 fail is "replicated but recipe-sensitive"
- A1 fail drops the lead-time claim

No same-horizon result, secondary cell, or pair-rank result may become the
headline after seeing the data.
