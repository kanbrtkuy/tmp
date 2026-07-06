# Stage3 KL-Transparent Pause Probe Review Brief

Date: 2026-07-06

## Objective

We are testing whether the new Stage2 KL-transparent pause-emission model creates a useful pause-position readout for unsafe CoT trajectories without merely reproducing prompt risk or ordinary early-CoT content-token signal.

Four-stage research logic:

1. Stage1: show latent separability in the base model using SafeChain-style safe/unsafe CoT labels.
2. Stage2: train a pause-emitting model that inserts pause tokens after cot_4 / before cot_5 while preserving continuation behavior and capability.
3. Stage3: test whether pause hidden states contain separability beyond prompt-only baselines and matched content controls.
4. Stage4: only if Stage3 passes, use the pause position as a steering port to reduce unsafe CoT while preserving capability and avoiding over-refusal/broken output.

This run is the first teacher-forced Stage3 screen on the Stage2 1.5B pilot checkpoint.

## Model And Config

Stage2 checkpoint:

`/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final`

Stage3 config:

`configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5_2xa6000.yaml`

Important details:

- Pause insertion: `insert_cot_offset: 5`, meaning after cot_4 / before cot_5.
- Hidden layers: 7, 14, 17, 21, 22, 28.
- Positions include prompt baselines (`last_prompt_token`, `pre_think`), pause positions (`pause_0..2`), pre/post pause windows, true content controls (`control_cot_5`, `control_cot_6` from matched no-pause forward), and cot offsets.
- Split uses source/label/prompt grouping, with `reasoningshield_test` held out.
- This is still teacher-forced, not on-policy generation.

## Result Files

Local copied result files for review:

- `review-stage/stage3_fable_review_260706/results/single/summary_grid.json`
- `review-stage/stage3_fable_review_260706/results/single/stage3_evidence_report.json`
- `review-stage/stage3_fable_review_260706/results/pooled/**/metrics.json`

Remote persistent copies are also in:

- `/workspace/cot-safety/legacy/PauseProbe/runs/probes/stage3_kl_transparent_1p5b_cot5_2xa6000_single`
- `/workspace/cot-safety/legacy/PauseProbe/runs/probes/stage3_kl_transparent_1p5b_cot5_2xa6000_pooled`
- `/workspace/cot-safety/legacy/PauseProbe/data/hidden/stage3_kl_transparent_1p5b_cot5_2xa6000`

The run completed successfully with `STAGE3_EXIT:0`.

## Main Evidence Report

`stage3_evidence_report.json` reports:

- `status`: `fail_no_independent_pause_signal`
- `pause_only_status`: `fail_no_independent_pause_signal`
- Required margin over baselines: `0.01`
- `pause_minus_best_baseline`: `0.00989860436425316`
- `pause_only_margin`: `0.00989860436425316`

Best validation-selected single-position results:

- Best pause: `pause_2`, layer 22, val AUROC 0.958247, test AUROC 0.966579.
- Best prompt baseline: `pre_think`, layer 21, val AUROC 0.933645, test AUROC 0.937866.
- Best true content control: `control_cot_5`, layer 17, val AUROC 0.958314, test AUROC 0.956680.
- Best post-pause: `post_pause_1`, layer 28, val AUROC 0.956206, test AUROC 0.950444.

Interpretation: the pause signal beats prompt baselines clearly, but barely fails the pre-registered 0.01 margin over the best true content control.

## Additional Single-Position Observations

Best by test AUROC:

- Pause: `pause_1`, layer 28, test AUROC 0.967668, heldout ReasoningShield AUROC 0.897589.
- Prompt baseline: `pre_think`, layer 21, test AUROC 0.937866.
- True content control: `control_cot_5`, layer 28, test AUROC 0.958550.
- cot content: `cot_4`, layer 21, test AUROC 0.964748.

Best by heldout ReasoningShield AUROC:

- Pause: `pause_2`, layer 22, heldout AUROC 0.906179.
- Prompt baseline: `pre_think`, layer 14, heldout AUROC 0.826585.
- True content control: `control_cot_6`, layer 17, heldout AUROC 0.862218.
- cot content: `cot_4`, layer 17, heldout AUROC 0.899722.

This again suggests pause is strong, but early content positions can be nearly as strong.

## Pooled Probe Observations

Best pooled test AUROC:

- `pre_mean_l21`: test AUROC 0.972015.
- `pre_mean_l22`: test AUROC 0.971369.
- `pause_mean_l21`: test AUROC 0.969345.
- `pause_mean_l22`: test AUROC 0.967379.
- `control_cot5_cot6_concat_layers_concat`: test AUROC 0.964993.

Best pooled heldout ReasoningShield AUROC:

- `pause_concat_layers_concat`: heldout AUROC 0.914136.
- `pause_mean_layers_concat`: heldout AUROC 0.913599.
- `pause_concat_l21`: heldout AUROC 0.910325.
- `pause_concat_l14`: heldout AUROC 0.910319.
- `control_cot5_cot6_concat_layers_concat`: heldout AUROC 0.899532.

Pooled pause has a clearer heldout margin than the single-position evidence report, but pre-pause and content controls remain strong.

## Known Limitations

1. This is teacher-forced. It is not yet on-policy sampled CoT with per-generation judge labels.
2. Confidence intervals are not in the evidence report; per-example prediction files exist remotely, but the current report does not bootstrap margins.
3. The confirmatory endpoint `within_prompt_auroc` is not implemented.
4. Stage4 liveness tests have not been run yet.
5. Stage4 should remain paused unless the Stage3 evidence is judged sufficient or we define a more appropriate next Stage3 screen.

## Questions For Fable

Please review the Stage3 code/results and answer:

1. Does this teacher-forced Stage3 result count as evidence that pause contains useful unsafe-CoT signal, or should we treat it as a negative/inconclusive result because the content controls are nearly as strong?
2. Is the `0.01` margin threshold appropriate here? Should the borderline single-position fail but stronger heldout pooled pause result change the decision?
3. Should we proceed to Stage4 liveness tests, or first implement the on-policy confirmatory Stage3 pipeline?
4. What exact next experiment would best distinguish:
   - pause as an independent steering port,
   - pause merely copying nearby early-CoT content state,
   - prompt risk classification?
5. Given the current result, what Stage4 design, if any, is still justified?

Please be strict and objective. If the result does not support Stage4, say so directly.
