# Stage1 Threshold And Matched-Horizon Literature Notes

Date: 2026-07-05

Purpose: external-database check for the Stage1 CPU reanalysis plan. This note
does not use raw prompts, raw CoTs, hidden activations, or live experiment
state.

## Takeaways

1. Thresholded accuracy can improve without changing representation quality.
   The appropriate CPU reanalysis is calibration/threshold selection on frozen
   validation scores, then evaluation on held-out test scores. AUROC remains the
   ranking metric; balanced accuracy reflects an operating point.

2. Comparing hidden prefixes against full-trajectory text is not an equal
   information comparison. The fair primary control is a matched-horizon text
   baseline that sees `prompt + first k generated CoT tokens`, with the same
   k as the hidden `cot_k` activation.

3. Prior probing work repeatedly warns that probe accuracy alone can reflect
   task/surface artifacts or probe capacity. The closest precedent is not the
   exact phrase "matched horizon", but the broader practice of adding control
   tasks, surface baselines, capacity/description-length controls, and residual
   tests before interpreting representation probes.

## Verified References

- Guo et al. 2017, "On Calibration of Modern Neural Networks", ICML/PMLR.
  Verified via PMLR. Supports held-out post-hoc calibration and the distinction
  between score ranking and calibrated probabilities.
  https://proceedings.mlr.press/v70/guo17a.html

- Fawcett 2006, "An introduction to ROC analysis", Pattern Recognition Letters.
  Verified via DOI/ScienceDirect. Supports AUROC as a threshold-sweeping ranking
  evaluation and cautions that operating-point decisions require separate
  treatment.
  https://doi.org/10.1016/j.patrec.2005.10.010

- Hewitt and Liang 2019, "Designing and Interpreting Probes with Control Tasks",
  EMNLP. Verified via arXiv record. Directly supports adding controls to test
  whether a probe reflects representation content rather than probe/task
  artifacts.
  https://arxiv.org/abs/1909.03368

- Voita and Titov 2020, "Information-Theoretic Probing with Minimum Description
  Length". Verified via arXiv record. Supports the point that raw probe accuracy
  can be misleading and that effort/capacity should be accounted for.
  https://arxiv.org/abs/2003.12298

- Pimentel et al. 2020, "Information-Theoretic Probing for Linguistic
  Structure", ACL. Verified via arXiv record. Supports treating probing as an
  information extraction question and comparing against baselines.
  https://arxiv.org/abs/2004.03061

- Conneau et al. 2018, "What you can cram into a single vector", ACL. Verified
  via arXiv record. Provides precedent for controlled probing tasks over
  sentence representations.
  https://arxiv.org/abs/1805.01070

- Tenney et al. 2019, "BERT Rediscovers the Classical NLP Pipeline", ACL.
  Verified via arXiv record. Supports position/layer-localized probing as a
  common analysis pattern, with caution about dynamic processing.
  https://arxiv.org/abs/1905.05950

- Zou et al. 2023/2025, "Representation Engineering: A Top-Down Approach to AI
  Transparency". Verified via arXiv record. Background for population-level
  representation directions in LLMs, not direct evidence for Stage1.
  https://arxiv.org/abs/2310.01405

- Ravfogel et al. 2022/2024, "Linear Adversarial Concept Erasure". Verified via
  arXiv record. Background for linear concept subspaces and causal erasure, not
  direct evidence for matched-horizon text controls.
  https://arxiv.org/abs/2201.12091

## Interpretation For Stage1

- Module T should be described as an operating-point correction. If balanced
  accuracy improves, that says the existing threshold was conservative; it does
  not by itself prove a stronger hidden signal.

- Module M is the key fairness correction. The primary comparison should be
  hidden `cot_k` vs text-at-k surface baselines, not hidden prefix vs full-text
  hindsight. Full-text and length-only baselines should remain diagnostic
  ceilings or artifact checks.

- E3 residual stacking is useful only as secondary evidence unless hidden
  train/OOF scores are exported. A validation-trained stacker evaluated on test
  is leakage-safe, but reuses validation after model/layer/family selection.

- Across-k curves are descriptive because pair-complete censoring changes the
  retained population at larger k.
