# cot-safety Refactor Plan

## Goal

Create a single reusable pipeline repository that unifies the existing `PauseProbe` and `COTPauseToken` code. The new repository should let future experiments switch models, datasets, layers, pause placement, judges, and hardware layout through config files rather than Python edits.

The pipeline must preserve the validated four-stage experiment flow:

1. Stage 1: trajectory-position separability probing.
2. Stage 2: intra-CoT pause-token SFT.
3. Stage 3: pause-position separability probing.
4. Stage 4: pause-only steering and evaluation.

## Existing Code To Preserve

### From `external/COTPauseToken`

- TRL/Hydra full-SFT training entry: `src/trl_train.py`.
- SFT formatting/collator utilities in `src/utils`.
- Intra-pause data builder: `scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py`.
- Training launcher pattern: `scripts/training/run_4gpu_intra_pause_sft.sh`.
- Existing configs under `configs/` as a baseline, but renamed and generalized.

### From `PauseProbe`

- Trajectory source normalization and filtering.
- Hidden extraction and position location logic.
- Linear/MLP probe training and evaluation logic.
- Intra-pause probe orchestration.
- Learned-delta pause-only steering training.
- Steered generation.
- Open judge evaluation.
- Capability/safety eval preparation and scoring.

## Problems In The Current Layout

1. Model assumptions are hard-coded in scripts:
   - DeepSeek chat tokens.
   - pause token string.
   - model paths under `/workspace`.
   - layer ids and probe positions.

2. Experiment recipes are mixed into Python defaults:
   - data source caps.
   - safe/unsafe balancing.
   - heldout source choice.
   - steering alpha/seed grids.

3. Launchers are run-specific shell files:
   - good for one run, but fragile for changing models.

4. Artifacts are not standardized:
   - outputs live under several `data`, `runs`, `outputs`, `analysis_reports` layouts.

5. Probe, SFT, steering, generation, judging, and reporting use different entry styles.

## Target Repository Layout

```text
cot-safety/
  README.md
  pyproject.toml
  configs/
    model/
      deepseek_r1_distill_qwen_1_5b.yaml
      deepseek_r1_distill_llama_8b.yaml
      template_deepseek_r1_distill.yaml
    data/
      stage1_positionscan_sources.yaml
      stage2_trusted_cot_18k.yaml
      stage3_intra_pause_probe_sources.yaml
      stage4_steering_eval_sources.yaml
    experiment/
      stage1_positionscan.yaml
      stage2_intra_pause_sft.yaml
      stage3_intra_pause_probe.yaml
      stage4_pause_steering.yaml
      full_four_stage.yaml
    runtime/
      local_cpu.yaml
      a6000_1x.yaml
      a100_4x.yaml
    judge/
      wildguard.yaml
      llamaguard.yaml
      harmbench.yaml
      prometheus.yaml
  src/cot_safety/
    cli.py
    config.py
    schemas.py
    registry.py
    utils/
    data/
      io.py
      sources.py
      normalize.py
      split.py
      sft_builder.py
      eval_builder.py
    formatting/
      chat_templates.py
      pause_insertion.py
      position_locator.py
    sft/
      trl_train.py
      collators.py
      launch.py
    hidden/
      extract.py
      shard.py
      merge.py
    probes/
      features.py
      train.py
      evaluate.py
      heatmaps.py
      run_scan.py
    steering/
      delta_train.py
      generate.py
      directions.py
      summarize.py
    eval/
      generate_comparison.py
      judges.py
      normalize_judges.py
      capability.py
      summarize.py
    reporting/
      manifests.py
      tables.py
      backup_notes.py
  scripts/
    run_stage1_positionscan.py
    run_stage2_sft.py
    run_stage3_intra_pause_probe.py
    run_stage4_steering.py
    run_full_pipeline.py
  pipelines/
    run_4xa100_full_pipeline.sh
    run_stage4_2gpu_eval.sh
  docs/
    architecture.md
    config_guide.md
    runpod_setup.md
    migration_from_pauseprobe_cotpausetoken.md
  tests/
    fixtures/
    test_pause_insertion.py
    test_position_locator.py
    test_probe_matrix.py
    test_steering_hook_scope.py
```

## Configuration Design

### Model Config

Each model gets a config with:

- base model id/path.
- SFT checkpoint path.
- tokenizer path.
- chat template family.
- BOS/user/assistant tokens or chat template mode.
- think open/close tokens.
- pause token string.
- n pause tokens.
- max sequence length.
- dtype.
- default layer ids.
- hidden-state layer indexing convention.

This is the main mechanism for moving from DeepSeek 1.5B to DeepSeek 8B or another architecture.

### Experiment Config

Each stage config defines:

- input data sources.
- filtering and caps.
- train/val/test/heldout split strategy.
- pause placement rule.
- positions to extract.
- layers to scan.
- probe model kind and hyperparameters.
- steering layer, alpha grid, seed grid.
- judges and eval datasets.
- artifact output root.

### Runtime Config

Runtime configs define:

- number of GPUs.
- CUDA device list.
- per-device batch sizes.
- extraction jobs.
- probe jobs.
- generation workers.
- judge workers.
- cache locations.
- Hugging Face token environment file path.

## Public CLI

The new CLI should expose stable commands:

```bash
cot-safety data prepare --config configs/experiment/stage3_intra_pause_probe.yaml
cot-safety sft train --config configs/experiment/stage2_intra_pause_sft.yaml
cot-safety hidden extract --config configs/experiment/stage3_intra_pause_probe.yaml
cot-safety probe scan --config configs/experiment/stage3_intra_pause_probe.yaml
cot-safety steer train-delta --config configs/experiment/stage4_pause_steering.yaml
cot-safety eval generate --config configs/experiment/stage4_pause_steering.yaml
cot-safety eval judge --config configs/experiment/stage4_pause_steering.yaml
cot-safety report summarize --config configs/experiment/stage4_pause_steering.yaml
cot-safety pipeline run --config configs/experiment/full_four_stage.yaml
```

The thin scripts under `scripts/` should call these same CLI internals.

## Artifact Layout

Every run writes under:

```text
runs/{run_name}/
  config.resolved.yaml
  manifest.json
  logs/
  data/
  hidden/
  probes/
  steering/
  generations/
  judges/
  reports/
```

Each step should be resumable with `skip_existing` and should write a manifest containing:

- command/config.
- git commit.
- source files.
- row counts.
- model paths.
- hardware/runtime info.
- output checksums where practical.

## Refactor Phases

### Phase A: Scaffold The New Package

Create `pyproject.toml`, `src/cot_safety`, `configs`, `docs`, `tests`, and a CLI skeleton. Add dependency groups for `sft`, `probe`, `judge`, and `dev`.

### Phase B: Move Shared Utilities

Port common JSON/JSONL IO, text cleaning, stable hashing, label canonicalization, manifest writing, and seed control into `cot_safety.utils` and `cot_safety.data`.

### Phase C: Generalize Formatting And Position Location

Move DeepSeek-specific chat rendering into a model-template config. Port pause insertion and hidden-state position location into reusable modules. Add tests for:

- exact pause insertion before `cot_3`.
- locating `pause_0/1/2`.
- locating pre/post/control positions.
- ensuring steering target scope is pause-only.

### Phase D: Port Stage 2 SFT

Port the TRL SFT entry and intra-pause SFT data builder. Keep Hydra compatibility where useful, but allow the top-level `cot-safety` config to drive model/data/output settings. Add 4×A100 launcher config.

### Phase E: Port Stage 1 And Stage 3 Probes

Port data prep, hidden extraction, probe training, scan orchestration, and heatmap reporting. Replace hard-coded source lists, layer lists, and positions with config. Preserve linear and MLP probe options.

### Phase F: Port Stage 4 Steering And Eval

Port learned-delta training, pause-only steered generation, base/SFT/SFT+steering comparisons, open judges, capability scoring, and summary tables. Make `target_positions: [pause_0, pause_1, pause_2]` explicit and validated.

### Phase G: Compatibility Configs And Smoke Tests

Create config files that reproduce the known 1.5B experiments. Add tiny fixture smoke tests that run without GPU and dry-run GPU commands. This proves the migration has not changed semantics.

## First Implementation Scope

For the first commit, do not attempt a full rewrite of every legacy launcher. Instead:

1. Create the package skeleton and configs.
2. Port the most stable reusable modules:
   - IO/labels/manifests.
   - chat templates.
   - pause insertion.
   - position locator.
   - probe feature matrix construction.
   - steering hook target validation.
3. Add docs explaining how old scripts map to new commands.
4. Add smoke tests for the critical token-position behavior.
5. Add wrappers that can call legacy scripts from config while the deeper port continues.

This keeps the new repo usable quickly while reducing risk.

## Non-Goals For This Refactor

- Do not change the experimental claim.
- Do not change the Stage 4 steering target away from pause tokens.
- Do not rewrite all numeric results.
- Do not put real Hugging Face tokens in GitHub.
- Do not delete old repos or old artifacts.

## Acceptance Criteria

The refactor is acceptable when:

1. `cot-safety` has a clean installable package.
2. Model/template/data/runtime settings live in configs.
3. DeepSeek 1.5B current experiments are represented by configs.
4. Adding DeepSeek 8B requires adding/editing a model config and output paths, not editing core Python.
5. Unit/smoke tests cover pause insertion, position location, and pause-only steering scope.
6. Docs explain how to run on a fresh RunPod 4×A100 80GB node.
7. The repo can be pushed to `kanbrtkuy/cot-safety.git`.
