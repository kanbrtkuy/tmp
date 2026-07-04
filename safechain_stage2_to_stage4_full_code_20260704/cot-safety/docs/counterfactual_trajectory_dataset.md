# Counterfactual Trajectory Dataset

This module builds same-prompt counterfactual trajectory triplets for Stage1
construct-validity experiments.

## Motivation

The original Stage1 data contains source-label confounding: some sources are
mostly safe while others are mostly unsafe or partial. A probe can therefore use
prompt/source/style shortcuts instead of trajectory-specific safety evidence.

The counterfactual dataset fixes the prompt and varies only the trajectory:

```text
same prompt
├── safe trajectory
├── partial trajectory
└── unsafe_redacted trajectory
```

This makes prompt-only probing a stronger baseline. If prompt-only AUROC drops
while early-CoT trajectory AUROC remains high, the result supports a trajectory
signal beyond prompt risk.

## Related Recipes

- Counterfactually Augmented Data: minimally edit examples to flip labels while
  preserving coherence and avoiding unnecessary edits.
- Contrast Sets and MiCE: use local edits to test whether models rely on the
  intended decision boundary.
- Zero-shot LLM-guided Counterfactual Generation: use instruction-following
  LLMs to create counterfactual examples without task-specific fine-tuning.
- Polyjuice: controllable counterfactual perturbations, useful as design
  inspiration but not directly suited to CoT trajectory safety triplets.

We keep the implementation inside this repo because existing public code targets
sentence-level input perturbation, not safety-labeled chain-of-thought
trajectories with non-operational unsafe redaction.

## Safety Policy

`unsafe_redacted` trajectories must show only the abstract direction of unsafe
reasoning. Concrete harmful details, procedures, code, quantities, target
selection, or evasion tactics must be replaced by:

```text
[REDACTED_ACTIONABLE_DETAIL]
```

The stored `final_answer` must remain non-actionable.

## Pilot

The pilot config is:

```bash
configs/data/counterfactual_trajectory_pilot.yaml
```

Prepare OpenAI Batch requests without sending data:

```bash
python3 scripts/data/generate_counterfactual_trajectories_openai.py \
  --config configs/data/counterfactual_trajectory_pilot.yaml \
  --mode batch_prepare
```

Run a small synchronous pilot after explicit approval to send the selected data
to OpenAI:

```bash
source /workspace/secrets/openai.env
python3 scripts/data/generate_counterfactual_trajectories_openai.py \
  --config configs/data/counterfactual_trajectory_pilot.yaml \
  --mode sync \
  --allow-external-api
```

Validate output:

```bash
python3 scripts/data/validate_counterfactual_trajectories.py \
  --input runs/counterfactual_pilot/reasoningshield_train_dpo_counterfactual_pilot.jsonl
```

## Batch API

For larger runs, use the Batch flow:

```bash
python3 scripts/data/generate_counterfactual_trajectories_openai.py \
  --config configs/data/counterfactual_trajectory_pilot.yaml \
  --mode batch_prepare

python3 scripts/data/generate_counterfactual_trajectories_openai.py \
  --config configs/data/counterfactual_trajectory_pilot.yaml \
  --mode batch_submit \
  --allow-external-api

python3 scripts/data/generate_counterfactual_trajectories_openai.py \
  --config configs/data/counterfactual_trajectory_pilot.yaml \
  --mode batch_collect \
  --allow-external-api
```

External API modes require `--allow-external-api` so a config cannot
accidentally exfiltrate private data.

## Current Pilot Result

On RunPod, a 3-prompt ReasoningShield-DPO pilot produced 9 trajectories:

```text
3 safe
3 partial
3 unsafe_redacted
```

The stricter v3 prompt passed the offline validator with `problem_count = 0`.
