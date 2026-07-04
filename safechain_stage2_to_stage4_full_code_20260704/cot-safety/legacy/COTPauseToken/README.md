# COTPauseToken

Minimal code for the SafeChain pause-token SFT experiments.

The active workflow is now the intra-think pause SFT experiment: train
`DeepSeek-R1-Distill-Qwen-1.5B` to remove the old three pause tokens before
`<think>` and instead place three `<|pause|>` tokens inside the early reasoning
trajectory, immediately before tokenizer-aligned `cot_3`.

The older `candidate_mix_10k` / pre-think `pause3` workflow remains documented
below as historical reference only. It is not the default data route for the
next SFT run.

The repository does not store model weights or large datasets. The current data,
models, and comparison artifacts are backed up separately at:

```text
safechain_gdrive:Research/COTPauseToken/
```

Current Google Drive layout:

```text
safechain_gdrive:Research/COTPauseToken/
  pre_think_pause3_sft_260610/
    code/
    data/
    outputs/
  intra_pause_cot3_sft_260615/
    data/
    logs/
    outputs/
```

`pre_think_pause3_sft_260610/` contains the older experiment that inserted
three `<|pause|>` tokens before `<think>`. `intra_pause_cot3_sft_260615/`
contains the current experiment that inserts three pause tokens inside
`<think>`, immediately before tokenizer-aligned `cot_3`.

## What This Repo Contains

```text
configs/
  data/deepseek_pause_sft.yaml
  experiment/trl_train/deepseek_pause_full_sft.yaml
  trainer/trl_trainer/sft.yaml

scripts/data_generation/pause_sft/
  build_trusted_cot_sft.py        Build the 18k trusted long-CoT raw pool.
  build_intra_think_pause_sft_splits.py
                                  Write intra-cot3 pause and matched controls.
  validate_intra_think_pause_sft_format.py
                                  Validate tokenizer-aligned intra-think pause placement.
  summarize_intra_think_pause_compliance.py
                                  Summarize generated intra-think pause compliance.
  run_intra_pause_cot3_data_prep.sh
                                  Data-prep wrapper for the active experiment.
  build_candidate_mix_sft.py      Build the raw 10k candidate mix.
  build_pause_sft_splits.py       Write pause3 and no-pause train/val/test splits.
  validate_pause_sft_format.py    Validate pause prefix, think blocks, and tokenizer ids.
  vllm_generate_sft_shard.py      Generate comparison outputs with vLLM.
  judge_prometheus_vllm_shard.py  Score outputs with Prometheus.
  summarize_prometheus_compare.py Summarize base/no-pause/pause3 judge scores.
  run_4gpu_vllm_generate.sh       4-GPU generation launcher.
  run_4gpu_judge.sh               4-GPU judge launcher.

scripts/training/
  run_4gpu_intra_pause_sft.sh     4-GPU SFT launcher for the active experiment.

src/
  trl_train.py                    Hydra/TRL SFT entrypoint.
  utils/                          Small formatting, Hydra, and logging helpers.
```

See the current experiment docs:

```text
docs/intra_think_pause_sft_experiment.md
docs/intra_think_pause_sft_experiment_zh.md
res/intra_think_pause_sft_handoff.md
res/intra_think_pause_sft_handoff_zh.md
```

## Install

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r pip_requirements.txt
```

For vLLM generation/judging, install vLLM in the same environment or in a
separate inference environment.

## Active Data Format

All active SFT rows are JSON objects with:

```json
{
  "id": "example_id",
  "input": "user prompt",
  "output": "<think>...<|pause|><|pause|><|pause|>...</think>\nfinal answer",
  "source": "source_name"
}
```

The active `intra_pause_cot3` variant starts directly with `<think>`, has no
pause token before `<think>`, and inserts exactly three pause tokens inside the
think block before tokenizer offset `cot_3`. The matched no-pause control uses
the same rows and splits but has no pause tokens.

## Active Intra-Think Pause SFT Workflow

Build the trusted open-source long-CoT pool:

```bash
python scripts/data_generation/pause_sft/build_trusted_cot_sft.py \
  --output_dir data/pause_sft/trusted_cot_18k \
  --source_quotas sky_t1_17k=6000,bespoke_stratos_17k=6000,openthoughts_114k_metadata=6000
```

Current prepared pool:

| Source | Rows | Notes |
| --- | ---: | --- |
| `NovaSky-AI/Sky-T1_data_17k` | 6000 | Open long-CoT reasoning SFT data. |
| `bespokelabs/Bespoke-Stratos-17k` | 6000 | Apache-2.0 reasoning traces. |
| `open-thoughts/OpenThoughts-114k`, metadata subset | 6000 | Uses `deepseek_reasoning` and `deepseek_solution`. |

Then build tokenizer-aligned intra-think pause splits:

```bash
bash scripts/data_generation/pause_sft/run_intra_pause_cot3_data_prep.sh \
  data/pause_sft/trusted_cot_18k/trusted_cot_raw.jsonl \
  /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B \
  data/pause_sft/trusted_cot_18k_intra_cot3 \
  17000 500 500
```

This writes three matched variants:

```text
data/pause_sft/trusted_cot_18k_intra_cot3/intra_pause_cot3/{train,val,test}.json
data/pause_sft/trusted_cot_18k_intra_cot3/no_pause_matched/{train,val,test}.json
data/pause_sft/trusted_cot_18k_intra_cot3/pre_think_pause3_matched/{train,val,test}.json
```

Current RunPod validation status:

| Check | Result |
| --- | --- |
| Raw rows | 18000 |
| Accepted rows after tokenizer placement | 18000 |
| Rejected rows | 0 |
| Split sizes | 17000 / 500 / 500 |
| `intra_pause_cot3` format errors | 0 |
| `no_pause_matched` format errors | 0 |
| `pre_think_pause3_matched` format errors | 0 |
| Empty think rows | 0 |

Train the main intra-think pause model on 4x A6000:

```bash
cd /workspace/COTPauseToken
PATH=/workspace/venvs/cotpause/bin:$PATH \
bash scripts/training/run_4gpu_intra_pause_sft.sh \
  /workspace/COTPauseToken/data/pause_sft/trusted_cot_18k_intra_cot3/intra_pause_cot3 \
  /workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615 \
  /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B \
  deepseek_intra_pause_cot3_full_sft \
  1
```

The launcher defaults to `NPROC_PER_NODE=4`,
`PER_DEVICE_TRAIN_BATCH_SIZE=2`, and `GRADIENT_ACCUMULATION_STEPS=2`, for
effective batch size 16. If the run OOMs, use
`PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=4`.

## Legacy Pre-Think Pause3 Workflow

This section records the completed pre-think `pause3` experiment. Do not use it
as the default recipe for the next SFT run.

### Build the 10k Candidate Mix

```bash
python scripts/data_generation/pause_sft/build_candidate_mix_sft.py \
  --output_dir data/pause_sft/candidate_mix_10k
```

The intended source mix is:

| Source | Rows |
| --- | ---: |
| OpenThoughts metadata | 4500 |
| Bespoke-Stratos-17k | 2500 |
| SmolTalk constraints | 1000 |
| SmolTalk rewrite | 750 |
| SmolTalk summarize | 750 |
| SmolTalk magpie-ultra | 500 |

Then build the train/val/test splits:

```bash
python scripts/data_generation/pause_sft/build_pause_sft_splits.py \
  --input_jsonl data/pause_sft/candidate_mix_10k/candidate_mix_10k_raw.jsonl \
  --output_root data/pause_sft/candidate_mix_sft_10k \
  --train_size 9000 \
  --val_size 500 \
  --test_size 500 \
  --n_pause_tokens 3
```

This writes:

```text
data/pause_sft/candidate_mix_sft_10k/pause3/{train,val,test}.json
data/pause_sft/candidate_mix_sft_10k/no_pause/{train,val,test}.json
```

### Validate the Splits

```bash
python scripts/data_generation/pause_sft/validate_pause_sft_format.py \
  --dataset_dir data/pause_sft/candidate_mix_sft_10k/pause3 \
  --expected_pause_tokens 3 \
  --tokenizer_path "$DEEPSEEK_MODEL_PATH" \
  --output_json data/pause_sft/candidate_mix_sft_10k/pause3_format_validation.json

python scripts/data_generation/pause_sft/validate_pause_sft_format.py \
  --dataset_dir data/pause_sft/candidate_mix_sft_10k/no_pause \
  --expected_pause_tokens 0 \
  --tokenizer_path "$DEEPSEEK_MODEL_PATH" \
  --output_json data/pause_sft/candidate_mix_sft_10k/no_pause_format_validation.json
```

The validation should report `total_errors: 0`. For pause3, `<|pause|>` should
be a single tokenizer token after it is added.

### Train Pause3

Set the model and data paths:

```bash
export DEEPSEEK_MODEL_PATH=/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B
export PAUSE_SFT_DATA_DIR=/workspace/SafeChain/data/pause_sft/candidate_mix_sft_10k/pause3
```

Run SFT:

```bash
python src/trl_train.py experiment=trl_train/deepseek_pause_full_sft
```

Common 4-GPU launch pattern:

```bash
torchrun --nproc_per_node=4 src/trl_train.py \
  experiment=trl_train/deepseek_pause_full_sft \
  trainer.args.gradient_accumulation_steps=4 \
  hydra.run.dir=/workspace/outputs/deepseek_pause3_candidate_mix_10k_lr2e5_260610
```

The final model is saved under:

```text
${hydra.run.dir}/final
```

### Reference Training Parameters

These are the parameters used for the completed 10k pause3 and no-pause SFT
runs. The experiment config stores the base settings; the 4-GPU runs overrode
`gradient_accumulation_steps` from `16` to `4`.

In this section, "pause token" means `<|pause|>`.

| Parameter | Value used |
| --- | --- |
| Base model | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Training data | `candidate_mix_sft_10k/{pause3,no_pause}` |
| Train / val / test rows | `9000 / 500 / 500` |
| Training method | Full-model SFT with TRL `SFTTrainer` |
| Added token, pause3 run | pause token |
| Added token, no-pause run | none |
| Epochs | `2.0` |
| Learning rate | `2e-5` |
| LR scheduler | linear |
| Warmup ratio | `0.03` |
| Weight decay | `0.0` |
| Max sequence length | `4096` |
| Per-device train batch size | `1` |
| Per-device eval batch size | `1` |
| Number of GPUs | `4` |
| Gradient accumulation | `4` in the 4-GPU reference run |
| Effective train batch size | `16` sequences |
| Precision | `bf16=true`, `fp16=false` |
| Gradient checkpointing | enabled |
| Evaluation cadence | every `200` steps |
| Checkpoint cadence | every `200` steps |
| Checkpoint retention | `save_total_limit=3` |
| Reporting | `report_to=none` |
| Final output directory, pause3 | `/workspace/outputs/deepseek_pause3_candidate_mix_10k_lr2e5_260610/final` |
| Final output directory, no-pause | `/workspace/outputs/deepseek_nopause_candidate_mix_10k_lr2e5_260610/final` |

The 4-GPU reference run therefore used:

```text
effective_batch_size = 4 GPUs * 1 sequence/GPU * 4 gradient accumulation = 16
```

### Train the No-Pause Control

Use the same config but point `PAUSE_SFT_DATA_DIR` at the no-pause split:

```bash
export PAUSE_SFT_DATA_DIR=/workspace/SafeChain/data/pause_sft/candidate_mix_sft_10k/no_pause

torchrun --nproc_per_node=4 src/trl_train.py \
  experiment=trl_train/deepseek_pause_full_sft \
  rl_algorithm.policy.model.special_tokens_to_add=[] \
  run_name=deepseek_nopause_full_sft \
  tags='[deepseek,nopause,full_sft]' \
  trainer.args.gradient_accumulation_steps=4 \
  hydra.run.dir=/workspace/outputs/deepseek_nopause_candidate_mix_10k_lr2e5_260610
```

### Compare with vLLM and Prometheus

Use the test split as prompts and compare:

```text
base    DeepSeek-R1-Distill-Qwen-1.5B
nopause no-pause SFT final
pause3  pause3 SFT final
```

The 4-GPU launchers in `scripts/data_generation/pause_sft/` are thin wrappers
around:

```bash
python scripts/data_generation/pause_sft/vllm_generate_sft_shard.py
python scripts/data_generation/pause_sft/judge_prometheus_vllm_shard.py
```

After generation, pass each model's generation directory to the judge launcher.
It reads `shard_00.jsonl` ... `shard_03.jsonl` and writes `all_judged.jsonl`:

```bash
bash scripts/data_generation/pause_sft/run_4gpu_judge.sh \
  /path/to/prometheus-7b-v2.0 \
  outputs/pause_sft_compare_vllm_10k_test_260610/pause3 \
  outputs/pause_sft_compare_vllm_10k_test_260610/prometheus_judge/pause3 \
  0,1,2,3
```

Once `base`, `nopause`, and `pause3` are judged, build the comparison summary:

```bash
python scripts/data_generation/pause_sft/summarize_prometheus_compare.py \
  --base outputs/pause_sft_compare_vllm_10k_test_260610/prometheus_judge/base/all_judged.jsonl \
  --nopause outputs/pause_sft_compare_vllm_10k_test_260610/prometheus_judge/nopause/all_judged.jsonl \
  --pause3 outputs/pause_sft_compare_vllm_10k_test_260610/prometheus_judge/pause3/all_judged.jsonl \
  --output_dir outputs/pause_sft_compare_vllm_10k_test_260610/analysis
```

Expected merged outputs:

```text
outputs/pause_sft_compare_vllm_10k_test_260610/{base,nopause,pause3}/generations.jsonl
outputs/pause_sft_compare_vllm_10k_test_260610/prometheus_judge/{base,nopause,pause3}/all_judged.jsonl
outputs/pause_sft_compare_vllm_10k_test_260610/analysis/prometheus_summary.json
```

### Current Reference Results

The completed 10k run used `learning_rate=2e-5`, two epochs, and four GPUs.

| Model | Eval loss near final | Prometheus mean | Pass@4/5 |
| --- | ---: | ---: | ---: |
| base | n/a | 2.632 | 22.0% |
| no-pause SFT | 0.5734 | 2.692 | 28.0% |
| pause3 SFT | 0.5623 | 2.652 | 24.6% |

Prompt-prefix checks on 500 test prompts:

| Model | Prefix behavior |
| --- | --- |
| no-pause SFT | 500/500 start with `<think>` |
| pause3 SFT | 498/500 start with exactly three pause tokens, then `<think>` |

Here, the pause token is `<|pause|>`. It is written outside the table because
Markdown tables treat `|` as a cell separator.

The pause3 model successfully learns the interface. It does not outperform the
same-data no-pause control on the current Prometheus aggregate, so the next
research step is to use the learned pause positions for hidden-state probing or
steering rather than treating pause-SFT alone as the final safety method.
