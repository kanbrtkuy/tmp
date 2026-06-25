# RunPod Setup

## Fresh Pod Checklist

1. Clone this repo.
2. Create Python environments or install editable package.
3. Inject Hugging Face token through SSH stdin.
4. Restore persistent experiment data/checkpoints from GDrive or download them.
5. Stage hot runtime files to local storage before any GPU job.
6. Validate configs and run smoke tests.

## Token Injection

Do not write the real token into GitHub, docs, shell history, or logs.

Local private token file:

```bash
mkdir -p ~/.safechain_secrets
chmod 700 ~/.safechain_secrets
printf '%s\n' '<PASTE_HF_TOKEN_HERE>' > ~/.safechain_secrets/hf_token
chmod 600 ~/.safechain_secrets/hf_token
```

Install on a pod:

```bash
POD_ALIAS=<new-runpod-ssh-alias>
ssh "${POD_ALIAS}" 'mkdir -p /workspace/secrets /workspace/hf_cache /root/.cache/huggingface && chmod 700 /workspace/secrets /root/.cache/huggingface && umask 077 && TOKEN=$(cat) && printf "export HF_TOKEN=%s\nexport HUGGING_FACE_HUB_TOKEN=%s\nexport HF_HOME=/workspace/hf_cache\n" "$TOKEN" "$TOKEN" > /workspace/secrets/hf.env && printf "%s" "$TOKEN" > /root/.cache/huggingface/token' < ~/.safechain_secrets/hf_token
```

Load for remote commands:

```bash
source /workspace/secrets/hf.env
```

## Storage Layout

RunPod network volumes mounted at `/workspace` are persistent, but they can be
FUSE-backed.  Do not load active base models, judge models, SFT checkpoints, or
write-heavy run outputs directly from that filesystem during GPU jobs.  Keep it
as cold storage and stage the hot working set to local storage first:

```bash
cd /workspace/cot-safety
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
bash pipelines/runpod_stage_hot_storage.sh --check-only
```

For machines with local NVMe or a RunPod high-performance volume, set
`COT_SAFETY_HOT_ROOT` to that mount instead of `/dev/shm`.  The hot root must
have enough space for the current base model, active SFT checkpoint, judge
models, eval data, and temporary outputs.

Stage only the files needed for the next job:

```bash
cd /workspace/cot-safety
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
bash pipelines/runpod_stage_hot_storage.sh \
  --model DeepSeek-R1-Distill-Llama-8B \
  --judge Llama-Guard-3-8B \
  --judge wildguard \
  --judge HarmBench-Llama-2-13b-cls \
  --output deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun/checkpoint-500 \
  --data model_comparison_eval/deepseek_8b_stage2
```

The pipeline wrappers source `pipelines/runpod_hot_env.sh`, which maps:

- `COT_SAFETY_MODEL_ROOT` to `$COT_SAFETY_HOT_ROOT/models`
- `COT_SAFETY_JUDGE_ROOT` to `$COT_SAFETY_HOT_ROOT/models/judges`
- `COT_SAFETY_DATA_ROOT` to `$COT_SAFETY_HOT_ROOT/data`
- `COT_SAFETY_OUTPUT_ROOT` to `$COT_SAFETY_HOT_ROOT/outputs`
- `COT_SAFETY_RUN_ROOT` to `$COT_SAFETY_HOT_ROOT/runs`
- `HF_HOME` to `$COT_SAFETY_HOT_ROOT/hf_cache`

After a successful run, copy only results/checkpoints that must persist back to
`/workspace` or GDrive.  Base models and judge models should be re-downloaded or
re-staged on the next pod, not backed up in experiment archives.

## GDrive Restore

Use GDrive backups for experiment artifacts only: code snapshots, configs, data
prepared from benchmarks, SFT checkpoints, run logs, and result summaries.  Do
not restore base models or judge models from backups unless the network is down;
they are large, downloadable, and make migration slower.

```bash
mkdir -p /workspace/restore_archives
rclone copy \
  --transfers=16 --checkers=32 --buffer-size=64M \
  --drive-chunk-size=256M --drive-upload-cutoff=256M \
  --stats=5s --progress \
  safechain_gdrive:Research/cot-safety/runpod_backups/<BACKUP_ID>/archives \
  /workspace/restore_archives
```

Unpack restored experiment artifacts into `/workspace`, then stage only the
active working set to hot storage:

```bash
tar -xzf /workspace/restore_archives/<ARCHIVE>.tar.gz -C /workspace
cd /workspace/cot-safety
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
bash pipelines/runpod_stage_hot_storage.sh \
  --output deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun/checkpoint-500 \
  --data model_comparison_eval/deepseek_8b_stage2
```

## D-State Triage

If `nvidia-smi` shows idle GPUs while a Python job still exists, check whether it
is blocked in kernel I/O wait:

```bash
pgrep -af 'python|run_open_judges|run_model_comparison_eval'
ps -o pid,stat,wchan:32,cmd -p <PID>
findmnt -T /workspace -no TARGET,FSTYPE,SOURCE,OPTIONS
```

`STAT` containing `D` with wait channels such as `folio_wait_bit_common` usually
means the process is stuck waiting on filesystem/page-cache I/O.  On RunPod this
has happened when models or checkpoints were loaded directly from the
FUSE-backed `/workspace` network volume.  Stop the stuck job, stage the hot files
to `$COT_SAFETY_HOT_ROOT`, and relaunch from the staged paths.

## 4×A100 80GB Runtime

Use:

```yaml
defaults:
  - configs/runtime/a100_4x.yaml
```

This is the preferred setup for DeepSeek 8B full SFT and larger hidden-state /
judge workloads.

Example hot-storage eval launch:

```bash
cd /workspace/cot-safety
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
CONFIG=configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt500_2xa100.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
bash pipelines/run_4xa100_model_comparison_eval.sh
```
