# RunPod Setup

## Fresh Pod Checklist

1. Confirm GPU count and storage mounts.
2. Install code without unpacking many small files on `/workspace`.
3. Create Python environments or install the editable package.
4. Inject Hugging Face token and rclone config, including the Cloudflare R2
   remote `cloudflare_r2_cot_safety`.
5. Restore prepared experiment data from GDrive to the pod volume.
6. Download base and judge models to the pod volume if they will be reused.
7. Stage only the active working set from the pod volume to hot storage.
8. Validate configs with `--dry_run` before starting any GPU job.
9. Launch the job and confirm all expected GPUs are active.
10. Sync hot outputs back to the pod volume and then to GDrive.

Fast hardware/storage check:

```bash
POD_ALIAS=<new-runpod-ssh-alias>
ssh "${POD_ALIAS}" 'hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /workspace /dev/shm; findmnt -T /workspace -no TARGET,FSTYPE,SOURCE,OPTIONS'
```

If `/workspace` is FUSE-backed, treat it as cold persistent storage only.  Do
not unpack code tarballs, load models, load checkpoints, or write training
outputs there during active setup or training.

## Fast Fresh-Pod Path

Use this path for short-lived RunPod migrations where the base and judge models
can be downloaded again.

1. Prepare secrets:

```bash
POD_ALIAS=<new-runpod-ssh-alias>
ssh "${POD_ALIAS}" 'mkdir -p /workspace/secrets /root/.config/rclone /dev/shm/cot-safety-hot && chmod 700 /workspace/secrets /root/.config/rclone'
scp ~/.safechain_secrets/hf_token "${POD_ALIAS}:/workspace/secrets/hf_token"
scp ~/.config/rclone/rclone.conf "${POD_ALIAS}:/root/.config/rclone/rclone.conf"
ssh "${POD_ALIAS}" 'chmod 600 /workspace/secrets/hf_token /root/.config/rclone/rclone.conf && python3 - <<PY
from pathlib import Path
t = Path("/workspace/secrets/hf_token").read_text().strip()
p = Path("/workspace/secrets/hf.env")
p.write_text(f"export HF_TOKEN={t}\nexport HUGGING_FACE_HUB_TOKEN={t}\n", encoding="utf-8")
p.chmod(0o600)
PY'
ssh "${POD_ALIAS}" 'rclone lsd cloudflare_r2_cot_safety:cot-safety >/dev/null && echo "Cloudflare R2 remote OK"'
```

Before copying `~/.config/rclone/rclone.conf`, make sure the local rclone config
contains the Cloudflare R2 remote.  Store the R2 credentials outside the repo,
for example in `~/.safechain_secrets/r2.env`, and source them only in the shell
that configures rclone:

```bash
# Run once on the local machine, or whenever the R2 token is rotated.
cd <LOCAL_COT_SAFETY_REPO>
source ~/.safechain_secrets/r2.env
bash scripts/ops/configure_r2_remote_from_env.sh
rclone lsd cloudflare_r2_cot_safety:cot-safety
```

Expected variables in `~/.safechain_secrets/r2.env`:

```bash
export R2_ACCESS_KEY_ID='<access-key-id>'
export R2_SECRET_ACCESS_KEY='<secret-access-key>'
export R2_ENDPOINT='https://<account-id>.r2.cloudflarestorage.com'
```

Do not commit `r2.env`, paste R2 secrets into experiment logs, or pass the
secret key directly as a command-line argument on shared machines.

2. Install code into hot storage.

Preferred: use GitHub SSH or an access token so the pod can clone directly.

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm && git clone git@github.com:kanbrtkuy/cot-safety.git cot-safety-src'
```

Fallback: if direct clone is blocked by private-repo auth, transfer a clean
source archive from a local checkout.  Keep it small, avoid Apple extended
attributes, and unpack under `/dev/shm`, not `/workspace`.

```bash
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/cot-safety-src.tar.gz \
  --exclude='cot-safety/.git' \
  --exclude='cot-safety/outputs' \
  --exclude='cot-safety/data' \
  --exclude='cot-safety/hf_cache' \
  --exclude='cot-safety/wandb' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C <LOCAL_PARENT_DIR> cot-safety
scp /tmp/cot-safety-src.tar.gz "${POD_ALIAS}:/dev/shm/cot-safety-src.tar.gz"
ssh "${POD_ALIAS}" 'rm -rf /dev/shm/cot-safety-src && mkdir -p /dev/shm/cot-safety-src && tar -xzf /dev/shm/cot-safety-src.tar.gz -C /dev/shm/cot-safety-src --strip-components=1'
```

Do not use a broad tar exclude such as `--exclude='data'`; it also drops
`configs/data/` and `src/cot_safety/data/`.  Exclude only the repository root's
large data directory, for example `--exclude='cot-safety/data'`.

3. Install dependencies by stage:

```bash
# Stage 1 PositionScan / probe.
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src && PIP_CACHE_DIR=/dev/shm/cot-safety-hot/pip-cache python3 -m pip install -e .'

# Stage 2 SFT.  This is intentionally separate because paged_adamw_8bit needs
# a working bitsandbytes native CUDA library.
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src && PIP_CACHE_DIR=/dev/shm/cot-safety-hot/pip-cache python3 -m pip install -e ".[sft]"'

# Stage 4 vLLM judge/rescore support.  Hook-based steering generation itself
# should not require vLLM.
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src && PIP_CACHE_DIR=/dev/shm/cot-safety-hot/pip-cache python3 -m pip install -e ".[judge]"'
```

Then validate the stage-specific environment before launching that stage:

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source pipelines/runpod_stage1_env.sh'
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source pipelines/runpod_stage2_env.sh'
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source pipelines/runpod_stage3_env.sh'
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source pipelines/runpod_stage4_env.sh'
```

Stage 1 now has a unified entry point that can run the position scan,
prompt/pre-CoT baseline, and leave-one-source-family-out generalization from one
config.  Use this instead of launching Stage1 and Stage1b separately for formal
runs:

```bash
# 1.5B on 2x A6000.
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source /workspace/secrets/hf.env; source pipelines/runpod_stage1_env.sh; python3 scripts/run_stage1.py --config configs/experiment/stage1_unified_1p5b_2xa6000.yaml --skip_existing --dry_run'

# 8B on 2x A6000.
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source /workspace/secrets/hf.env; source pipelines/runpod_stage1_env.sh; python3 scripts/run_stage1.py --config configs/experiment/stage1_unified_8b_2xa6000.yaml --skip_existing --dry_run'
```

For debugging a single component, add `--only position_scan`,
`--only prompt_baseline`, or `--only loso`.  The unified runner writes active
hidden-state caches and probe outputs under
`${COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}` while a module is running.
After each successful position-scan, prompt-baseline, or LOSO fold, it syncs
that module's hot `data/`, hidden-state cache, logs, and probe output directory
to `/workspace` and then removes the hot copy after a `.synced_to_cold` marker
exists.  This keeps `/dev/shm` from accumulating all LOSO folds.  Resolved
configs are written under `runs/`.  LOSO summaries are aggregated into
`${COT_SAFETY_RUN_ROOT:-runs}/stage1_loso_summary/`:

- `stage1_loso_summary_grid.tsv/json`
- `stage1_loso_best_by_family.tsv/json`
- `stage1_loso_prompt_vs_trajectory.tsv/json`

The LOSO aggregation uses combined heldout predictions for a source family when
available.  This matters because some raw sources are safe-only or unsafe-only,
so raw per-source AUROC can be undefined.  The default LOSO folds are therefore
label-balanced heldout folds: mixed-label ReasoningShield splits are held out
directly, while safe-only families such as STAR/AIDSAFE are paired with the
HarmThoughts unsafe source and reported as combined family-level metrics.  This
is the paper-facing number; raw source columns remain available for auditing.

If Stage 2 fails here with a bitsandbytes or `libnvJitLink` error, fix the
CUDA/bitsandbytes environment before training.  Do not switch optimizers unless
the experiment config explicitly changes `runtime.sft.optim`.

4. Restore prepared data from GDrive to the pod volume.  This is the persistent
copy.  Do not restore backup archives straight into `/dev/shm`; `/dev/shm` is
only the hot working cache.

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src && bash pipelines/runpod_restore_gdrive_backup.sh --backup-id <BACKUP_ID> --archive data.tar.gz'
```

The restore script stores the raw archive under
`/workspace/restore_archives/<BACKUP_ID>/` and unpacks it into `/workspace`.
If an archive contains a leading `workspace/data` prefix, the script normalizes
it to `/workspace/data`.

5. Stage only the needed data from the pod volume to hot storage:

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source /workspace/secrets/hf.env; source pipelines/runpod_base_env.sh; bash pipelines/runpod_stage_hot_storage.sh --data pause_sft/trusted_cot_18k_intra_cot4 --data pause_sft/trusted_cot_18k_intra_cot3_control'
```

Verify the SFT data path before training:

```bash
ssh "${POD_ALIAS}" 'find /dev/shm/cot-safety-hot/data/pause_sft/trusted_cot_18k_intra_cot4 -maxdepth 2 -type f \( -name train.json -o -name manifest.json -o -name "*validation*.json" \) -print'
```

6. Download the base model to the pod volume, then stage it to hot storage.
Base and judge models are not backed up to GDrive, but keeping them on the pod
volume avoids re-downloading within the same pod lifetime.

```bash
ssh "${POD_ALIAS}" 'source /workspace/secrets/hf.env; mkdir -p /workspace/models /workspace/hf_cache; HF_HOME=/workspace/hf_cache python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    local_dir="/workspace/models/DeepSeek-R1-Distill-Llama-8B",
    token=True,
    max_workers=8,
)
PY'
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source pipelines/runpod_base_env.sh; bash pipelines/runpod_stage_hot_storage.sh --model DeepSeek-R1-Distill-Llama-8B'
```

7. Run a dry run and check the printed launch environment:

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; source /workspace/secrets/hf.env; source pipelines/runpod_stage2_env.sh; python3 scripts/run_stage2_sft.py --config configs/experiment/<CONFIG>.yaml --skip_existing --dry_run'
```

For the 8B cot4 format-only 250-step run, the dry run should print:

- `NPROC_PER_NODE=4`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=8`
- `EVAL_STEPS=50`
- `SAVE_STEPS=50`
- `SAVE_TOTAL_LIMIT=null`
- `FORMAT_ONLY=true`
- `MAX_STEPS=250`

8. Launch and monitor:

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; mkdir -p /dev/shm/cot-safety-hot/runs/logs; source /workspace/secrets/hf.env; source pipelines/runpod_stage2_env.sh; nohup python3 scripts/run_stage2_sft.py --config configs/experiment/<CONFIG>.yaml --skip_existing > /dev/shm/cot-safety-hot/runs/logs/stage2_sft.log 2>&1 & echo $! > /dev/shm/cot-safety-hot/runs/stage2_sft.pid'
ssh "${POD_ALIAS}" 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; tail -n 80 /dev/shm/cot-safety-hot/runs/logs/stage2_sft.log'
```

During tokenization/map, GPU utilization can be low.  Training has really
started when the log shows `Starting training` and a progress bar such as
`0/250`; at that point all expected GPUs should have high utilization.

9. Persist hot outputs back to the pod volume, then release hot copies after
they have been verified on cold storage.  Do this after each checkpoint that
must survive a pod restart, and again at the end of the run.

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; bash pipelines/runpod_sync_hot_to_cold.sh --output deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-50 --all-runs --remove-hot-after-sync'
```

For longer runs, start a checkpoint watcher immediately after launch:

```bash
ssh "${POD_ALIAS}" 'cd /dev/shm/cot-safety-src; nohup bash pipelines/runpod_watch_hot_checkpoints.sh --output deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250 --stop-pid-file /dev/shm/cot-safety-hot/runs/stage2_sft.pid --remove-hot-after-sync > /dev/shm/cot-safety-hot/runs/logs/watch_hot_checkpoints.log 2>&1 &'
```

The checkpoint is then available under:

```text
/workspace/outputs/deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-50
```

The cold copy contains a `.synced_to_cold` marker.  Only after that marker is
present should the corresponding `/dev/shm` checkpoint be removed.  The
`--remove-hot-after-sync` flag performs that removal automatically for explicit
`--output` paths and for checkpoint watcher syncs.

## 2026-06-25 Fresh 4xA100 Notes

These were the deployment issues observed on a fresh 4x A100 SXM pod and the
fixes that should be applied next time.

| Symptom | Cause | Fix |
| --- | --- | --- | --- |
| `git clone https://github.com/kanbrtkuy/cot-safety.git` failed with `could not read Username` | Private repo auth was not configured on the pod. | Prefer GitHub SSH/token clone. If unavailable, transfer a clean source archive and unpack it under `/dev/shm`. |
| A small code tar unpack on `/workspace` hung in `D` state | `/workspace` was a FUSE-backed RunPod network volume; unpacking many small files and Apple xattrs triggered filesystem I/O wait. | Do not unpack source archives on `/workspace`; use `/dev/shm/cot-safety-src` or local NVMe. Build tarballs with `COPYFILE_DISABLE=1 tar --no-xattrs`. |
| `tar` tried to preserve local macOS uid/gid and printed ownership errors | The archive was created on macOS. | Use `tar --no-same-owner` when unpacking on Linux, and avoid macOS xattrs. |
| Training dry run failed because `configs/data/stage2_trusted_cot_18k.yaml` was missing | The source archive used `--exclude='data'`, which also excluded `configs/data` and package code under `src/cot_safety/data`. | Exclude only `cot-safety/data`, not every path named `data`. |
| First training launch failed with `ModuleNotFoundError: No module named 'rich'` | The legacy COTPauseToken trainer imports `rich`, but it was missing from the SFT optional dependencies. | Keep `rich` in the `sft` extra and install with `pip install -e ".[sft]"`. |
| Data restore appeared empty at `$COT_SAFETY_HOT_ROOT/data` | The GDrive archive was restored directly to hot storage and unpacked with a leading `workspace/data` prefix. | Restore GDrive backups to `/workspace` with `runpod_restore_gdrive_backup.sh`, normalize there, then stage selected data to hot storage. |
| Pod volume looked empty while training was running | The job was writing hot outputs to `/dev/shm` only. | Use `runpod_sync_hot_to_cold.sh` after each important checkpoint and at job end. |
| `/dev/shm` stayed full after checkpoints were backed up | The old flow synced to `/workspace` but did not release the hot checkpoint copy. | Use `--remove-hot-after-sync`; it removes hot checkpoints only after the cold copy has a `.synced_to_cold` marker. |
| A later checkpoint could be lost if the pod stops before manual sync | Checkpoints were written to hot storage first. | Start `runpod_watch_hot_checkpoints.sh --remove-hot-after-sync` after launch so completed checkpoints are copied to `/workspace` automatically and then pruned from `/dev/shm`. |
| GPU utilization was low after launch | The job was still in CPU-side dataset map/tokenization. | Wait for `Starting training` and the step progress bar before judging GPU utilization. |
| `rclone ls` over the full backup was slow | It recursively listed large checkpoint trees. | Use known archive paths, `rclone lsf --dirs-only`, `rclone size`, or direct `rclone cat .../archives/data.tar.gz`. |

## Token And R2 Injection

Do not write real Hugging Face or Cloudflare R2 secrets into GitHub, docs,
shell history, command-line arguments, or logs.

Local private secret files:

```bash
mkdir -p ~/.safechain_secrets
chmod 700 ~/.safechain_secrets
printf '%s\n' '<PASTE_HF_TOKEN_HERE>' > ~/.safechain_secrets/hf_token
chmod 600 ~/.safechain_secrets/hf_token
cat > ~/.safechain_secrets/r2.env <<'EOF'
export R2_ACCESS_KEY_ID='<access-key-id>'
export R2_SECRET_ACCESS_KEY='<secret-access-key>'
export R2_ENDPOINT='https://<account-id>.r2.cloudflarestorage.com'
EOF
chmod 600 ~/.safechain_secrets/r2.env
```

Configure local rclone once, then copy the resulting rclone config to each pod:

```bash
cd <LOCAL_COT_SAFETY_REPO>
source ~/.safechain_secrets/r2.env
bash scripts/ops/configure_r2_remote_from_env.sh
rclone lsd cloudflare_r2_cot_safety:cot-safety
```

Install on a pod:

```bash
POD_ALIAS=<new-runpod-ssh-alias>
ssh "${POD_ALIAS}" 'mkdir -p /workspace/secrets /workspace/hf_cache /root/.cache/huggingface /root/.config/rclone && chmod 700 /workspace/secrets /root/.cache/huggingface /root/.config/rclone && umask 077 && TOKEN=$(cat) && printf "export HF_TOKEN=%s\nexport HUGGING_FACE_HUB_TOKEN=%s\nexport HF_HOME=/workspace/hf_cache\n" "$TOKEN" "$TOKEN" > /workspace/secrets/hf.env && printf "%s" "$TOKEN" > /root/.cache/huggingface/token' < ~/.safechain_secrets/hf_token
scp ~/.config/rclone/rclone.conf "${POD_ALIAS}:/root/.config/rclone/rclone.conf"
ssh "${POD_ALIAS}" 'chmod 600 /root/.config/rclone/rclone.conf && rclone lsd cloudflare_r2_cot_safety:cot-safety >/dev/null && echo "Cloudflare R2 remote OK"'
```

Load for remote commands:

```bash
source /workspace/secrets/hf.env
rclone size cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000 --fast-list
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

The shared storage environment is `pipelines/runpod_base_env.sh`, which maps:

- `COT_SAFETY_MODEL_ROOT` to `$COT_SAFETY_HOT_ROOT/models`
- `COT_SAFETY_JUDGE_ROOT` to `$COT_SAFETY_HOT_ROOT/models/judges`
- `COT_SAFETY_DATA_ROOT` to `$COT_SAFETY_HOT_ROOT/data`
- `COT_SAFETY_OUTPUT_ROOT` to `$COT_SAFETY_HOT_ROOT/outputs`
- `COT_SAFETY_RUN_ROOT` to `$COT_SAFETY_HOT_ROOT/runs`
- `HF_HOME` to `$COT_SAFETY_HOT_ROOT/hf_cache`

Stage-specific GPU jobs should source their full stage env instead of the
base-only env:

| Stage | Environment file | Wrapper | Main extra checks |
| --- | --- | --- |
| Stage 1 PositionScan | `pipelines/runpod_stage1_env.sh` | `pipelines/run_4xa100_stage1_positionscan.sh` | Base inference/probe stack only. |
| Stage 2 SFT | `pipelines/runpod_stage2_env.sh` | `pipelines/run_4xa100_stage2_sft.sh` | TRL/DDP stack and bitsandbytes native CUDA load for `paged_adamw_8bit`. |
| Stage 3 probe | `pipelines/runpod_stage3_env.sh` | `pipelines/run_4xa100_stage3_probe.sh` | Probe imports for SFT-checkpoint hidden extraction and probe training. |
| Stage 4 steering/judge | `pipelines/runpod_stage4_env.sh` | `pipelines/run_4xa100_stage4_steering_eval.sh` | Steering generation imports; optional vLLM check with `COT_SAFETY_STAGE4_REQUIRE_VLLM=1`. |

`pipelines/runpod_hot_env.sh` remains as a backward-compatible base alias for
older utility commands.  Do not use it as proof that Stage 2 is ready.

After a successful run, copy only results/checkpoints that must persist back to
`/workspace` or GDrive:

```bash
cd /workspace/cot-safety
bash pipelines/runpod_sync_hot_to_cold.sh --all-outputs --all-runs
```

For specific checkpoints or final model directories, prefer explicit output
syncs with hot pruning:

```bash
bash pipelines/runpod_sync_hot_to_cold.sh \
  --output deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-250 \
  --all-runs \
  --remove-hot-after-sync
bash pipelines/runpod_sync_hot_to_cold.sh \
  --output deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/final \
  --all-runs \
  --remove-hot-after-sync
```

Base models and judge models should be re-downloaded or re-staged on the next
pod, not backed up in experiment archives.

## GDrive Restore

Use GDrive backups for experiment artifacts only: code snapshots, configs, data
prepared from benchmarks, SFT checkpoints, run logs, and result summaries.  Do
not restore base models or judge models from backups unless the network is down;
they are large, downloadable, and make migration slower.

Restore archives to the pod volume first:

```bash
cd /workspace/cot-safety
bash pipelines/runpod_restore_gdrive_backup.sh \
  --backup-id <BACKUP_ID> \
  --archive data.tar.gz
```

Then stage only the active working set to hot storage:

```bash
cd /workspace/cot-safety
export COT_SAFETY_HOT_ROOT=/dev/shm/cot-safety-hot
bash pipelines/runpod_stage_hot_storage.sh \
  --output deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun/checkpoint-500 \
  --data model_comparison_eval/deepseek_8b_stage2 \
  --data pause_sft/trusted_cot_18k_intra_cot4
```

## Cloudflare R2 Archives

### Stage1 Post-HB A100 Archive, 2026-07-05

The current post-HB Stage1 A100 archive is:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/
```

It contains the post-HB LOSO run, `/workspace/stage1-results`, ordinary hidden
archives, excluded-source cot-only lead-time hidden archives, `/dev/shm` run/log
sidecars, code/config/docs snapshots, and backup manifests. See:

```text
docs/stage1_post_hb_r2_archive_260705_zh.md
docs/stage1_post_hb_r2_archive_260705.md
```

Important restore prefixes:

```text
runs/stage1_post_hb_260705_after_hb_n100_loso/
runs/stage1-results/stage1_post_hb_260705_retune12288_b20/
runs/hidden_archives/
runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly/
runs/dev_shm/cot-safety-hot/runs/
```

Do not restore model caches from this archive; re-download base models and judge
models on a fresh pod.

### Stage1/Stage1b A6000 Archive, 2026-07-01

Stage1/Stage1b A6000 archives are stored in Cloudflare R2 under one canonical
prefix:

```text
cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/
├── deepseek-1p5b/
│   ├── data/
│   └── runs/
│       ├── hidden/
│       ├── logs/
│       └── results/
└── deepseek-8b/
    └── runs/
        └── hidden/
```

Verified size after removing the old duplicated `runpod-backups/` prefixes:

```text
deepseek-1p5b: 55 objects, 152.862 GiB
deepseek-8b:   14 objects, 555.736 GiB
total:         69 objects, 708.598 GiB
```

The 8B R2 archive currently contains hidden-state tar files only.  The 8B
Stage1/Stage1b result summaries are tracked in GitHub under `res/deepseek-8b/`
and `res/stage1_stage1b_prompt_baseline_summary_20260630*.md`; original 8B
RunPod `runs/results/` and `runs/logs/` tar archives were not present in the
source backup that was migrated to R2.

Configure the R2 remote on a fresh machine without putting secrets in shell
history:

```bash
export R2_ACCESS_KEY_ID='<access-key-id>'
export R2_SECRET_ACCESS_KEY='<secret-access-key>'
export R2_ENDPOINT='https://<account-id>.r2.cloudflarestorage.com'
bash scripts/ops/configure_r2_remote_from_env.sh
```

List or restore from the canonical archive:

```bash
rclone lsf cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/ --recursive --max-depth 4
rclone copy \
  cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/deepseek-1p5b/runs/results \
  /workspace/cot-safety/runs/results \
  --transfers=16 --checkers=32 --fast-list --progress
```

The helper script `scripts/ops/organize_r2_stage1_backups.sh` documents the
canonical R2 layout and can be reused if future backups first land under a
temporary prefix.  Keep old prefixes until the new target is verified with
`rclone size`; delete them only after the canonical archive size matches.

The later A100 natural-pair Stage 1 workspace snapshot is stored separately:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

Its layout, verification status, restore examples, and links to the relevant
`plan/` and `res/` documents are recorded in
`docs/stage1_paired_r2_archive_260703.md` and
`docs/stage1_paired_r2_archive_260703_zh.md`.

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
