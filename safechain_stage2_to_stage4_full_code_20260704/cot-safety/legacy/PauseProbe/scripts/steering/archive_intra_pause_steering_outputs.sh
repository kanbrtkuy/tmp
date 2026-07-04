#!/usr/bin/env bash
set -euo pipefail

# Archive the intra-pause steering pilot outputs from a RunPod workspace and copy
# them directly to Google Drive via the pod's rclone config. This intentionally
# skips HF cache directories and other credential-bearing files.

REPO_ROOT="${REPO_ROOT:-/workspace/PauseProbe}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/workspace/intra_pause_steering_backup_260619_archive}"
DEST="${DEST:-safechain_gdrive:Research/PauseProbe_transfer/intra_pause_steering_260619_archive}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs/intra_pause_steering_backup_260619}"

mkdir -p "${ARCHIVE_ROOT}" "${LOG_ROOT}"

write_restore_doc() {
  cat > "${ARCHIVE_ROOT}/RESTORE.md" <<'EOF'
# Intra-Pause Steering Backup 2026-06-19

This archive stores the RunPod-side outputs for the intra-pause pause-only
steering pilot. It is intended for restoring experiment artifacts after the GPU
pod is terminated.

## Source Pod

- Workspace: `/workspace/PauseProbe`
- Main run directory: `/workspace/PauseProbe/runs/steering`
- Model checkpoint used by generation:
  `/workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final`
- Probe/data artifacts reused from the 2026-06-16 intra-pause probe backup.

## Restore

```bash
rclone copy safechain_gdrive:Research/PauseProbe_transfer/intra_pause_steering_260619_archive ./intra_pause_steering_260619_archive --transfers=16 --checkers=32 --progress
cd intra_pause_steering_260619_archive
sha256sum -c SHA256SUMS.txt
mkdir -p restored
tar -xf runs_steering_260619.tar -C restored
tar -xf scripts_steering_260619.tar -C restored
```

The full SFT model is not duplicated here if it already exists in the
COTPauseToken backup:

`safechain_gdrive:Research/COTPauseToken/intra_pause_cot3_sft_260615/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final`
EOF
}

write_manifest() {
  {
    printf "artifact\tremote_source\tnotes\n"
    printf "runs_steering_260619.tar\t%s/runs/steering\tall intra-pause steering run outputs, including learned-delta checkpoints and WildGuard judged generations\n" "${REPO_ROOT}"
    printf "scripts_steering_260619.tar\t%s/scripts/steering\tsteering scripts used to train learned deltas, generate, judge, and run multiseed sweeps\n" "${REPO_ROOT}"
    printf "RESTORE.md\t%s\trestore instructions\n" "${ARCHIVE_ROOT}"
    printf "MANIFEST.tsv\t%s\tthis manifest\n" "${ARCHIVE_ROOT}"
    printf "SHA256SUMS.txt\t%s\tchecksums for tarballs and metadata\n" "${ARCHIVE_ROOT}"
  } > "${ARCHIVE_ROOT}/MANIFEST.tsv"
}

pack_artifacts() {
  tar -cf "${ARCHIVE_ROOT}/runs_steering_260619.tar" -C "${REPO_ROOT}" runs/steering
  tar -cf "${ARCHIVE_ROOT}/scripts_steering_260619.tar" -C "${REPO_ROOT}" scripts/steering
  (
    cd "${ARCHIVE_ROOT}"
    sha256sum RESTORE.md MANIFEST.tsv runs_steering_260619.tar scripts_steering_260619.tar > SHA256SUMS.txt
  )
}

copy_to_gdrive() {
  rclone mkdir "${DEST}"
  rclone copy "${ARCHIVE_ROOT}" "${DEST}" \
    --transfers=16 \
    --checkers=32 \
    --buffer-size=64M \
    --drive-chunk-size=128M \
    --drive-upload-cutoff=128M \
    --drive-acknowledge-abuse \
    --drive-allow-import-name-change \
    --tpslimit=10 \
    --tpslimit-burst=20 \
    --stats=10s \
    --progress
  rclone check "${ARCHIVE_ROOT}" "${DEST}" \
    --one-way \
    --log-file="${LOG_ROOT}/rclone_check.log" \
    --log-level=INFO
  rclone size "${DEST}" | tee "${LOG_ROOT}/gdrive_size.txt"
}

write_restore_doc
write_manifest
pack_artifacts
copy_to_gdrive

echo "Backup complete: ${DEST}"
