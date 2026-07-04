#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-safechain_gdrive:Research/cot-safety/runpod_backups}"
COT_SAFETY_COLD_ROOT="${COT_SAFETY_COLD_ROOT:-/workspace}"
RESTORE_DIR="${RESTORE_DIR:-${COT_SAFETY_COLD_ROOT}/restore_archives}"
BACKUP_ID=""
ARCHIVES=()
UNPACK=1
NORMALIZE_WORKSPACE_PREFIX=1
RESTORE_DIR_USER_SET=0

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/runpod_restore_gdrive_backup.sh --backup-id ID [options]

Options:
  --backup-id ID       Backup folder under $REMOTE_ROOT.
  --archive NAME       Archive to restore from archives/. May be repeated.
                       Default: data.tar.gz.
  --remote-root PATH   Rclone remote backup root.
                       Default: safechain_gdrive:Research/cot-safety/runpod_backups
  --cold-root PATH     Persistent cold root. Default: /workspace.
  --restore-dir PATH   Where archives are stored on cold root.
                       Default: $COT_SAFETY_COLD_ROOT/restore_archives.
  --archive-only       Copy archive(s) to cold root but do not unpack.
  --no-normalize       Do not move a leading workspace/ prefix into cold root.
  --list               List available archives for the backup and exit.

Examples:
  bash pipelines/runpod_restore_gdrive_backup.sh \
    --backup-id 20260624T235259Z_deepseek8b_all_sft_ckpts_no_models

  bash pipelines/runpod_restore_gdrive_backup.sh \
    --backup-id 20260624T235259Z_deepseek8b_all_sft_ckpts_no_models \
    --archive data.tar.gz \
    --archive logs.tar.gz
USAGE
}

require_backup_id() {
  if [[ -z "${BACKUP_ID}" ]]; then
    echo "missing --backup-id" >&2
    usage >&2
    exit 2
  fi
}

copy_dir_contents() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}/." "${dst}/"
  fi
}

normalize_workspace_prefix() {
  local nested="${COT_SAFETY_COLD_ROOT}/workspace"
  [[ "${NORMALIZE_WORKSPACE_PREFIX}" == "1" ]] || return 0
  [[ -d "${nested}" ]] || return 0

  for name in data outputs runs cot-safety; do
    if [[ -e "${nested}/${name}" ]]; then
      if [[ -e "${COT_SAFETY_COLD_ROOT}/${name}" ]]; then
        echo "merging ${nested}/${name} -> ${COT_SAFETY_COLD_ROOT}/${name}"
        copy_dir_contents "${nested}/${name}" "${COT_SAFETY_COLD_ROOT}/${name}"
        rm -rf "${nested:?}/${name}"
      else
        echo "moving ${nested}/${name} -> ${COT_SAFETY_COLD_ROOT}/${name}"
        mv "${nested}/${name}" "${COT_SAFETY_COLD_ROOT}/${name}"
      fi
    fi
  done
  rmdir "${nested}" 2>/dev/null || true
}

copy_archive() {
  local archive="$1"
  local remote="${REMOTE_ROOT}/${BACKUP_ID}/archives/${archive}"
  local local_dir="${RESTORE_DIR}/${BACKUP_ID}"
  local local_path="${local_dir}/${archive}"

  mkdir -p "${local_dir}"
  if [[ -s "${local_path}" ]]; then
    echo "archive already exists: ${local_path}"
  else
    echo "copying ${remote} -> ${local_dir}/"
    rclone copy \
      --transfers="${RCLONE_TRANSFERS:-16}" \
      --checkers="${RCLONE_CHECKERS:-32}" \
      --buffer-size="${RCLONE_BUFFER_SIZE:-64M}" \
      --drive-chunk-size="${RCLONE_DRIVE_CHUNK_SIZE:-256M}" \
      --drive-upload-cutoff="${RCLONE_DRIVE_UPLOAD_CUTOFF:-256M}" \
      --stats="${RCLONE_STATS:-5s}" \
      --progress \
      "${remote}" \
      "${local_dir}/"
  fi

  if [[ "${UNPACK}" == "1" ]]; then
    echo "unpacking ${local_path} -> ${COT_SAFETY_COLD_ROOT}"
    tar --no-same-owner -xzf "${local_path}" -C "${COT_SAFETY_COLD_ROOT}"
    normalize_workspace_prefix
  fi
}

LIST_ONLY=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --backup-id)
      shift
      BACKUP_ID="$1"
      ;;
    --archive)
      shift
      ARCHIVES+=("$1")
      ;;
    --remote-root)
      shift
      REMOTE_ROOT="$1"
      ;;
    --cold-root)
      shift
      COT_SAFETY_COLD_ROOT="$1"
      if [[ "${RESTORE_DIR_USER_SET}" == "0" ]]; then
        RESTORE_DIR="${COT_SAFETY_COLD_ROOT}/restore_archives"
      fi
      ;;
    --restore-dir)
      shift
      RESTORE_DIR="$1"
      RESTORE_DIR_USER_SET=1
      ;;
    --archive-only)
      UNPACK=0
      ;;
    --no-normalize)
      NORMALIZE_WORKSPACE_PREFIX=0
      ;;
    --list)
      LIST_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

require_backup_id

if [[ "${LIST_ONLY}" == "1" ]]; then
  rclone lsf "${REMOTE_ROOT}/${BACKUP_ID}/archives"
  exit 0
fi

if [[ "${#ARCHIVES[@]}" -eq 0 ]]; then
  ARCHIVES=(data.tar.gz)
fi

mkdir -p "${COT_SAFETY_COLD_ROOT}" "${RESTORE_DIR}/${BACKUP_ID}"
for archive in "${ARCHIVES[@]}"; do
  copy_archive "${archive}"
done

echo "GDrive restore complete."
echo "Cold root: ${COT_SAFETY_COLD_ROOT}"
echo "Archive dir: ${RESTORE_DIR}/${BACKUP_ID}"
