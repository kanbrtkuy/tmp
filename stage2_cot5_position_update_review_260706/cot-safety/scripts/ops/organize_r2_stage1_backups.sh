#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cloudflare_r2_cot_safety}"
BUCKET_PREFIX="${BUCKET_PREFIX:-cot-safety}"
DATE_TAG="${DATE_TAG:-20260701-a6000}"
OLD_1P5B="${OLD_1P5B:-runpod-backups/20260701-stage1-1p5b-a6000-tarstream}"
OLD_8B="${OLD_8B:-runpod-backups/20260701-stage1-8b-a6000-stage1}"
NEW_ROOT="${NEW_ROOT:-stage1/${DATE_TAG}}"
TRANSFERS="${TRANSFERS:-16}"
CHECKERS="${CHECKERS:-32}"
S3_CHUNK_SIZE="${S3_CHUNK_SIZE:-1G}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_OLD="${DELETE_OLD:-0}"

src() {
  printf '%s:%s/%s' "$REMOTE" "$BUCKET_PREFIX" "$1"
}

dst() {
  printf '%s:%s/%s' "$REMOTE" "$BUCKET_PREFIX" "$1"
}

copy_dir() {
  local from="$1"
  local to="$2"
  shift 2
  echo "== copy =="
  echo "from: $(src "$from")"
  echo "to:   $(dst "$to")"
  local args=(copy "$(src "$from")" "$(dst "$to")"
    --fast-list
    --transfers="$TRANSFERS"
    --checkers="$CHECKERS"
    --s3-chunk-size="$S3_CHUNK_SIZE"
    --stats=30s
    --stats-one-line
    "$@"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  rclone "${args[@]}"
}

size_path() {
  local path="$1"
  echo "== size: $path =="
  rclone size "$(dst "$path")" --fast-list || true
}

echo "Remote: $REMOTE"
echo "Destination root: $(dst "$NEW_ROOT")"
echo "Dry run: $DRY_RUN"

copy_dir "$OLD_1P5B/data" "${NEW_ROOT}/deepseek-1p5b/data"
copy_dir "$OLD_1P5B/runs" "${NEW_ROOT}/deepseek-1p5b/runs/results" --include "/*.tar"
copy_dir "$OLD_1P5B/runs/hidden" "${NEW_ROOT}/deepseek-1p5b/runs/hidden"
copy_dir "$OLD_1P5B/runs/logs" "${NEW_ROOT}/deepseek-1p5b/runs/logs"
copy_dir "$OLD_8B/runs_hidden" "${NEW_ROOT}/deepseek-8b/runs/hidden"

echo "== final sizes =="
size_path "$NEW_ROOT/deepseek-1p5b"
size_path "$NEW_ROOT/deepseek-8b"
size_path "$NEW_ROOT"

if [[ "$DELETE_OLD" == "1" && "$DRY_RUN" != "1" ]]; then
  echo "DELETE_OLD=1: deleting old backup prefixes after copy."
  rclone purge "$(src "$OLD_1P5B")"
  rclone purge "$(src "$OLD_8B")"
else
  echo "Old prefixes preserved. Set DELETE_OLD=1 after manual verification to remove them."
fi
