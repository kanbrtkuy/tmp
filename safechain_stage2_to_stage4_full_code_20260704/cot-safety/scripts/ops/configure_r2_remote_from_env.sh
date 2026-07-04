#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cloudflare_r2_cot_safety}"

required=(
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_ENDPOINT
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
done

rclone config create "$REMOTE" s3 \
  provider Cloudflare \
  env_auth false \
  access_key_id "$R2_ACCESS_KEY_ID" \
  secret_access_key "$R2_SECRET_ACCESS_KEY" \
  endpoint "$R2_ENDPOINT" \
  acl private

echo "Configured rclone remote: $REMOTE"
rclone lsd "${REMOTE}:cot-safety" >/dev/null
echo "Verified remote can list bucket: ${REMOTE}:cot-safety"
