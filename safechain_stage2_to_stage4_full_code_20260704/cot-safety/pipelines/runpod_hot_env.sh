#!/usr/bin/env bash
# Backward-compatible alias for older commands.
# New stage wrappers should source runpod_stage{1,2,3,4}_env.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runpod_base_env.sh"
