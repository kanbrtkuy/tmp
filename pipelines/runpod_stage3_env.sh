#!/usr/bin/env bash
# RunPod environment for Stage 3 intra-pause probe.
#
# Stage 3 reads an SFT checkpoint, extracts hidden states around pause tokens,
# and trains probe models.  It needs the base inference/probe stack but does
# not require the Stage 2 SFT optimizer stack.

if [[ -n "${COT_SAFETY_STAGE3_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export COT_SAFETY_STAGE3_ENV_SOURCED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runpod_base_env.sh"

export COT_SAFETY_STAGE="${COT_SAFETY_STAGE:-stage3}"
export COT_SAFETY_LEGACY_ROOT="${COT_SAFETY_LEGACY_ROOT:-${COT_SAFETY_REPO_ROOT}/legacy/PauseProbe}"

if [[ "${COT_SAFETY_STAGE3_CHECK_IMPORTS:-1}" == "1" ]]; then
  python_bin="${PYTHON:-python3}"
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Stage3 env check failed: PYTHON=${python_bin} is not available." >&2
    return 1 2>/dev/null || exit 1
  fi
  if ! "${python_bin}" - <<'PY'
missing = []
for module in ("torch", "transformers", "numpy", "sklearn"):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("missing Stage3 dependency/import: " + "; ".join(missing))
print("Stage3 probe imports OK")
PY
  then
    return 1 2>/dev/null || exit 1
  fi
fi
