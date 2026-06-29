#!/usr/bin/env bash
# RunPod environment for Stage 4 steering generation and judging.
#
# Stage 4 generation uses custom forward hooks for hidden-state steering, so it
# uses the HF/PyTorch generation stack.  Judge/rescore phases can additionally
# use vLLM, but vLLM is optional because hook-based steering generation should
# not require it.

if [[ -n "${COT_SAFETY_STAGE4_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export COT_SAFETY_STAGE4_ENV_SOURCED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runpod_base_env.sh"

export COT_SAFETY_STAGE="${COT_SAFETY_STAGE:-stage4}"
export DEVICES="${DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICES}}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export COT_SAFETY_LEGACY_ROOT="${COT_SAFETY_LEGACY_ROOT:-${COT_SAFETY_REPO_ROOT}/legacy/PauseProbe}"

if [[ "${COT_SAFETY_STAGE4_CHECK_IMPORTS:-1}" == "1" ]]; then
  python_bin="${PYTHON:-python3}"
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Stage4 env check failed: PYTHON=${python_bin} is not available." >&2
    return 1 2>/dev/null || exit 1
  fi
  if ! "${python_bin}" - <<'PY'
missing = []
for module in ("torch", "transformers", "numpy"):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("missing Stage4 dependency/import: " + "; ".join(missing))
print("Stage4 steering imports OK")
PY
  then
    return 1 2>/dev/null || exit 1
  fi
fi

if [[ "${COT_SAFETY_STAGE4_REQUIRE_VLLM:-0}" == "1" ]]; then
  python_bin="${PYTHON:-python3}"
  if ! "${python_bin}" - <<'PY'
try:
    import vllm  # noqa: F401
except Exception as exc:
    raise SystemExit(f"vLLM import failed: {exc}") from exc
print("Stage4 vLLM import OK")
PY
  then
    cat >&2 <<'EOF'
Stage4 env check failed: vLLM is not usable.

Set COT_SAFETY_STAGE4_REQUIRE_VLLM=1 only for judge/rescore phases that are
supposed to use vLLM.  Hook-based steering generation should not require vLLM.
EOF
    return 1 2>/dev/null || exit 1
  fi
fi
