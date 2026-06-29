#!/usr/bin/env bash
# RunPod environment for Stage 2 SFT.
#
# Stage 2 is intentionally separate from Stage 1 because it relies on the SFT
# dependency stack: TRL, torch.distributed, and the optimizer implementation
# selected by config.  The default configs use paged_adamw_8bit, so this file
# checks that bitsandbytes can load its native CUDA library before a DDP launch.

if [[ -n "${COT_SAFETY_STAGE2_ENV_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export COT_SAFETY_STAGE2_ENV_SOURCED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runpod_base_env.sh"

export COT_SAFETY_STAGE="stage2"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# pip wheels for CUDA libraries sometimes place nvJitLink under
# site-packages/nvidia/nvjitlink/lib.  Add it when present; this is harmless if
# the runtime uses system CUDA instead.
if command -v python3 >/dev/null 2>&1; then
  nvjitlink_dir="$(
    python3 - <<'PY' 2>/dev/null || true
from pathlib import Path
import site

for root in site.getsitepackages() + [site.getusersitepackages()]:
    path = Path(root) / "nvidia" / "nvjitlink" / "lib"
    if path.exists():
        print(path)
        break
PY
  )"
  if [[ -n "${nvjitlink_dir}" ]]; then
    export LD_LIBRARY_PATH="${nvjitlink_dir}:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ "${COT_SAFETY_STAGE2_CHECK_BNB:-1}" == "1" ]]; then
  python_bin="${PYTHON:-python3}"
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Stage2 env check failed: PYTHON=${python_bin} is not available." >&2
    return 1 2>/dev/null || exit 1
  fi
  if ! "${python_bin}" - <<'PY'
import sys

try:
    import bitsandbytes.cextension as cextension
except Exception as exc:
    raise SystemExit(f"bitsandbytes import failed: {exc}") from exc

native_lib = getattr(cextension, "lib", None)
if native_lib is None:
    raise SystemExit("bitsandbytes native CUDA library did not load")
if native_lib.__class__.__name__ == "ErrorHandlerMockBNBNativeLibrary":
    raise SystemExit("bitsandbytes native CUDA library is an error-handler mock")
if getattr(native_lib, "compiled_with_cuda", False) is False:
    raise SystemExit("bitsandbytes native CUDA library was not compiled with CUDA")

print("bitsandbytes native CUDA library loaded")
PY
  then
    cat >&2 <<'EOF'
Stage2 env check failed: bitsandbytes is not usable.

The default Stage2 configs use OPTIM=paged_adamw_8bit, so fix the SFT
environment before launching training.  Do not switch optimizers unless the
experiment config explicitly asks for that change.

Set COT_SAFETY_STAGE2_CHECK_BNB=0 only for configs that do not use bitsandbytes.
EOF
    return 1 2>/dev/null || exit 1
  fi
fi
