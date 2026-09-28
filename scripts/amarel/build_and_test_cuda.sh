#!/usr/bin/env bash
# Build the native CUDA covariance backend and validate it on Amarel.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
python_executable="${PYTHON:-python}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is unavailable; load the Amarel CUDA module before running this script." >&2
    exit 2
fi

if ! command -v "${python_executable}" >/dev/null 2>&1; then
    echo "Python executable '${python_executable}' is unavailable." >&2
    exit 2
fi

# CUDA 11.8 or newer is required to compile a native sm_89 cubin for L40S.
# The same installed extension also contains sm_80 code for A100.  Do not
# inherit a stale site/user TORCH_CUDA_ARCH_LIST (for example, 8.0 only): the
# production binary must be portable across both project GPU families.
export TORCH_CUDA_ARCH_LIST="${GEMS_TCO_CUDA_ARCH_LIST:-8.0;8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
# The local macOS workflow builds the CPU extension separately.  Amarel needs
# the Linux/CUDA binary; avoid spending the GPU-job prologue on an unused CPU
# covariance extension unless explicitly requested.
export GEMS_TCO_BUILD_TORCH_EXT="${GEMS_TCO_BUILD_TORCH_EXT:-0}"
export GEMS_TCO_BUILD_CUDA_EXT=1

echo "Repository: ${repository_root}"
echo "Python: $(${python_executable} --version 2>&1)"
echo "nvcc: $(nvcc --version | tail -n 1)"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${python_executable}" -c \
    'import torch; assert torch.version.cuda is not None, "CPU-only PyTorch build"; print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")'
"${python_executable}" -c '
import setuptools

try:
    major = int(setuptools.__version__.split(".", 1)[0])
except (AttributeError, ValueError) as error:
    raise SystemExit(f"cannot parse Setuptools version: {setuptools.__version__!r}") from error
if major < 77:
    raise SystemExit(
        "Setuptools >=77 is required for this project metadata and the "
        f"non-isolated CUDA build; found {setuptools.__version__}. "
        "Run the package-first upload helper before submitting the job."
    )
print("Setuptools", setuptools.__version__)
'

# Architecture flags alone are not always considered by incremental distutils
# rebuild checks.  Force recompilation so an older sm_80-only object or shared
# library can never be reused for an L40S run.
(
    cd "${repository_root}"
    "${python_executable}" setup.py build_ext --inplace --force
)

"${python_executable}" -m pip install \
    --editable "${repository_root}" \
    --no-build-isolation \
    --no-deps

"${python_executable}" -m unittest discover \
    -s "${repository_root}/tests" \
    -p 'test_vecchia_cuda*.py' \
    -v

"${python_executable}" "${script_dir}/cuda_covariance_smoke.py" "$@"
