#!/usr/bin/env bash
# Build both native covariance backends and validate the CUDA path on Amarel.

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
# The same installed extension also contains sm_80 code for A100.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
export GEMS_TCO_BUILD_TORCH_EXT=1
export GEMS_TCO_BUILD_CUDA_EXT=1

echo "Repository: ${repository_root}"
echo "Python: $(${python_executable} --version 2>&1)"
echo "nvcc: $(nvcc --version | tail -n 1)"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${python_executable}" -c \
    'import torch; assert torch.version.cuda is not None, "CPU-only PyTorch build"; print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")'

"${python_executable}" -m pip install \
    --editable "${repository_root}" \
    --no-build-isolation \
    --no-deps

"${python_executable}" -m unittest discover \
    -s "${repository_root}/tests" \
    -p 'test_vecchia_cuda_covariance.py' \
    -v

"${python_executable}" "${script_dir}/cuda_covariance_smoke.py" "$@"
