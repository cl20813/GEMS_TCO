#!/bin/bash -l
#SBATCH --job-name=gems-cuda-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gems-cuda-smoke-%j.out

set -euo pipefail

module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GEMS_TCO_CONDA_ENV:-gems_gpu}"

repository_root="${GEMS_TCO_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
build_script="${repository_root}/scripts/amarel/build_and_test_cuda.sh"
if [[ ! -f "${build_script}" ]]; then
    echo "Cannot find ${build_script}; submit from the repository root or set GEMS_TCO_REPOSITORY_ROOT." >&2
    exit 2
fi
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
exec bash "${build_script}"
