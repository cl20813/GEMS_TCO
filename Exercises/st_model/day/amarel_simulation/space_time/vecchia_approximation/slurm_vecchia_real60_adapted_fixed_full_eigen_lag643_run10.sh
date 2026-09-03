#!/bin/bash
#SBATCH --job-name=vecc_real60_r10
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc_real60_r10_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc_real60_r10_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu[015-017,019-028]
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="${REMOTE_DIR}/vecchia_real60_adapted_fixed_full_eigen_lag643_run10.py"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_real60_adapted_fixed_full_eigen_lag643_run10"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${SCRIPT}" || { echo "Missing script: ${SCRIPT}" >&2; exit 2; }

export PYTHONPATH="/home/jl2815/tco:${REMOTE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${OUTPUT_ROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" \
  /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Dates: 2024-07-01..30 and 2025-07-01..30 (60 total)"
echo "Methods: adapted, fixed; union excluded"
echo "Fit chunk: 256 blocks; full eigen: 400 x 8 = 3200 points"
nvidia-smi || true

srun "${PYTHON_BIN}" "${SCRIPT}" \
  --real-data-root /home/jl2815/tco/data \
  --output-root "${OUTPUT_ROOT}" \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --smooth 0.5 \
  --target-chunk-size 256 \
  --points-per-hour 400 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 40 \
  --grad-tol 1e-5 \
  --cov-jitter 1e-8 \
  --suppress-fit-prints

echo "Finished: $(date)"

