#!/bin/bash
#SBATCH --job-name=vecc_n400_900
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc_n400_900_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc_n400_900_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
# gpu017 returned CUDA unknown error on 2026-09-08; keep it out of this run.
#SBATCH --nodelist=gpu[015-016,019-028]
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="${REMOTE_DIR}/vecchia_one_day_sampling_convergence_full_eigen_lag643.py"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_20240703_sampling_convergence_full_eigen_lag643"
FIT_CHECKPOINT="/home/jl2815/tco/exercise_output/summer/vecchia_real59_adapted_fixed_full_eigen_lag643_clean_v2/fit_checkpoint_full_precision.json"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${SCRIPT}" || { echo "Missing script: ${SCRIPT}" >&2; exit 2; }

export PYTHONPATH="/home/jl2815/tco:${REMOTE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/jl2815_matplotlib_${SLURM_JOB_ID:-manual}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" \
  /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Date: 2024-07-03"
echo "Counts: 400, 600, 900 common spatial sites x 8 hours"
echo "Designs: nested common max-min and exact-size snapped quasi-regular"
echo "Methods: fitted adapted and fixed lag-6/4/3; parameters held fixed"
echo "Dense eigensystems: 3,200; 4,800; 7,200"
echo "SLURM job/node: ${SLURM_JOB_ID:-unset} / $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"${PYTHON_BIN}" -c 'import torch; print("torch", torch.__version__, "built CUDA", torch.version.cuda); print("CUDA available", torch.cuda.is_available(), "count", torch.cuda.device_count()); assert torch.cuda.is_available(); print("device 0", torch.cuda.get_device_name(0))'

"${PYTHON_BIN}" "${SCRIPT}" \
  --date 2024-07-03 \
  --counts=400,600,900 \
  --real-data-root /home/jl2815/tco/data \
  --output-root "${OUTPUT_ROOT}" \
  --fit-checkpoint "${FIT_CHECKPOINT}" \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --smooth 0.5 \
  --target-chunk-size 256 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 40 \
  --grad-tol 1e-5 \
  --cov-jitter 1e-8 \
  --graph-neighbors 6 \
  --suppress-fit-prints

echo "Finished: $(date)"
echo "Re-submit the same job if interrupted; completed eigensystems are cached."
