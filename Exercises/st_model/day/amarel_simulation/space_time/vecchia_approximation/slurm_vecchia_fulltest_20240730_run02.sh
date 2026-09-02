#!/bin/bash
#SBATCH --job-name=vecc_0730_r02
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc_0730_r02_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc_0730_r02_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840 || true
module load cuda/12.1.0 || true

if ! command -v conda >/dev/null 2>&1; then
  source "${HOME}/.bashrc" || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate faiss_env

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
SCRIPT="${REMOTE_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py"
SELECTION="${REMOTE_DIR}/vecchia_fulltest_20240730_run02_selection.json"
REAL_DATA_ROOT="/home/jl2815/tco/data"
SYNTHETIC_DATA_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_20240730_run02"
RUN_MODE="${RUN_MODE:-run-task}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

export PYTHONPATH="/home/jl2815/tco:${REMOTE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export MPLCONFIGDIR="${OUTPUT_ROOT}/.mplconfig_${SLURM_JOB_ID:-manual}_${TASK_INDEX}"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Run mode: ${RUN_MODE}"
echo "Task index: ${TASK_INDEX}"
echo "Output root: ${OUTPUT_ROOT}"
which python
nvidia-smi || true

python - <<'PY'
import numpy, pandas, scipy, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

if [[ "${RUN_MODE}" == "aggregate" ]]; then
  python "${SCRIPT}" \
    --mode aggregate \
    --selection-file "${SELECTION}" \
    --output-root "${OUTPUT_ROOT}" \
    --allow-partial-aggregate
else
  python "${SCRIPT}" \
    --mode run-task \
    --task-index "${TASK_INDEX}" \
    --selection-file "${SELECTION}" \
    --real-data-root "${REAL_DATA_ROOT}" \
    --synthetic-data-root "${SYNTHETIC_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --smooth 0.5 \
    --keep-exact-loc \
    --daily-stride 2 \
    --target-chunk-size 32 \
    --union-target-chunk-size 8 \
    --diag-chunk-size 64 \
    --union-diag-chunk-size 16 \
    --fit-order union,adapted,shifted,fixed \
    --lbfgs-lr 1.0 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-history 10 \
    --grad-tol 1e-5 \
    --empirical-max-lat-offset 20 \
    --empirical-max-lon-offset 20 \
    --empirical-min-pair-count 1000 \
    --empirical-smooth-bandwidth-deg 0.063 \
    --subgrid-max-condition-number 100 \
    --resample-grid 500 \
    --device cuda \
    --require-cuda \
    --suppress-fit-prints
fi

echo "Finished: $(date)"
