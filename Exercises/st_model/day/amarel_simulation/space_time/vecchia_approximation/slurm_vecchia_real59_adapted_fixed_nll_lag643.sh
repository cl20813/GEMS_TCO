#!/bin/bash
#SBATCH --job-name=vecc59_nll_rerun
#SBATCH --output=/home/jl2815/tco/exercise_output/fall_26/logs/vecc59_nll_rerun_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fall_26/logs/vecc59_nll_rerun_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
# gpu017 returned a CUDA initialization error in earlier runs.
#SBATCH --nodelist=gpu[015-016,019-028]
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1-vecchia-rerun-20260927"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="${REMOTE_DIR}/vecchia_real59_adapted_fixed_nll_lag643.py"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/fall_26/vecchia_real59_adapted_fixed_nll_lag643_rerun_20260927"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${SCRIPT}" || { echo "Missing script: ${SCRIPT}" >&2; exit 2; }

export GEMS_TCO_SRC="${REMOTE_PROJECT}/src"
export PYTHONPATH="${GEMS_TCO_SRC}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/jl2815_matplotlib_${SLURM_JOB_ID:-manual}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" \
  /home/jl2815/tco/exercise_output/fall_26/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Dates: July 1--30 in 2024 and 2025; 2025-07-24 excluded (59 total)"
echo "Methods: adapted and fixed lag-6/4/3"
echo "Comparison: native Vecchia NLL on each method's own conditioning graph"
echo "Eigen/SLQ/Lanczos/Ritz diagnostics: disabled"
echo "Fresh rerun: no earlier checkpoint is imported"
echo "Output: ${OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" -c \
  'import pathlib, torch, GEMS_TCO; from GEMS_TCO.data.loading import ProcessedDataLoader; from GEMS_TCO.vecchia.corridor_neighbors import DirectionalLag643CorridorVecchia; from GEMS_TCO import _maxmin; root=pathlib.Path(GEMS_TCO.__file__).resolve().parent; expected=pathlib.Path("/home/jl2815/tco/GEMS_TCO-1-vecchia-rerun-20260927/src/GEMS_TCO").resolve(); assert root == expected, (root, expected); assert torch.cuda.is_available(); print("GEMS_TCO:", root); print("data loader:", pathlib.Path(__import__("GEMS_TCO.data.loading", fromlist=["x"]).__file__).resolve()); print("maxmin:", pathlib.Path(_maxmin.__file__).resolve()); print("GPU:", torch.cuda.get_device_name(0)); print("CUDA:", torch.version.cuda)'

"${PYTHON_BIN}" "${SCRIPT}" \
  --real-data-root /home/jl2815/tco/data \
  --output-root "${OUTPUT_ROOT}" \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --smooth 0.5 \
  --target-chunk-size 256 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 40 \
  --grad-tol 1e-5 \
  --device cuda \
  --suppress-fit-prints

echo "Finished: $(date)"
echo "Re-submit this job after interruption; completed date/method fits are reused."
