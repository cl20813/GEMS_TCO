#!/bin/bash -l
#SBATCH --job-name=tcfit643
#SBATCH --output=/home/jl2815/tco/exercise_output/fall_26/logs/tcfit643_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fall_26/logs/tcfit643_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu[015-017,019-028]
#SBATCH --gres=gpu:1

set -uo pipefail

REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1"
STUDY_DIR="${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
DATA_ROOT="/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/fall_26/sim_two_contrast_fitted_cross_643_092826"
RUN_MODE="${RUN_MODE:-full}"
GPU_PROFILE="${GPU_PROFILE:-a100}"

module use /projects/community/modulefiles
module load cuda/12.1.0

export GEMS_TCO_SRC="${REMOTE_PROJECT}/src"
export PYTHONPATH="${REMOTE_PROJECT}/src:${STUDY_DIR}:${STUDY_DIR}/../sim_two_contrast_cross_center_092726:${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/jl2815_tcfit643_${SLURM_JOB_ID:-manual}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" /home/jl2815/tco/exercise_output/fall_26/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Run mode: ${RUN_MODE}"
echo "One sequential job, no array"
echo "Fit: matched-family joint model, nugget 0, direction-adapted lag 6/4/3"
echo "Chunks: Matérn 256, generalized Cauchy 128"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON_BIN}" -c '
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "GPU", name, "capability", major, minor)
assert (major, minor) == (8, 0) and "A100" in name.upper(), (name, major, minor)
'

# Rebuild and verify the exact native Matérn/GC covariance kernels before fits.
export PYTHON="${PYTHON_BIN}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-12}"
export GEMS_TCO_CUDA_ARCH_LIST="8.0;8.9"
bash "${REMOTE_PROJECT}/scripts/amarel/build_and_test_cuda.sh" \
  --batch-size 16 --points 224 --iterations 3
"${PYTHON_BIN}" -c \
  'from GEMS_TCO.vecchia._native_covariance import native_generalized_cauchy_covariance_available; assert native_generalized_cauchy_covariance_available("cuda")'

# Fail before any expensive daily fit if the frozen contrast schema, lag key,
# model classes, or exact Var(L) decomposition drift out of sync.
"${PYTHON_BIN}" "${STUDY_DIR}/validate_fitted_cross_643.py"

if [[ "${RUN_MODE}" == "pilot" ]]; then
  DATE_ARGS=(--dates 2024-07-01)
elif [[ "${RUN_MODE}" == "full" ]]; then
  DATE_ARGS=()
else
  echo "RUN_MODE must be pilot or full" >&2
  exit 2
fi

overall_status=0
for scenario_index in 0 1 2 3 4 5; do
  echo "===== scenario ${scenario_index} started: $(date) ====="
  if ! "${PYTHON_BIN}" "${STUDY_DIR}/run_fitted_cross_643.py" \
      --scenario-index "${scenario_index}" \
      --config "${STUDY_DIR}/fitted_cross_643_config.json" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda \
      "${DATE_ARGS[@]}"; then
    overall_status=1
    echo "Scenario ${scenario_index} contains failed dates; continuing to preserve GPU time" >&2
  fi
  echo "===== scenario ${scenario_index} finished: $(date) ====="
done

aggregate_args=()
if [[ "${RUN_MODE}" == "full" ]]; then
  aggregate_args+=(--require-complete)
fi
if ! "${PYTHON_BIN}" "${STUDY_DIR}/aggregate_fitted_cross_643.py" \
    --config "${STUDY_DIR}/fitted_cross_643_config.json" \
    --output-root "${OUTPUT_ROOT}" \
    "${aggregate_args[@]}"; then
  overall_status=1
fi

echo "Finished: $(date), status=${overall_status}"
exit "${overall_status}"
