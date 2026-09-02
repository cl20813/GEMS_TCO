#!/bin/bash
#SBATCH --job-name=vecc_3g_643_r07
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc_3g_643_r07_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc_3g_643_r07_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%1

set -euo pipefail

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="${REMOTE_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py"
SELECTION="${REMOTE_DIR}/vecchia_real30_synth30_three_geometry_lag643_090226_selection.json"
REAL_DATA_ROOT="/home/jl2815/tco/data"
SYNTHETIC_DATA_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_three_geometry_lag643_real30_synth30_run07_nugget0_serial"
RUN_MODE="${RUN_MODE:-run-task}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# Use the existing environment directly. This intentionally avoids the broken
# Lmod/Lua posix path shown in the earlier Amarel logs.
test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }

export PYTHONPATH="/home/jl2815/tco:${REMOTE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${OUTPUT_ROOT}/.mplconfig_${SLURM_JOB_ID:-manual}_${TASK_INDEX}"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Run mode: ${RUN_MODE}"
echo "Task index: ${TASK_INDEX}"
nvidia-smi || true

if [[ "${RUN_MODE}" == "aggregate" ]]; then
  "${PYTHON_BIN}" "${SCRIPT}" \
    --mode aggregate \
    --selection-file "${SELECTION}" \
    --output-root "${OUTPUT_ROOT}"
else
  "${PYTHON_BIN}" "${SCRIPT}" \
    --mode run-task \
    --task-index "${TASK_INDEX}" \
    --selection-file "${SELECTION}" \
    --real-data-root "${REAL_DATA_ROOT}" \
    --synthetic-data-root "${SYNTHETIC_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --smooth 0.5 \
    --truth-nugget 0.0 \
    --fixed-nugget 0.0 \
    --keep-exact-loc \
    --daily-stride 2 \
    --target-chunk-size 16 \
    --union-target-chunk-size 4 \
    --diag-chunk-size 32 \
    --union-diag-chunk-size 8 \
    --fit-order union,adapted,fixed \
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
