#!/bin/bash -l
#SBATCH --job-name=stint092726
#SBATCH --output=/home/jl2815/tco/exercise_output/fall_26/logs/stint092726_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fall_26/logs/stint092726_%A_%a.err
#SBATCH --array=0-5%2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=main

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jl2815/.conda/envs/faiss_env/bin/python}"
REPOSITORY_ROOT="${GEMS_TCO_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
SCRIPT="${REPOSITORY_ROOT}/simulate_data/092726/generate_st_interaction_suite_092726.py"
CONFIG="${REPOSITORY_ROOT}/simulate_data/092726/st_interaction_scenarios_092726.json"
INPUT_ROOT="${GEMS_TCO_INPUT_ROOT:-/home/jl2815/tco/data}"
FULL_OUTPUT_ROOT="/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726"
RUN_MODE="${RUN_MODE:-full}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [[ "${RUN_MODE}" == "pilot" ]]; then
  OUTPUT_ROOT="${FULL_OUTPUT_ROOT}_pilot"
  # The gate must exercise the exact production grid and spectral correction;
  # only the number of independent day replicates is reduced.
  EXTRA_ARGS=(--n-days 1 --axis-n-days 30)
elif [[ "${RUN_MODE}" == "full" ]]; then
  OUTPUT_ROOT="${FULL_OUTPUT_ROOT}"
  EXTRA_ARGS=()
else
  echo "RUN_MODE must be pilot or full; got ${RUN_MODE}" >&2
  exit 2
fi

test -x "${PYTHON_BIN}"
test -r "${SCRIPT}"
test -r "${CONFIG}"
test -r "${INPUT_ROOT}/pickle_2024/tco_grid_24_07.pkl"
mkdir -p "${OUTPUT_ROOT}"

echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "job=${SLURM_JOB_ID:-none} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "mode=${RUN_MODE} output=${OUTPUT_ROOT}"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown} memory=${SLURM_MEM_PER_NODE:-unknown}MB"

"${PYTHON_BIN}" - <<'PY'
import sys
import numpy
import pandas
import scipy

print("python", sys.executable, sys.version, flush=True)
print("numpy", numpy.__version__, flush=True)
print("pandas", pandas.__version__, flush=True)
print("scipy", scipy.__version__, flush=True)
PY

# The preflight resolves the requested dates and reports the exact embedding
# and memory estimate without allocating the FFT cube.
"${PYTHON_BIN}" "${SCRIPT}" \
  --config "${CONFIG}" \
  --scenario-index "${SLURM_ARRAY_TASK_ID}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  --preflight-only

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
  --config "${CONFIG}" \
  --scenario-index "${SLURM_ARRAY_TASK_ID}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}"

echo "finished=$(date --iso-8601=seconds)"
