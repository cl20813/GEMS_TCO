#!/bin/bash -l
#SBATCH --job-name=tcross092726
#SBATCH --output=/home/jl2815/tco/exercise_output/fall_26/logs/tcross092726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fall_26/logs/tcross092726_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=main

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jl2815/.conda/envs/faiss_env/bin/python}"
REPOSITORY_ROOT="${GEMS_TCO_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
STUDY_DIR="${REPOSITORY_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_cross_center_092726"
RUNNER="${STUDY_DIR}/run_two_contrast_diagnostic.py"
VALIDATOR="${STUDY_DIR}/validate_two_contrast_diagnostic.py"
AGGREGATOR="${STUDY_DIR}/aggregate_two_contrast_diagnostic.py"
CONFIG="${STUDY_DIR}/two_contrast_diagnostic_092726.json"
DATA_ROOT="${GEMS_TCO_SIM_DATA_ROOT:-/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726}"
OUTPUT_ROOT="${GEMS_TCO_DIAGNOSTIC_OUTPUT_ROOT:-/home/jl2815/tco/exercise_output/fall_26/sim_two_contrast_cross_center_092726}"
RUN_MODE="${RUN_MODE:-full}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/tcross_pycache_${SLURM_JOB_ID:-local}"

test -x "${PYTHON_BIN}"
test -r "${RUNNER}"
test -r "${VALIDATOR}"
test -r "${AGGREGATOR}"
test -r "${CONFIG}"
test -d "${DATA_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${RUN_MODE}" == "pilot" ]]; then
  EXTRA_ARGS=(--dates 2024-07-01)
  AGGREGATE_ARGS=()
elif [[ "${RUN_MODE}" == "full" ]]; then
  EXTRA_ARGS=()
  AGGREGATE_ARGS=(--require-complete)
else
  echo "RUN_MODE must be pilot or full; got ${RUN_MODE}" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "job=${SLURM_JOB_ID:-none} sequential_scenarios=0-5"
echo "mode=${RUN_MODE} data=${DATA_ROOT} output=${OUTPUT_ROOT}"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown} memory=${SLURM_MEM_PER_NODE:-unknown}MB"

"${PYTHON_BIN}" "${VALIDATOR}" --config "${CONFIG}"

for SCENARIO_INDEX in 0 1 2 3 4 5; do
  echo "scenario_index=${SCENARIO_INDEX} started=$(date --iso-8601=seconds)"
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
    "${PYTHON_BIN}" "${RUNNER}" \
    --config "${CONFIG}" \
    --scenario-index "${SCENARIO_INDEX}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    "${EXTRA_ARGS[@]}"
  echo "scenario_index=${SCENARIO_INDEX} finished=$(date --iso-8601=seconds)"
done

echo "aggregation started=$(date --iso-8601=seconds)"
srun --ntasks=1 --cpus-per-task=1 \
  "${PYTHON_BIN}" "${AGGREGATOR}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "${AGGREGATE_ARGS[@]}"
echo "aggregation finished=$(date --iso-8601=seconds)"

echo "finished=$(date --iso-8601=seconds)"
