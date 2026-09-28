#!/bin/bash -l
#SBATCH --job-name=tcrossagg092726
#SBATCH --output=/home/jl2815/tco/exercise_output/fall_26/logs/tcrossagg092726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fall_26/logs/tcrossagg092726_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=main

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jl2815/.conda/envs/faiss_env/bin/python}"
REPOSITORY_ROOT="${GEMS_TCO_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
STUDY_DIR="${REPOSITORY_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_cross_center_092726"
OUTPUT_ROOT="${GEMS_TCO_DIAGNOSTIC_OUTPUT_ROOT:-/home/jl2815/tco/exercise_output/fall_26/sim_two_contrast_cross_center_092726}"
DATA_ROOT="${GEMS_TCO_SIM_DATA_ROOT:-/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/tcrossagg_pycache_${SLURM_JOB_ID:-local}"
RUN_MODE="${RUN_MODE:-full}"
if [[ "${RUN_MODE}" == "full" ]]; then
  EXTRA_ARGS=(--require-complete)
elif [[ "${RUN_MODE}" == "pilot" ]]; then
  EXTRA_ARGS=()
else
  echo "RUN_MODE must be pilot or full; got ${RUN_MODE}" >&2
  exit 2
fi

"${PYTHON_BIN}" "${STUDY_DIR}/aggregate_two_contrast_diagnostic.py" \
  --config "${STUDY_DIR}/two_contrast_diagnostic_092726.json" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}"
