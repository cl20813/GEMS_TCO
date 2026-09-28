#!/bin/bash
set -euo pipefail

MODE="${1:-pilot}"
REPOSITORY_ROOT="${GEMS_TCO_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
SBATCH_FILE="${REPOSITORY_ROOT}/simulate_data/092726/slurm_generate_st_interaction_suite_092726.sbatch"
INPUT_FILE="${GEMS_TCO_INPUT_ROOT:-/home/jl2815/tco/data}/pickle_2024/tco_grid_24_07.pkl"
RUN_ROOT="/home/jl2815/tco/exercise_output/fall_26"
LOG_ROOT="${RUN_ROOT}/logs"
OUTPUT_ROOT="${RUN_ROOT}/st_interaction_july2024_nugget0_092726"

if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "usage: $0 [pilot|full]" >&2
  exit 2
fi

test -r "${SBATCH_FILE}"
test -r "${INPUT_FILE}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

if squeue -h -u "${USER}" -n stint092726 | grep -q .; then
  echo "A stint092726 array is already queued or running; refusing a duplicate." >&2
  squeue -u "${USER}" -n stint092726
  exit 3
fi

PARTITION="${GEMS_TCO_SIM_PARTITION:-main}"
MEMORY="${GEMS_TCO_SIM_MEMORY:-160G}"
CONCURRENCY="${GEMS_TCO_SIM_CONCURRENCY:-2}"
ARRAY="0-5%${CONCURRENCY}"

echo "Submitting ${MODE}: partition=${PARTITION}, memory=${MEMORY}, array=${ARRAY}"
sbatch \
  --partition="${PARTITION}" \
  --mem="${MEMORY}" \
  --array="${ARRAY}" \
  --export="ALL,RUN_MODE=${MODE}" \
  "${SBATCH_FILE}"
