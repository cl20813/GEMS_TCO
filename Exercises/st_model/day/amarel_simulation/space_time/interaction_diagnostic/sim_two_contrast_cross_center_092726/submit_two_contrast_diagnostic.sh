#!/bin/bash
set -euo pipefail

MODE="${1:-pilot}"
REPOSITORY_ROOT="${GEMS_TCO_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
STUDY_DIR="${REPOSITORY_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_cross_center_092726"
JOB_FILE="${STUDY_DIR}/slurm_two_contrast_diagnostic.sbatch"
DATA_ROOT="${GEMS_TCO_SIM_DATA_ROOT:-/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726}"
OUTPUT_ROOT="${GEMS_TCO_DIAGNOSTIC_OUTPUT_ROOT:-/home/jl2815/tco/exercise_output/fall_26/sim_two_contrast_cross_center_092726}"
LOG_ROOT="/home/jl2815/tco/exercise_output/fall_26/logs"

if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "usage: $0 [pilot|full]" >&2
  exit 2
fi

test -r "${JOB_FILE}"
test -d "${DATA_ROOT}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

if squeue -h -u "${USER}" -n tcross092726 | grep -q .; then
  echo "A tcross092726 job is already queued or running; refusing a duplicate." >&2
  squeue -u "${USER}" -n tcross092726
  exit 3
fi

MAIN_JOB_ID="$({ sbatch \
  --parsable \
  --export="ALL,RUN_MODE=${MODE},GEMS_TCO_SIM_DATA_ROOT=${DATA_ROOT},GEMS_TCO_DIAGNOSTIC_OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "${JOB_FILE}"; } | cut -d';' -f1)"

echo "submitted mode=${MODE} single_job=${MAIN_JOB_ID}"
echo "output=${OUTPUT_ROOT}"
echo "logs=${LOG_ROOT}/tcross092726_${MAIN_JOB_ID}.out"
