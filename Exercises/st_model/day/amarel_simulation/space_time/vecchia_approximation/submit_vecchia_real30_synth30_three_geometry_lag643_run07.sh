#!/bin/bash
set -euo pipefail

RUN_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
SLURM_FILE="slurm_vecchia_real30_synth30_three_geometry_lag643_run07.sh"

cd "${RUN_DIR}"
fit_job=$(sbatch --parsable "${SLURM_FILE}")
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${fit_job}" \
  --array=0-0 \
  --export=ALL,RUN_MODE=aggregate \
  "${SLURM_FILE}")

echo "fit_job=${fit_job}"
echo "aggregate_job=${aggregate_job}"
echo "60 data tasks; maximum concurrent GPU tasks=1"
