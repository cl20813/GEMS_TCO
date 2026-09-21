#!/bin/bash
set -euo pipefail

LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_HOST="jl2815@amarel-new.hpc.rutgers.edu"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_20240703_sampling_convergence_full_eigen_lag643"

ssh "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}' /home/jl2815/tco/exercise_output/summer/logs"

scp -r \
  "${LOCAL_ROOT}/src/GEMS_TCO" \
  "${REMOTE_HOST}:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/vecchia_adapted_fixed_lag643_core.py" \
  "${LOCAL_DIR}/vecchia_real60_adapted_fixed_full_eigen_lag643_run10.py" \
  "${LOCAL_DIR}/vecchia_one_day_sampling_convergence_full_eigen_lag643.py" \
  "${LOCAL_DIR}/slurm_vecchia_one_day_sampling_convergence_full_eigen_lag643.sh" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Upload complete. Submit with:"
echo "ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && sbatch slurm_vecchia_one_day_sampling_convergence_full_eigen_lag643.sh'"
echo
echo "Download after completion with:"
echo "scp -r ${REMOTE_HOST}:${OUTPUT_ROOT} '${LOCAL_DIR}/'"
