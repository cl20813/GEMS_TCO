#!/bin/bash
set -euo pipefail

LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_HOST="jl2815@amarel-new.hpc.rutgers.edu"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"

ssh "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}' /home/jl2815/tco/exercise_output/summer/logs"

scp -r \
  "${LOCAL_ROOT}/src/GEMS_TCO" \
  "${REMOTE_HOST}:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py" \
  "${LOCAL_DIR}/vecchia_real60_adapted_fixed_full_eigen_lag643_run10.py" \
  "${LOCAL_DIR}/slurm_vecchia_real60_adapted_fixed_full_eigen_lag643_run10.sh" \
  "${LOCAL_DIR}/AMAREL_VECCHIA_GPU_OPTIMIZATION_MEMO_090326.txt" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Upload complete. Submit on Amarel with:"
echo "cd ${REMOTE_DIR}"
echo "sbatch slurm_vecchia_real60_adapted_fixed_full_eigen_lag643_run10.sh"

