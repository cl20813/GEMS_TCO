#!/bin/bash
set -euo pipefail

LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PRECISION_HELPER="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/vecchia_sparse_precision_operator_090326.py"
REMOTE_HOST="jl2815@amarel-new.hpc.rutgers.edu"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643"

ssh "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}' /home/jl2815/tco/exercise_output/summer/logs"

scp -r \
  "${LOCAL_ROOT}/src/GEMS_TCO" \
  "${REMOTE_HOST}:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/vecchia_adapted_fixed_lag643_core.py" \
  "${LOCAL_DIR}/vecchia_real60_adapted_fixed_full_eigen_lag643_run10.py" \
  "${LOCAL_DIR}/vecchia_real59_adapted_fixed_threeway_slq512_lag643.py" \
  "${LOCAL_DIR}/vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643.py" \
  "${LOCAL_DIR}/slurm_vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643.sh" \
  "${PRECISION_HELPER}" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Upload complete. Submit with:"
echo "ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && sbatch slurm_vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643.sh'"
echo
echo "After completion, download with:"
echo "scp -r ${REMOTE_HOST}:${OUTPUT_ROOT} '${LOCAL_DIR}/'"
