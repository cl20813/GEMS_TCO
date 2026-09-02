#!/bin/bash
set -euo pipefail

LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_HOST="jl2815@amarel.rutgers.edu"
HOST_KEY_ALIAS="amarel-new.hpc.rutgers.edu"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"

ssh -o HostKeyAlias="${HOST_KEY_ALIAS}" "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}' /home/jl2815/tco/exercise_output/summer/logs"

scp -r -o HostKeyAlias="${HOST_KEY_ALIAS}" \
  "${LOCAL_ROOT}/src/GEMS_TCO" \
  "${REMOTE_HOST}:/home/jl2815/tco/"

scp -o HostKeyAlias="${HOST_KEY_ALIAS}" \
  "${LOCAL_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py" \
  "${LOCAL_DIR}/vecchia_real30_synth30_three_geometry_lag643_090226_selection.json" \
  "${LOCAL_DIR}/slurm_vecchia_real30_synth30_three_geometry_lag643_run07.sh" \
  "${LOCAL_DIR}/submit_vecchia_real30_synth30_three_geometry_lag643_run07.sh" \
  "${LOCAL_DIR}/README_vecchia_real30_synth30_three_geometry_lag643_run07.md" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Upload complete. No job was submitted."
