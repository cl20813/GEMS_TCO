#!/bin/bash
set -euo pipefail

ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
AMAREL_HOST="${AMAREL_HOST:-jl2815@amarel-new.hpc.rutgers.edu}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/jl2815/tco/GEMS_TCO-1}"
RELATIVE_STUDY="Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826"
REMOTE_STUDY="${REMOTE_PROJECT}/${RELATIVE_STUDY}"
REMOTE_OUTPUT="/home/jl2815/tco/exercise_output/fall_26/sim_two_contrast_fitted_cross_643_092826"
REMOTE_LOG="/home/jl2815/tco/exercise_output/fall_26/logs"
LOCAL_DOWNLOAD="${SCRIPT_DIR}/downloaded_results"

FILES=(
  fitted_cross_643_config.json
  fitted_cross_643_core.py
  run_fitted_cross_643.py
  aggregate_fitted_cross_643.py
  validate_fitted_cross_643.py
  slurm_fitted_cross_643.sbatch
  submit_fitted_cross_643.sh
  README.md
)

upload() {
  ssh "${AMAREL_HOST}" "mkdir -p '${REMOTE_STUDY}' '${REMOTE_OUTPUT}' '${REMOTE_LOG}'"
  local paths=()
  local name
  for name in "${FILES[@]}"; do paths+=("${SCRIPT_DIR}/${name}"); done
  rsync -avh --progress "${paths[@]}" "${AMAREL_HOST}:${REMOTE_STUDY}/"
  # These two existing research modules are runtime dependencies of the new study.
  rsync -avh --progress \
    "${SCRIPT_DIR}/../sim_two_contrast_cross_center_092726/two_contrast_core.py" \
    "${SCRIPT_DIR}/../sim_two_contrast_cross_center_092726/run_two_contrast_diagnostic.py" \
    "${SCRIPT_DIR}/../sim_two_contrast_cross_center_092726/two_contrast_diagnostic_092726.json" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_cross_center_092726/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/simulate_data/092726/st_interaction_scenarios_092726.json" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/simulate_data/092726/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation/vecchia_adapted_fixed_lag643_core.py" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation/"
  # Keep the fitted-model implementation reproducible even if the remote
  # checkout predates the local lag-643 or generalized-Cauchy CUDA work.
  rsync -avh --progress \
    --exclude='*.so' --exclude='__pycache__/' --exclude='*.pyc' \
    "${PROJECT_ROOT}/src/GEMS_TCO/" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/src/GEMS_TCO/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/cpp/" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/cpp/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/scripts/amarel/build_and_test_cuda.sh" \
    "${PROJECT_ROOT}/scripts/amarel/cuda_covariance_smoke.py" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/scripts/amarel/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/tests/test_vecchia_cuda"*.py \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/tests/"
  rsync -avh --progress \
    "${PROJECT_ROOT}/setup.py" "${PROJECT_ROOT}/pyproject.toml" \
    "${AMAREL_HOST}:${REMOTE_PROJECT}/"
  echo "Study and exact package sources uploaded. The Slurm prologue rebuilds and verifies CUDA kernels."
}

submit() {
  local mode="$1"
  ssh "${AMAREL_HOST}" "bash '${REMOTE_STUDY}/submit_fitted_cross_643.sh' '${mode}'"
}

download() {
  local tag="${DOWNLOAD_TAG:-$(date +%Y%m%d_%H%M%S)}"
  local destination="${LOCAL_DOWNLOAD}/${tag}"
  mkdir -p "${destination}"
  rsync -avh --partial --progress "${AMAREL_HOST}:${REMOTE_OUTPUT}/" "${destination}/"
  echo "downloaded to ${destination}"
}

case "${ACTION}" in
  upload) upload ;;
  submit-pilot) submit pilot ;;
  submit-full) submit full ;;
  upload-submit-pilot) upload; submit pilot ;;
  upload-submit-full) upload; submit full ;;
  status)
    ssh "${AMAREL_HOST}" "squeue -u jl2815 -n tcfit643; sacct -X -n -S today --name tcfit643 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
    ;;
  progress)
    ssh "${AMAREL_HOST}" "test -r '${REMOTE_OUTPUT}/master_progress.json' && cat '${REMOTE_OUTPUT}/master_progress.json' || echo 'no progress yet'"
    ;;
  logs)
    ssh "${AMAREL_HOST}" "ls -lh '${REMOTE_LOG}'/tcfit643_* 2>/dev/null || true"
    ;;
  download) download ;;
  validate-local)
    PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/tcfit643_pycache" \
      /opt/anaconda3/envs/faiss_env/bin/python "${SCRIPT_DIR}/validate_fitted_cross_643.py"
    ;;
  help|*)
    echo "Usage: $(basename "$0") {validate-local|upload|submit-pilot|submit-full|upload-submit-pilot|upload-submit-full|status|progress|logs|download}"
    ;;
esac
