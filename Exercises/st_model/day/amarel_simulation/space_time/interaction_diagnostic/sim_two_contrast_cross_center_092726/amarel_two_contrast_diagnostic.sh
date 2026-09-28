#!/bin/bash
set -euo pipefail

ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMAREL_HOST="${AMAREL_HOST:-jl2815@amarel-new.hpc.rutgers.edu}"
REMOTE_REPOSITORY_ROOT="${REMOTE_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
RELATIVE_STUDY_DIR="Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_cross_center_092726"
REMOTE_STUDY_DIR="${REMOTE_REPOSITORY_ROOT}/${RELATIVE_STUDY_DIR}"
REMOTE_RUN_ROOT="/home/jl2815/tco/exercise_output/fall_26"
REMOTE_OUTPUT_ROOT="${REMOTE_RUN_ROOT}/sim_two_contrast_cross_center_092726"
REMOTE_LOG_ROOT="${REMOTE_RUN_ROOT}/logs"
LOCAL_DOWNLOAD_ROOT="${SCRIPT_DIR}/downloaded_results"

FILES=(
  two_contrast_diagnostic_092726.json
  two_contrast_core.py
  run_two_contrast_diagnostic.py
  aggregate_two_contrast_diagnostic.py
  validate_two_contrast_diagnostic.py
  slurm_two_contrast_diagnostic.sbatch
  slurm_aggregate_two_contrast.sbatch
  submit_two_contrast_diagnostic.sh
  README.md
)

upload() {
  ssh "${AMAREL_HOST}" \
    "mkdir -p '${REMOTE_STUDY_DIR}' '${REMOTE_LOG_ROOT}' '${REMOTE_OUTPUT_ROOT}'"
  local paths=()
  local name
  for name in "${FILES[@]}"; do
    paths+=("${SCRIPT_DIR}/${name}")
  done
  rsync -avh --progress "${paths[@]}" "${AMAREL_HOST}:${REMOTE_STUDY_DIR}/"
}

submit_mode() {
  local mode="$1"
  ssh "${AMAREL_HOST}" \
    "GEMS_TCO_REPOSITORY_ROOT='${REMOTE_REPOSITORY_ROOT}' bash '${REMOTE_STUDY_DIR}/submit_two_contrast_diagnostic.sh' '${mode}'"
}

download_results() {
  local tag="${DOWNLOAD_TAG:-$(date +%Y%m%d_%H%M%S)}"
  local destination="${LOCAL_DOWNLOAD_ROOT}/${tag}"
  mkdir -p "${destination}"
  rsync -avh --partial --progress \
    "${AMAREL_HOST}:${REMOTE_OUTPUT_ROOT}/" "${destination}/"
  echo "downloaded to ${destination}"
}

case "${ACTION}" in
  upload)
    upload
    ;;
  submit-pilot)
    submit_mode pilot
    ;;
  submit-full)
    submit_mode full
    ;;
  upload-submit-pilot)
    upload
    submit_mode pilot
    ;;
  upload-submit-full)
    upload
    submit_mode full
    ;;
  status)
    ssh "${AMAREL_HOST}" \
      "squeue -u jl2815 -n tcross092726; sacct -X -n -S today --name tcross092726 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
    ;;
  logs)
    ssh "${AMAREL_HOST}" \
      "ls -lh '${REMOTE_LOG_ROOT}'/tcross092726_* 2>/dev/null || true"
    ;;
  progress)
    ssh "${AMAREL_HOST}" \
      "test -r '${REMOTE_OUTPUT_ROOT}/master_progress.json' && cat '${REMOTE_OUTPUT_ROOT}/master_progress.json' || echo 'no progress file yet'"
    ;;
  download)
    download_results
    ;;
  validate-local)
    PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/tcross_validate_pycache" \
      /opt/anaconda3/envs/faiss_env/bin/python \
      "${SCRIPT_DIR}/validate_two_contrast_diagnostic.py"
    ;;
  help|*)
    cat <<EOF
Usage: $(basename "$0") ACTION

Actions:
  upload               copy the diagnostic files to the Amarel checkout
  submit-pilot         run July 1 for all six scenarios
  submit-full          run/restart all 30 days for all six scenarios
  upload-submit-pilot  upload and submit the one-day gate
  upload-submit-full   upload and submit/restart the full study
  status               show Slurm queue and accounting status
  logs                 list diagnostic log files
  progress             print the current master progress JSON
  download             copy the current output into a timestamped local folder
  validate-local       run deterministic analytic Q_A/Q_B regression tests

Override AMAREL_HOST or REMOTE_REPOSITORY_ROOT when necessary. A full
resubmission is safe: compatible day JSONs are verified and skipped.
EOF
    ;;
esac
