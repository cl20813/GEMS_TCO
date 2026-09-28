#!/bin/bash
set -euo pipefail

ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMAREL_HOST="${AMAREL_HOST:-jl2815@amarel-new.hpc.rutgers.edu}"
REMOTE_REPOSITORY_ROOT="${REMOTE_REPOSITORY_ROOT:-/home/jl2815/tco/GEMS_TCO-1}"
REMOTE_SIM_DIR="${REMOTE_REPOSITORY_ROOT}/simulate_data/092726"
REMOTE_RUN_ROOT="/home/jl2815/tco/exercise_output/fall_26"
REMOTE_LOG_ROOT="${REMOTE_RUN_ROOT}/logs"
REMOTE_OUTPUT_ROOT="${REMOTE_RUN_ROOT}/st_interaction_july2024_nugget0_092726"
LOCAL_OUTPUT_ROOT="${SCRIPT_DIR}/st_interaction_july2024_nugget0_092726"

FILES=(
  generate_st_interaction_suite_092726.py
  validate_st_interaction_suite_092726.py
  st_interaction_scenarios_092726.json
  slurm_generate_st_interaction_suite_092726.sbatch
  submit_st_interaction_suite_092726.sh
)

upload() {
  ssh "${AMAREL_HOST}" "mkdir -p '${REMOTE_SIM_DIR}' '${REMOTE_LOG_ROOT}' '${REMOTE_OUTPUT_ROOT}'"
  local paths=()
  local name
  for name in "${FILES[@]}"; do
    paths+=("${SCRIPT_DIR}/${name}")
  done
  rsync -avh --progress "${paths[@]}" "${AMAREL_HOST}:${REMOTE_SIM_DIR}/"
}

submit_mode() {
  local mode="$1"
  ssh "${AMAREL_HOST}" \
    "GEMS_TCO_REPOSITORY_ROOT='${REMOTE_REPOSITORY_ROOT}' bash '${REMOTE_SIM_DIR}/submit_st_interaction_suite_092726.sh' '${mode}'"
}

download_mode() {
  local suffix="$1"
  local remote="${REMOTE_OUTPUT_ROOT}${suffix}"
  local local_root="${LOCAL_OUTPUT_ROOT}${suffix}"
  mkdir -p "${local_root}"
  # Consolidated monthly assets are sufficient locally.  Daily checkpoints
  # stay on Amarel for restart/resume without doubling transfer and disk use.
  rsync -avh --partial --progress --exclude='day_checkpoints/' \
    "${AMAREL_HOST}:${remote}/" "${local_root}/"
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
    ssh "${AMAREL_HOST}" "squeue -u jl2815 -n stint092726; sacct -X -n -S today --name stint092726 --format=JobID,State,Elapsed,MaxRSS,ExitCode"
    ;;
  logs)
    ssh "${AMAREL_HOST}" "ls -lh '${REMOTE_LOG_ROOT}'/stint092726_* 2>/dev/null || true"
    ;;
  download-pilot)
    download_mode _pilot
    ;;
  download)
    download_mode ""
    ;;
  validate-pilot)
    /opt/anaconda3/envs/faiss_env/bin/python \
      "${SCRIPT_DIR}/validate_st_interaction_suite_092726.py" \
      --root "${LOCAL_OUTPUT_ROOT}_pilot" --expected-days 1 --deep
    ;;
  validate)
    /opt/anaconda3/envs/faiss_env/bin/python \
      "${SCRIPT_DIR}/validate_st_interaction_suite_092726.py" \
      --root "${LOCAL_OUTPUT_ROOT}" --expected-days 30 --deep
    ;;
  help|*)
    cat <<EOF
Usage: $(basename "$0") ACTION

Actions:
  upload               copy generator/config/job files to Amarel
  submit-pilot         submit six one-day production-resolution x100/x10 pilots
  submit-full          submit six 30-day x100/x10 production tasks
  upload-submit-pilot  upload and submit pilot
  upload-submit-full   upload and submit production
  status               show Slurm queue/accounting state
  logs                 list array logs
  download-pilot       rsync pilot results into ${LOCAL_OUTPUT_ROOT}_pilot
  download             rsync full results into ${LOCAL_OUTPUT_ROOT}
  validate-pilot       deeply validate the downloaded one-day pilot
  validate             deeply validate the downloaded 30-day suite

Override the login endpoint with AMAREL_HOST and the remote checkout with
REMOTE_REPOSITORY_ROOT.  Generated data live under Amarel exercise_output and
are copied back under this repository's simulate_data directory.
EOF
    ;;
esac
