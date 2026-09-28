#!/bin/bash
set -euo pipefail

# Deploy, submit, inspect, or download the 59-day likelihood-only run.
# Amarel refuses extra SSH sessions opened through ControlMaster, so every mode
# below deliberately uses exactly one authenticated remote connection.

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(git -C "${LOCAL_DIR}" rev-parse --show-toplevel)"
REMOTE_HOST="${AMAREL_HOST:-jl2815@amarel-new.hpc.rutgers.edu}"
REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1-vecchia-rerun-20260927"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_BASE="/home/jl2815/tco/exercise_output/fall_26"
REMOTE_OUTPUT_NAME="vecchia_real59_adapted_fixed_nll_lag643_rerun_20260927"
REMOTE_OUTPUT="${REMOTE_BASE}/${REMOTE_OUTPUT_NAME}"
REMOTE_LOG_DIR="${REMOTE_BASE}/logs"
REMOTE_PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
LOCAL_DOWNLOAD_ROOT="${LOCAL_DIR}/downloaded_results"
LOCAL_OUTPUT="${LOCAL_DOWNLOAD_ROOT}/${REMOTE_OUTPUT_NAME}"
VALIDATOR="${LOCAL_DIR}/verify_vecchia_real59_nll_results.py"
MODE="${1:-submit}"

LOCAL_STAGE=""
REMOTE_TRANSCRIPT=""

usage() {
  cat <<'EOF'
Usage:
  bash run_vecchia_real59_gpu_on_amarel.sh submit
  bash run_vecchia_real59_gpu_on_amarel.sh status
  bash run_vecchia_real59_gpu_on_amarel.sh pull

submit  Upload, install, verify, and submit through one SSH login.
status  Read the Slurm state, newest logs, and completion marker through one login.
pull    Download results and matching logs through one rsync login, then validate.

Set AMAREL_HOST to override the default Rutgers login host.
EOF
}

cleanup_local() {
  if [[ -n "${LOCAL_STAGE}" && -d "${LOCAL_STAGE}" ]]; then
    case "${LOCAL_STAGE}" in
      "${TMPDIR:-/tmp}"/gems_tco_vecchia_deploy.*|/private/tmp/gems_tco_vecchia_deploy.*|/tmp/gems_tco_vecchia_deploy.*)
        rm -rf -- "${LOCAL_STAGE}"
        ;;
      *)
        echo "Refusing to remove unexpected staging path: ${LOCAL_STAGE}" >&2
        ;;
    esac
  fi
  if [[ -n "${REMOTE_TRANSCRIPT}" && -f "${REMOTE_TRANSCRIPT}" ]]; then
    rm -f -- "${REMOTE_TRANSCRIPT}"
  fi
}
trap cleanup_local EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 2
  }
}

prepare_local_stage() {
  LOCAL_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/gems_tco_vecchia_deploy.XXXXXX")"
  mkdir -p "${LOCAL_STAGE}/project/src" \
    "${LOCAL_STAGE}/project/cpp" \
    "${LOCAL_STAGE}/project/scripts/amarel" \
    "${LOCAL_STAGE}/experiment"

  rsync -a \
    "${LOCAL_ROOT}/pyproject.toml" \
    "${LOCAL_ROOT}/setup.py" \
    "${LOCAL_ROOT}/MANIFEST.in" \
    "${LOCAL_ROOT}/README.md" \
    "${LOCAL_ROOT}/LICENSE" \
    "${LOCAL_ROOT}/CITATION.cff" \
    "${LOCAL_ROOT}/THIRD_PARTY_NOTICES.md" \
    "${LOCAL_STAGE}/project/"
  rsync -a \
    --exclude='*.so' --exclude='*.dylib' --exclude='*.pyd' \
    --exclude='__pycache__/' --exclude='.DS_Store' \
    "${LOCAL_ROOT}/src/" "${LOCAL_STAGE}/project/src/"
  rsync -a \
    --exclude='__pycache__/' --exclude='.DS_Store' \
    "${LOCAL_ROOT}/cpp/" "${LOCAL_STAGE}/project/cpp/"
  rsync -a \
    --exclude='__pycache__/' --exclude='.DS_Store' \
    "${LOCAL_ROOT}/scripts/amarel/" "${LOCAL_STAGE}/project/scripts/amarel/"

  rsync -a \
    "${LOCAL_DIR}/vecchia_adapted_fixed_lag643_core.py" \
    "${LOCAL_DIR}/vecchia_real59_adapted_fixed_nll_lag643.py" \
    "${LOCAL_DIR}/slurm_vecchia_real59_adapted_fixed_nll_lag643.sh" \
    "${LOCAL_DIR}/verify_vecchia_real59_nll_results.py" \
    "${LOCAL_DIR}/README.md" \
    "${LOCAL_DIR}/AMAREL_VECCHIA_GPU_OPTIMIZATION_MEMO_090326.txt" \
    "${LOCAL_STAGE}/experiment/"
}

deploy_and_submit() {
  prepare_local_stage
  REMOTE_TRANSCRIPT="$(mktemp "${TMPDIR:-/tmp}/gems_tco_vecchia_remote.XXXXXX")"

  remote_command=""
  IFS= read -r -d '' remote_command <<EOF || true
set -euo pipefail
deploy_stage=\$(mktemp -d /home/jl2815/tco/.vecchia_deploy.XXXXXX)
project_backup=""
project_replaced=0
cleanup_remote() {
  status=\$?
  if [[ "\${status}" -ne 0 && "\${project_replaced}" -eq 1 && -n "\${project_backup}" && -d "\${project_backup}" ]]; then
    echo "Deployment failed; restoring the preceding maintained source tree." >&2
    rm -rf -- '${REMOTE_PROJECT}'
    mv -- "\${project_backup}" '${REMOTE_PROJECT}'
    project_backup=""
  fi
  case "\${deploy_stage}" in
    /home/jl2815/tco/.vecchia_deploy.*) rm -rf -- "\${deploy_stage}" ;;
    *) echo "Refusing to remove unexpected remote staging path: \${deploy_stage}" >&2 ;;
  esac
  if [[ -n "\${project_backup}" && -d "\${project_backup}" ]]; then
    case "\${project_backup}" in
      '${REMOTE_PROJECT}'.previous.*) rm -rf -- "\${project_backup}" ;;
      *) echo "Refusing to remove unexpected backup path: \${project_backup}" >&2 ;;
    esac
  fi
  exit "\${status}"
}
trap cleanup_remote EXIT

echo "Receiving the maintained package and likelihood-only experiment..."
tar -xf - -C "\${deploy_stage}"
case '${REMOTE_PROJECT}' in
  /home/jl2815/tco/GEMS_TCO-1-vecchia-rerun-*) ;;
  *) echo 'Refusing to replace an unexpected remote project path.' >&2; exit 2 ;;
esac
if [[ -e '${REMOTE_PROJECT}' ]]; then
  project_backup='${REMOTE_PROJECT}'.previous.\$\$
  mv -- '${REMOTE_PROJECT}' "\${project_backup}"
fi
mv -- "\${deploy_stage}/project" '${REMOTE_PROJECT}'
project_replaced=1
mkdir -p '${REMOTE_DIR}' '${REMOTE_OUTPUT}' '${REMOTE_LOG_DIR}'
cp -R "\${deploy_stage}/experiment/." '${REMOTE_DIR}/'

echo "Checking inputs and installing the current package..."
test -r /home/jl2815/tco/data/pickle_2024/tco_grid_24_07.pkl
test -r /home/jl2815/tco/data/pickle_2025/tco_grid_25_07.pkl
test -x '${REMOTE_PYTHON}'
'${REMOTE_PYTHON}' -c 'import pybind11, setuptools; assert int(setuptools.__version__.split(chr(46), 1)[0]) >= 77; print("build dependencies:", pybind11.__version__, setuptools.__version__)' || \
  '${REMOTE_PYTHON}' -m pip install --disable-pip-version-check --upgrade 'setuptools>=77' 'pybind11>=2.12'
cd '${REMOTE_PROJECT}'
GEMS_TCO_BUILD_TORCH_EXT=0 GEMS_TCO_BUILD_CUDA_EXT=0 \
  '${REMOTE_PYTHON}' -m pip install --no-build-isolation --no-deps --editable .
GEMS_TCO_SRC='${REMOTE_PROJECT}/src' PYTHONPATH='${REMOTE_PROJECT}/src' \
  '${REMOTE_PYTHON}' -c 'import pathlib, GEMS_TCO; from GEMS_TCO.data.loading import ProcessedDataLoader; from GEMS_TCO.vecchia.corridor_neighbors import DirectionalLag643CorridorVecchia; from GEMS_TCO import _maxmin; root=pathlib.Path(GEMS_TCO.__file__).resolve().parent; expected=pathlib.Path("${REMOTE_PROJECT}/src/GEMS_TCO").resolve(); assert root == expected, (root, expected); print("verified package:", root); print("verified data loader:", pathlib.Path(__import__("GEMS_TCO.data.loading", fromlist=["x"]).__file__).resolve()); print("verified maxmin:", pathlib.Path(_maxmin.__file__).resolve())'
if [[ -n "\${project_backup}" && -d "\${project_backup}" ]]; then
  rm -rf -- "\${project_backup}"
  project_backup=""
fi

echo "Submitting the fresh single-A100 likelihood rerun..."
job_id=\$(cd '${REMOTE_DIR}' && sbatch --parsable slurm_vecchia_real59_adapted_fixed_nll_lag643.sh)
echo "CODEX_JOB_ID=\${job_id}"
EOF

  echo "One Amarel password prompt should appear. Keep this connection open through submission."
  COPYFILE_DISABLE=1 tar -C "${LOCAL_STAGE}" -cf - project experiment \
    | ssh -o ControlMaster=no -o ControlPath=none "${REMOTE_HOST}" "${remote_command}" \
    | tee "${REMOTE_TRANSCRIPT}"

  job_id="$(sed -n 's/^CODEX_JOB_ID=//p' "${REMOTE_TRANSCRIPT}" | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "Remote deployment ended without a Slurm job id." >&2
    exit 3
  fi
  mkdir -p "${LOCAL_DOWNLOAD_ROOT}"
  {
    echo "job_id=${job_id}"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "remote_host=${REMOTE_HOST}"
    echo "remote_output=${REMOTE_OUTPUT}"
  } > "${LOCAL_DOWNLOAD_ROOT}/last_gpu_rerun_submission.txt"
  echo "Submitted Slurm job ${job_id}"
  echo "Check it with:"
  echo "  bash ${LOCAL_DIR}/run_vecchia_real59_gpu_on_amarel.sh status"
}

show_status() {
  echo "One Amarel password prompt should appear."
  ssh -o ControlMaster=no -o ControlPath=none "${REMOTE_HOST}" "
    set -e
    echo 'Matching Slurm jobs:'
    squeue -u jl2815 -n vecc59_nll_rerun -o '%.18i %.20j %.10T %.10M %.6D %R'
    echo
    echo 'Newest matching logs:'
    find '${REMOTE_LOG_DIR}' -maxdepth 1 -type f \\
      \( -name 'vecc59_nll_rerun_*.out' -o -name 'vecc59_nll_rerun_*.err' \) \\
      -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\\n' | sort | tail -n 8
    echo
    echo 'Completion marker:'
    if test -s '${REMOTE_OUTPUT}/RUN_COMPLETE.json'; then
      cat '${REMOTE_OUTPUT}/RUN_COMPLETE.json'
    else
      echo 'not complete yet'
    fi
  "
}

pull_results() {
  mkdir -p "${LOCAL_DOWNLOAD_ROOT}"
  echo "One Amarel password prompt should appear."
  echo "Downloading/resuming ${REMOTE_OUTPUT} and its Slurm logs"
  rsync -avh --partial --partial-dir=.rsync-partial --progress --itemize-changes \
    -e "ssh -o ControlMaster=no -o ControlPath=none" \
    --include="/${REMOTE_OUTPUT_NAME}/" \
    --include="/${REMOTE_OUTPUT_NAME}/***" \
    --include='/logs/' \
    --include='/logs/vecc59_nll_rerun_*.out' \
    --include='/logs/vecc59_nll_rerun_*.err' \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_BASE}/" "${LOCAL_DOWNLOAD_ROOT}/"
  python3 "${VALIDATOR}" "${LOCAL_OUTPUT}"
  echo "Validated local result: ${LOCAL_OUTPUT}"
}

case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  submit)
    require_command git
    require_command rsync
    require_command tar
    require_command ssh
    deploy_and_submit
    ;;
  status)
    require_command ssh
    show_status
    ;;
  pull)
    require_command rsync
    require_command ssh
    require_command python3
    test -f "${VALIDATOR}" || {
      echo "Missing validator: ${VALIDATOR}" >&2
      exit 2
    }
    pull_results
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
