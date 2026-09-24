#!/bin/bash
set -euo pipefail

LOCAL_STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${LOCAL_STUDY_DIR}/../../../../../../.." && pwd)"
# Current Rutgers Amarel public login endpoint.  The AMAREL_HOST override
# remains available if Rutgers changes the endpoint again.
REMOTE_HOST="${AMAREL_HOST:-jl2815@amarel-new.hpc.rutgers.edu}"
REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1"
REMOTE_STUDY_DIR="${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/$(basename "${LOCAL_STUDY_DIR}")"
REMOTE_DATA_ROOT="/home/jl2815/tco/data"
REMOTE_OUTPUT="/home/jl2815/tco/exercise_output/summer/fixed_geographic_three_model_092426"
LOCAL_PYTHON="${LOCAL_GEMS_PYTHON:-/opt/anaconda3/envs/faiss_env/bin/python}"
DOWNLOAD_TAG="${DOWNLOAD_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_DOWNLOAD="${LOCAL_STUDY_DIR}/downloaded_results/${DOWNLOAD_TAG}"
MODE="${1:-push}"
PACKAGE_SYNC_ID="$(date +%Y%m%d_%H%M%S)_$$"
REMOTE_PACKAGE_ARCHIVE="/home/jl2815/tco/.gems_tco_package_${PACKAGE_SYNC_ID}.tar.gz"
REMOTE_PACKAGE_STAGE="/home/jl2815/tco/.gems_tco_package_${PACKAGE_SYNC_ID}"
LOCAL_PACKAGE_ARCHIVE=""
SSH_CONTROL_DIR=""
SSH_CONTROL_PATH=""
SSH_OPTIONS=()

cleanup_local_archive() {
  if [[ -n "${LOCAL_PACKAGE_ARCHIVE}" && -f "${LOCAL_PACKAGE_ARCHIVE}" ]]; then
    rm -f -- "${LOCAL_PACKAGE_ARCHIVE}"
  fi
}

close_ssh_master() {
  if [[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]]; then
    ssh -S "${SSH_CONTROL_PATH}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SSH_CONTROL_PATH}" && -e "${SSH_CONTROL_PATH}" ]]; then
    rm -f -- "${SSH_CONTROL_PATH}"
  fi
  if [[ -n "${SSH_CONTROL_DIR}" && -d "${SSH_CONTROL_DIR}" ]]; then
    rmdir -- "${SSH_CONTROL_DIR}" 2>/dev/null || true
  fi
}

cleanup() {
  cleanup_local_archive
  close_ssh_master
}

trap cleanup EXIT

require_local_mac() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This helper must be run in the local Mac terminal, not on an Amarel login node." >&2
    echo "Expected local study directory: /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/$(basename "${LOCAL_STUDY_DIR}")" >&2
    exit 2
  fi
}

start_ssh_master() {
  require_local_mac
  if [[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]]; then
    return
  fi
  SSH_CONTROL_DIR="$(mktemp -d /tmp/gems_tco_ssh.XXXXXX)"
  SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/control"
  SSH_OPTIONS=(
    -o ControlMaster=auto
    -o ControlPersist=600
    -o "ControlPath=${SSH_CONTROL_PATH}"
  )
  echo "Opening one reusable SSH connection to ${REMOTE_HOST}..."
  ssh "${SSH_OPTIONS[@]}" -MNf "${REMOTE_HOST}"
}

check_remote_data_files() {
  echo "Checking the two Amarel monthly input files before package deployment..."
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "set -euo pipefail
     for data_file in \
       '${REMOTE_DATA_ROOT}/pickle_2024/tco_grid_24_07.pkl' \
       '${REMOTE_DATA_ROOT}/pickle_2025/tco_grid_25_07.pkl'; do
       if [[ ! -r \"\${data_file}\" ]]; then
         echo \"Missing or unreadable Amarel input: \${data_file}\" >&2
         exit 2
       fi
     done
     ls -lh \
       '${REMOTE_DATA_ROOT}/pickle_2024/tco_grid_24_07.pkl' \
       '${REMOTE_DATA_ROOT}/pickle_2025/tco_grid_25_07.pkl'"
}

sync_package_sources() {
  echo "Creating an exact clean archive of the current local GEMS_TCO working tree..."
  LOCAL_PACKAGE_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/gems_tco_package.XXXXXX")"
  (
    cd "${LOCAL_ROOT}"
    COPYFILE_DISABLE=1 tar --no-xattrs \
      -czf "${LOCAL_PACKAGE_ARCHIVE}" \
      --exclude='.DS_Store' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.so' \
      --exclude='*.pyd' \
      pyproject.toml \
      setup.py \
      MANIFEST.in \
      README.md \
      LICENSE \
      CITATION.cff \
      THIRD_PARTY_NOTICES.md \
      src/GEMS_TCO \
      cpp \
      tests \
      docs \
      scripts/amarel
  )
  LOCAL_PACKAGE_SHA256="$(shasum -a 256 "${LOCAL_PACKAGE_ARCHIVE}" | awk '{print $1}')"

  # Check only the persistent runtime environment before replacing any remote
  # package source.  Build requirements are deliberately checked by pip in an
  # isolated PEP 517 environment below, so an older system setuptools does not
  # parse this project's modern PEP 639 license metadata.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "command -v c++ >/dev/null && \
     /home/jl2815/.conda/envs/faiss_env/bin/python -c 'import sys; import numpy, pandas, scipy, sklearn, torch; assert sys.version_info >= (3, 11); print(\"runtime environment ok:\", sys.version.split()[0], \"torch\", torch.__version__)'"

  scp "${SSH_OPTIONS[@]}" \
    "${LOCAL_PACKAGE_ARCHIVE}" "${REMOTE_HOST}:${REMOTE_PACKAGE_ARCHIVE}"
  REMOTE_PACKAGE_SHA256="$(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" "sha256sum '${REMOTE_PACKAGE_ARCHIVE}' | cut -d' ' -f1")"
  if [[ "${REMOTE_PACKAGE_SHA256}" != "${LOCAL_PACKAGE_SHA256}" ]]; then
    echo "Package archive checksum mismatch after SCP" >&2
    exit 1
  fi

  # Extract first, validate the staged tree, then replace only package/build
  # sources. Data, exercises outside this study, and result directories are
  # deliberately untouched.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "set -euo pipefail
     rm -rf '${REMOTE_PACKAGE_STAGE}'
     mkdir -p '${REMOTE_PACKAGE_STAGE}'
     tar -xzf '${REMOTE_PACKAGE_ARCHIVE}' -C '${REMOTE_PACKAGE_STAGE}'
     test -f '${REMOTE_PACKAGE_STAGE}/src/GEMS_TCO/__init__.py'
     test -f '${REMOTE_PACKAGE_STAGE}/src/GEMS_TCO/vecchia/corridor_neighbors/separable_exponential.py'
     test -f '${REMOTE_PACKAGE_STAGE}/cpp/maxmin_order.cpp'
     test -f '${REMOTE_PACKAGE_STAGE}/cpp/vecchia_covariance_cpu.cpp'
     test -f '${REMOTE_PACKAGE_STAGE}/cpp/vecchia_covariance_cuda.cpp'
     test -f '${REMOTE_PACKAGE_STAGE}/cpp/vecchia_covariance_cuda_kernel.cu'
     mkdir -p '${REMOTE_PROJECT}/src' '${REMOTE_PROJECT}/scripts'
     rm -rf '${REMOTE_PROJECT}/src/GEMS_TCO' \
       '${REMOTE_PROJECT}/src/GEMS_TCO.egg-info' \
       '${REMOTE_PROJECT}/build' \
       '${REMOTE_PROJECT}/cpp' \
       '${REMOTE_PROJECT}/tests' \
       '${REMOTE_PROJECT}/docs' \
       '${REMOTE_PROJECT}/scripts/amarel'
     mv '${REMOTE_PACKAGE_STAGE}/src/GEMS_TCO' '${REMOTE_PROJECT}/src/GEMS_TCO'
     mv '${REMOTE_PACKAGE_STAGE}/cpp' '${REMOTE_PROJECT}/cpp'
     mv '${REMOTE_PACKAGE_STAGE}/tests' '${REMOTE_PROJECT}/tests'
     mv '${REMOTE_PACKAGE_STAGE}/docs' '${REMOTE_PROJECT}/docs'
     mv '${REMOTE_PACKAGE_STAGE}/scripts/amarel' '${REMOTE_PROJECT}/scripts/amarel'
     mv -f '${REMOTE_PACKAGE_STAGE}/pyproject.toml' \
       '${REMOTE_PACKAGE_STAGE}/setup.py' \
       '${REMOTE_PACKAGE_STAGE}/MANIFEST.in' \
       '${REMOTE_PACKAGE_STAGE}/README.md' \
       '${REMOTE_PACKAGE_STAGE}/LICENSE' \
       '${REMOTE_PACKAGE_STAGE}/CITATION.cff' \
       '${REMOTE_PACKAGE_STAGE}/THIRD_PARTY_NOTICES.md' \
       '${REMOTE_PROJECT}/'"

  # Build only the portable max-min extension. These three study models use
  # the exact Torch covariance path, so optional covariance extensions remain
  # source-available but disabled for this install.  Keep PEP 517 build
  # isolation enabled: it installs the versions declared in [build-system]
  # (notably setuptools>=77 for the SPDX license fields) without changing the
  # persistent Amarel environment. --no-deps applies only to runtime deps.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "GEMS_TCO_BUILD_TORCH_EXT=0 GEMS_TCO_BUILD_CUDA_EXT=0 \
      /home/jl2815/.conda/envs/faiss_env/bin/python -m pip install \
      --no-deps -e '${REMOTE_PROJECT}'"

  # Verify package origin, the rebuilt Linux extension, retired-module cleanup,
  # all three study model imports, their six-parameter Torch backends, and a
  # finite float64 covariance/gradient calculation.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && PYTHONPATH='${REMOTE_PROJECT}/src' \
      /home/jl2815/.conda/envs/faiss_env/bin/python \
      '${REMOTE_PROJECT}/scripts/amarel/verify_package_install.py' \
      --project-root '${REMOTE_PROJECT}' \
      --require-distribution-metadata"

  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "rm -rf '${REMOTE_PACKAGE_STAGE}' '${REMOTE_PACKAGE_ARCHIVE}'"
  echo "GEMS_TCO package source, install, and imports verified before study upload."
}

push_sources() {
  require_local_mac
  test -x "${LOCAL_PYTHON}" || {
    echo "Missing local study Python: ${LOCAL_PYTHON}" >&2
    exit 2
  }
  start_ssh_master
  check_remote_data_files
  sync_package_sources

  # The study output lives elsewhere, so replacing this code-only directory
  # prevents stale array scripts or older configs from surviving the upload.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "rm -rf '${REMOTE_STUDY_DIR}' && \
     mkdir -p '${REMOTE_STUDY_DIR}' /home/jl2815/tco/exercise_output/summer/logs"
  scp "${SSH_OPTIONS[@]}" \
    "${LOCAL_STUDY_DIR}"/*.py \
    "${LOCAL_STUDY_DIR}"/*.sh \
    "${LOCAL_STUDY_DIR}"/*.toml \
    "${LOCAL_STUDY_DIR}"/*.json \
    "${LOCAL_STUDY_DIR}"/*.csv \
    "${LOCAL_STUDY_DIR}/README.md" \
    "${REMOTE_HOST}:${REMOTE_STUDY_DIR}/"

  LOCAL_STUDY_SIGNATURE="$(
    "${LOCAL_PYTHON}" -c \
      'import sys; from pathlib import Path; study=Path(sys.argv[1]); sys.path.insert(0, str(study)); from fixed_geo_three_model_core import study_signature; print(study_signature(study / "fixed_geo_three_model.toml", study / "frozen_design.json", study / "evaluation_dates.csv"))' \
      "${LOCAL_STUDY_DIR}"
  )"
  REMOTE_STUDY_SIGNATURE="$(
    ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
      "cd '${REMOTE_STUDY_DIR}' && PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_STUDY_DIR}' \
       /home/jl2815/.conda/envs/faiss_env/bin/python -c \
       'from pathlib import Path; from fixed_geo_three_model_core import study_signature; study=Path(\".\").resolve(); print(study_signature(study / \"fixed_geo_three_model.toml\", study / \"frozen_design.json\", study / \"evaluation_dates.csv\"))'"
  )"
  if [[ "${REMOTE_STUDY_SIGNATURE}" != "${LOCAL_STUDY_SIGNATURE}" ]]; then
    echo "Study signature mismatch after SCP" >&2
    echo "local:  ${LOCAL_STUDY_SIGNATURE}" >&2
    echo "remote: ${REMOTE_STUDY_SIGNATURE}" >&2
    exit 1
  fi

  # Final no-fit validation uses the newly installed package and all 60 remote
  # data-date/slot checks before the user submits a GPU job.
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "cd '${REMOTE_STUDY_DIR}' && PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_STUDY_DIR}' \
      /home/jl2815/.conda/envs/faiss_env/bin/python \
      validate_fixed_geo_three_model.py --check-data \
      --data-root '${REMOTE_DATA_ROOT}'"

  echo "Verified study signature: ${LOCAL_STUDY_SIGNATURE}"
  echo "Package-first update and study upload complete: ${REMOTE_STUDY_DIR}"
  echo "Smoke test: ssh ${REMOTE_HOST} 'cd ${REMOTE_STUDY_DIR} && bash submit_fixed_geo_three_model.sh smoke'"
  echo "Full run:   ssh ${REMOTE_HOST} 'cd ${REMOTE_STUDY_DIR} && bash submit_fixed_geo_three_model.sh full'"
}

submit_remote() {
  local run_mode="$1"
  start_ssh_master
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "cd '${REMOTE_STUDY_DIR}' && bash submit_fixed_geo_three_model.sh '${run_mode}'"
}

pull_results() {
  start_ssh_master
  mkdir -p "${LOCAL_DOWNLOAD}"
  scp "${SSH_OPTIONS[@]}" -r \
    "${REMOTE_HOST}:${REMOTE_OUTPUT}/." "${LOCAL_DOWNLOAD}/"
  echo "Downloaded results to ${LOCAL_DOWNLOAD}"
}

case "${MODE}" in
  push) push_sources ;;
  push-smoke) push_sources; submit_remote smoke ;;
  push-full) push_sources; submit_remote full ;;
  submit-smoke) submit_remote smoke ;;
  submit-full) submit_remote full ;;
  pull) pull_results ;;
  *)
    echo "Usage: $0 [push|push-smoke|push-full|submit-smoke|submit-full|pull]" >&2
    exit 2
    ;;
esac
