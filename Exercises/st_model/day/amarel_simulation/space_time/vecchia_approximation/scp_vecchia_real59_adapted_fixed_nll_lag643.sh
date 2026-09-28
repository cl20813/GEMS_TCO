#!/bin/bash
set -euo pipefail

# Backward-compatible entry point.  The maintained helper also installs the
# current package and submits the fresh GPU rerun after uploading the files.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/run_vecchia_real59_gpu_on_amarel.sh" submit
