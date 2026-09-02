#!/bin/bash
set -uo pipefail

REPO_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
RUN_DIR="${REPO_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/opt/anaconda3/envs/faiss_env/bin/python"
SCRIPT="${RUN_DIR}/vecchia_local_adapted_vs_fixed_lag432_090226.py"
SELECTION="${RUN_DIR}/vecchia_local_lag432_selection_090226.json"
REAL_DATA_ROOT="/Users/joonwonlee/Documents/GEMS_DATA"
SYNTHETIC_DATA_ROOT="/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"
OUTPUT_ROOT="${REPO_ROOT}/outputs/summer_26/vecchia_four_geometry_lag432_local_real5_synthetic5_run01"

export PYTHONPATH="${REPO_ROOT}/src:${RUN_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${OUTPUT_ROOT}/.mplconfig"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

FAILED_TASKS=()
for TASK_INDEX in $(seq 0 9); do
  echo "Starting local lag432 task ${TASK_INDEX}/9 at $(date)"
  if "${PYTHON_BIN}" "${SCRIPT}" \
    --mode run-task \
    --task-index "${TASK_INDEX}" \
    --selection-file "${SELECTION}" \
    --real-data-root "${REAL_DATA_ROOT}" \
    --synthetic-data-root "${SYNTHETIC_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --smooth 0.5 \
    --truth-nugget 1.0 \
    --fixed-nugget 0.0 \
    --keep-exact-loc \
    --daily-stride 2 \
    --target-chunk-size 32 \
    --union-target-chunk-size 8 \
    --diag-chunk-size 64 \
    --union-diag-chunk-size 16 \
    --fit-order union,adapted,shifted,fixed \
    --lbfgs-lr 1.0 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-history 10 \
    --grad-tol 1e-5 \
    --empirical-max-lat-offset 20 \
    --empirical-max-lon-offset 20 \
    --empirical-min-pair-count 1000 \
    --empirical-smooth-bandwidth-deg 0.063 \
    --subgrid-max-condition-number 100 \
    --resample-grid 500 \
    --device cpu \
    --suppress-fit-prints
  then
    echo "Completed local lag432 task ${TASK_INDEX}/9 at $(date)"
  else
    TASK_EXIT=$?
    FAILED_TASKS+=("${TASK_INDEX}:${TASK_EXIT}")
    echo "FAILED local task ${TASK_INDEX}/9 with exit ${TASK_EXIT}; continuing." >&2
  fi
done

AGGREGATE_EXIT=0
"${PYTHON_BIN}" "${SCRIPT}" \
  --mode aggregate \
  --selection-file "${SELECTION}" \
  --output-root "${OUTPUT_ROOT}" \
  --allow-partial-aggregate || AGGREGATE_EXIT=$?

if (( AGGREGATE_EXIT != 0 )); then
  echo "ERROR: aggregate failed with exit ${AGGREGATE_EXIT}." >&2
  exit "${AGGREGATE_EXIT}"
fi
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "ERROR: failed local tasks: ${FAILED_TASKS[*]}" >&2
  exit 1
fi
echo "All ten local lag432 tasks and aggregate completed successfully."
