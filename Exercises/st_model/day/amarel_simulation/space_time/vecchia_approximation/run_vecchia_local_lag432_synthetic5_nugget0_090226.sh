#!/bin/bash
set -uo pipefail

REPO_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
RUN_DIR="${REPO_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/opt/anaconda3/envs/faiss_env/bin/python"
SCRIPT="${RUN_DIR}/vecchia_local_adapted_vs_fixed_lag432_090226.py"
SELECTION="${RUN_DIR}/vecchia_local_lag432_synthetic5_nugget0_selection_090226.json"
SYNTHETIC_DATA_ROOT="${REPO_ROOT}/outputs/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_matched5_090226"
OUTPUT_ROOT="${REPO_ROOT}/outputs/summer_26/vecchia_four_geometry_lag432_local_synthetic5_nugget0_matched_run04"

export PYTHONPATH="${REPO_ROOT}/src:${RUN_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${OUTPUT_ROOT}/.mplconfig"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

run_one_task() {
  local TASK_INDEX="$1"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --mode run-task \
    --task-index "${TASK_INDEX}" \
    --selection-file "${SELECTION}" \
    --synthetic-data-root "${SYNTHETIC_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --smooth 0.5 \
    --truth-nugget 0.0 \
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
}

PIDS=()
TASKS=()
for TASK_INDEX in $(seq 0 4); do
  TASK_LOG="${OUTPUT_ROOT}/task_${TASK_INDEX}.driver.log"
  echo "Starting matched nugget0 lag432 task ${TASK_INDEX}/4 at $(date); log=${TASK_LOG}"
  run_one_task "${TASK_INDEX}" >"${TASK_LOG}" 2>&1 &
  PIDS+=("$!")
  TASKS+=("${TASK_INDEX}")
done

FAILED_TASKS=()
for POSITION in "${!PIDS[@]}"; do
  TASK_INDEX="${TASKS[${POSITION}]}"
  if wait "${PIDS[${POSITION}]}"; then
    echo "Completed task ${TASK_INDEX}/4 at $(date)"
  else
    TASK_EXIT=$?
    FAILED_TASKS+=("${TASK_INDEX}:${TASK_EXIT}")
    echo "FAILED task ${TASK_INDEX}/4 with exit ${TASK_EXIT}; see task log." >&2
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
echo "All five matched nugget0 lag432 tasks completed successfully."
