#!/bin/bash
#SBATCH --job-name=stint3_seq
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stint3_seq_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stint3_seq_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
# gpu020 is an A100 node currently listed in the mixed state. This is a node
# choice within the gpu partition, not a second partition or an array.
#SBATCH --nodelist=gpu020
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1"
STUDY_DIR="${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/fixed_geographic_three_model_092426"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/fixed_geographic_three_model_092426"
MANIFEST="${STUDY_DIR}/evaluation_dates.csv"
RUN_MODE="${RUN_MODE:-full}"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${STUDY_DIR}/run_fixed_geo_three_model_day.py" || {
  echo "Missing study runner under ${STUDY_DIR}" >&2
  exit 2
}
test -f "${MANIFEST}" || { echo "Missing manifest: ${MANIFEST}" >&2; exit 2; }

TASK_COUNT="$(${PYTHON_BIN} -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "${MANIFEST}")"
test "${TASK_COUNT}" -eq 60 || {
  echo "Expected 60 manifest rows, found ${TASK_COUNT}" >&2
  exit 2
}

if [[ "${RUN_MODE}" == "smoke" ]]; then
  TASK_IDS=(0 30)
elif [[ "${RUN_MODE}" == "full" ]]; then
  TASK_IDS=()
  for ((task_id = 0; task_id < TASK_COUNT; task_id++)); do
    TASK_IDS+=("${task_id}")
  done
else
  echo "RUN_MODE must be smoke or full, got: ${RUN_MODE}" >&2
  exit 2
fi

export PYTHONPATH="${REMOTE_PROJECT}/src:${STUDY_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/jl2815_stint3_seq_${SLURM_JOB_ID:-manual}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Sequential job: ${SLURM_JOB_ID:-manual}"
echo "Run mode: ${RUN_MODE}"
echo "Task order: ${TASK_IDS[*]}"
echo "Study: fixed geographic, frozen A/B, lag 1, GC/JM0.5/advected-separable"
echo "Vecchia compute setting: corridor lag 6/4/3, target chunk size 256"
echo "Output: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"${PYTHON_BIN}" -c 'import torch; print("torch", torch.__version__, "built CUDA", torch.version.cuda); assert torch.cuda.is_available(); print("device", torch.cuda.get_device_name(0))'

for TASK_ID in "${TASK_IDS[@]}"; do
  echo "===== task ${TASK_ID} started: $(date) ====="
  "${PYTHON_BIN}" "${STUDY_DIR}/run_fixed_geo_three_model_day.py" \
    --task-id "${TASK_ID}" \
    --config "${STUDY_DIR}/fixed_geo_three_model.toml" \
    --design "${STUDY_DIR}/frozen_design.json" \
    --dates "${MANIFEST}" \
    --data-root /home/jl2815/tco/data \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda
  echo "===== task ${TASK_ID} finished: $(date) ====="
done

echo "===== aggregation started: $(date) ====="
"${PYTHON_BIN}" "${STUDY_DIR}/aggregate_fixed_geo_three_model.py" \
  --config "${STUDY_DIR}/fixed_geo_three_model.toml" \
  --design "${STUDY_DIR}/frozen_design.json" \
  --dates "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}"
echo "===== aggregation finished: $(date) ====="
echo "Single-GPU sequential run finished: $(date)"
