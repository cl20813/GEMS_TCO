#!/bin/bash -l
#SBATCH --job-name=stint3gc_seq
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stint3gc_seq_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stint3gc_seq_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
# Direct submission defaults to the known A100 pool. The supported submit
# wrapper overrides this with the combined compatible pool or a single-family
# A100/L40S pool.
#SBATCH --nodelist=gpu[015-017,019-028]
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_PROJECT="/home/jl2815/tco/GEMS_TCO-1"
STUDY_DIR="${REMOTE_PROJECT}/Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/fixed_geographic_three_model_092426"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/fixed_geographic_three_model_gc128_native_092526"
MANIFEST="${STUDY_DIR}/evaluation_dates.csv"
RUN_MODE="${RUN_MODE:-full}"
GPU_PROFILE="${GPU_PROFILE:-compatible}"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${STUDY_DIR}/run_fixed_geo_three_model_day.py" || {
  echo "Missing study runner under ${STUDY_DIR}" >&2
  exit 2
}
test -f "${MANIFEST}" || { echo "Missing manifest: ${MANIFEST}" >&2; exit 2; }

module use /projects/community/modulefiles
module load cuda/12.1.0

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
echo "GPU profile: ${GPU_PROFILE}"
echo "Task order: ${TASK_IDS[*]}"
echo "Study: fixed geographic, frozen A/B, lag 1, GC/JM0.5/advected-separable"
echo "Vecchia compute setting: corridor lag 6/4/3, GC chunk 128, Matérn/separable chunk 256"
echo "Output: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Slurm resources: nodes=${SLURM_JOB_NUM_NODES:-unset}, tasks=${SLURM_NTASKS:-unset}, cpus=${SLURM_CPUS_PER_TASK:-unset}, mem-per-node-MB=${SLURM_MEM_PER_NODE:-unset}, time-limit=${SLURM_TIMELIMIT:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_PROFILE="${GPU_PROFILE}" \
"${PYTHON_BIN}" -c '
import os
import torch

assert torch.cuda.is_available(), "CUDA is unavailable in the allocated job"
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
capability = f"{major}.{minor}"
profile = os.environ["GPU_PROFILE"]
allowed = {
    "compatible": (((8, 0), "A100"), ((8, 9), "L40S")),
    "a100": (((8, 0), "A100"),),
    "l40s": (((8, 9), "L40S"),),
}
if profile not in allowed:
    raise SystemExit(f"unsupported GPU_PROFILE={profile!r}")
print("torch", torch.__version__, "built CUDA", torch.version.cuda)
print("allocated device", name, "compute capability", capability)
if not any((major, minor) == cc and label in name.upper() for cc, label in allowed[profile]):
    raise SystemExit(
        f"allocated GPU {name!r} (sm_{major}{minor}) is incompatible with "
        f"requested profile {profile!r}"
    )
'

# Build on the allocated Linux/CUDA host. Local macOS binaries are excluded
# from upload and are never reused on Amarel. The build script runs both the
# Matérn and generalized-Cauchy CUDA covariance/gradient parity suites before
# any expensive daily fit starts.
export PYTHON="${PYTHON_BIN}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-12}"
# Always compile native cubins for both production families.  The build helper
# deliberately ignores an unrelated ambient TORCH_CUDA_ARCH_LIST.
export GEMS_TCO_CUDA_ARCH_LIST="8.0;8.9"
bash "${REMOTE_PROJECT}/scripts/amarel/build_and_test_cuda.sh" \
  --batch-size 16 --points 224 --iterations 3
"${PYTHON_BIN}" -c \
  'from GEMS_TCO.vecchia._native_covariance import native_generalized_cauchy_covariance_available; assert native_generalized_cauchy_covariance_available("cuda"); print("GC CUDA native backend verified")'

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
