#!/bin/bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-full}"
GPU_PROFILE="${2:-compatible}"

# Resource requests are intentionally centralized here.  Command-line sbatch
# options override the conservative defaults in the Slurm script and make the
# selected GPU family visible before submission.
JOB_TIME="${GEMS_TCO_SLURM_TIME:-05:00:00}"
JOB_MEMORY="${GEMS_TCO_SLURM_MEM:-128G}"
JOB_CPUS="${GEMS_TCO_SLURM_CPUS:-12}"

mkdir -p /home/jl2815/tco/exercise_output/summer/logs

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [smoke|full] [compatible|a100|l40s]" >&2
  exit 2
fi

case "${GPU_PROFILE}" in
  compatible)
    # One scheduler request may land on either supported architecture. gpu018
    # is intentionally excluded because it is currently invalid.
    GPU_NODELIST="${GEMS_TCO_COMPATIBLE_NODELIST:-gpu[015-017,019-048]}"
    EXPECTED_GPU_DESCRIPTION="A100 sm_80 or L40S sm_89"
    ;;
  a100)
    GPU_NODELIST="${GEMS_TCO_A100_NODELIST:-gpu[015-017,019-028]}"
    EXPECTED_GPU_DESCRIPTION="A100 sm_80"
    ;;
  l40s)
    # These are the current Ada candidates.  The allocated device is checked
    # with PyTorch/nvidia-smi inside the job and must actually identify as an
    # L40S before compilation or fitting starts.
    GPU_NODELIST="${GEMS_TCO_L40S_NODELIST:-gpu[029-048]}"
    EXPECTED_GPU_DESCRIPTION="L40S sm_89"
    ;;
  *)
    echo "GPU profile must be compatible, a100, or l40s, got: ${GPU_PROFILE}" >&2
    exit 2
    ;;
esac

EXISTING_JOBS="$(squeue -h -u "$(id -un)" -n stint3gc_seq -o '%A' | paste -sd, -)"
if [[ -n "${EXISTING_JOBS}" ]]; then
  echo "Refusing to submit a duplicate stint3gc_seq job." >&2
  echo "Existing job ID(s): ${EXISTING_JOBS}" >&2
  echo "Inspect with: squeue -j ${EXISTING_JOBS}" >&2
  exit 3
fi

echo "Submitting one sequential ${GPU_PROFILE} job with:"
echo "  partition=gpu"
echo "  nodelist=${GPU_NODELIST}"
echo "  gpu=1, cpus=${JOB_CPUS}, system-memory=${JOB_MEMORY}, time=${JOB_TIME}"
echo "  accepted device=${EXPECTED_GPU_DESCRIPTION}"

SEQUENTIAL_JOB="$(
  sbatch --parsable \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${JOB_CPUS}" \
    --mem="${JOB_MEMORY}" \
    --time="${JOB_TIME}" \
    --gres=gpu:1 \
    --nodelist="${GPU_NODELIST}" \
    --export="ALL,RUN_MODE=${MODE},GPU_PROFILE=${GPU_PROFILE}" \
    "${STUDY_DIR}/slurm_fixed_geo_three_model_sequential.sh"
)"

echo "Submitted exactly one GPU job: ${SEQUENTIAL_JOB} (${MODE}, ${GPU_PROFILE})"
echo "No Slurm array: dates and three models run sequentially on one GPU."
echo "Aggregation runs at the end of the same job."
echo "Monitor: squeue -u jl2815"
echo "Log: tail -f /home/jl2815/tco/exercise_output/summer/logs/stint3gc_seq_${SEQUENTIAL_JOB}.out"
echo "Accounting: sacct -j ${SEQUENTIAL_JOB} --format=JobID,State,Elapsed,MaxRSS,ExitCode"
