#!/bin/bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-full}"
JOB_TIME="${GEMS_TCO_SLURM_TIME:-12:00:00}"
JOB_MEMORY="${GEMS_TCO_SLURM_MEM:-128G}"
JOB_CPUS="${GEMS_TCO_SLURM_CPUS:-12}"
GPU_NODELIST="${GEMS_TCO_A100_NODELIST:-gpu[015-017,019-028]}"

if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [pilot|full]" >&2
  exit 2
fi

mkdir -p /home/jl2815/tco/exercise_output/fall_26/logs
existing="$(squeue -h -u "$(id -un)" -n tcfit643 -o '%A' | paste -sd, -)"
if [[ -n "${existing}" ]]; then
  echo "Refusing duplicate tcfit643 job; existing=${existing}" >&2
  exit 3
fi

job_id="$(
  sbatch --parsable \
    --partition=gpu --nodes=1 --ntasks=1 \
    --cpus-per-task="${JOB_CPUS}" --mem="${JOB_MEMORY}" --time="${JOB_TIME}" \
    --gres=gpu:1 --nodelist="${GPU_NODELIST}" \
    --export="ALL,RUN_MODE=${MODE},GPU_PROFILE=a100" \
    "${STUDY_DIR}/slurm_fitted_cross_643.sbatch"
)"

echo "Submitted one sequential A100 job: ${job_id} (${MODE})"
echo "No array; six scenarios and their days run sequentially."
echo "Monitor: squeue -j ${job_id}"
echo "Log: tail -f /home/jl2815/tco/exercise_output/fall_26/logs/tcfit643_${job_id}.out"
