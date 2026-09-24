#!/bin/bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-full}"

mkdir -p /home/jl2815/tco/exercise_output/summer/logs

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

EXISTING_JOBS="$(squeue -h -u "$(id -un)" -n stint3_seq -o '%A' | paste -sd, -)"
if [[ -n "${EXISTING_JOBS}" ]]; then
  echo "Refusing to submit a duplicate stint3_seq job." >&2
  echo "Existing job ID(s): ${EXISTING_JOBS}" >&2
  echo "Inspect with: squeue -j ${EXISTING_JOBS}" >&2
  exit 3
fi

SEQUENTIAL_JOB="$(sbatch --parsable --export="ALL,RUN_MODE=${MODE}" "${STUDY_DIR}/slurm_fixed_geo_three_model_sequential.sh")"

echo "Submitted exactly one GPU job: ${SEQUENTIAL_JOB} (${MODE})"
echo "No Slurm array: dates and three models run sequentially on one GPU."
echo "Aggregation runs at the end of the same job."
echo "Monitor: squeue -u jl2815"
echo "Log: tail -f /home/jl2815/tco/exercise_output/summer/logs/stint3_seq_${SEQUENTIAL_JOB}.out"
echo "Accounting: sacct -j ${SEQUENTIAL_JOB} --format=JobID,State,Elapsed,MaxRSS,ExitCode"
