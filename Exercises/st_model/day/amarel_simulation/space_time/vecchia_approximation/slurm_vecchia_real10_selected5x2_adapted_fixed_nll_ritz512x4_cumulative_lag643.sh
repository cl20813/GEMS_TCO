#!/bin/bash
#SBATCH --job-name=vecc10_r512x4
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc10_r512x4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc10_r512x4_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu
# gpu017 returned CUDA unknown error on 2026-09-08; keep it out of this run.
#SBATCH --nodelist=gpu[015-016,019-028]
#SBATCH --gres=gpu:1

set -euo pipefail

REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
PYTHON_BIN="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="${REMOTE_DIR}/vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643.py"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643"

test -x "${PYTHON_BIN}" || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${SCRIPT}" || { echo "Missing script: ${SCRIPT}" >&2; exit 2; }

export PYTHONPATH="/home/jl2815/tco:${REMOTE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/jl2815_matplotlib_${SLURM_JOB_ID:-manual}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" \
  /home/jl2815/tco/exercise_output/summer/logs

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Test dates: July 3, 5, 7, 12, 15 in both 2024 and 2025 (10 dates)"
echo "Methods: adapted and fixed lag-6/4/3"
echo "Diagnostics: native full-data NLL; full-data SLQ/Ritz-512x4 cumulative y=x"
echo "Dense 400x8 eigenanalysis: omitted"
echo "SLQ: 8 paired probes x 256; Ritz: 4 paired starts x 1536 candidates"
echo "Selected per start: 170/170/172 = 512; curves averaged across starts"
echo "Output: ${OUTPUT_ROOT}"
echo "SLURM job/node: ${SLURM_JOB_ID:-unset} / $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"${PYTHON_BIN}" -c 'import torch; print("torch", torch.__version__, "built CUDA", torch.version.cuda); print("CUDA available", torch.cuda.is_available(), "count", torch.cuda.device_count()); assert torch.cuda.is_available(); print("device 0", torch.cuda.get_device_name(0))'

"${PYTHON_BIN}" "${SCRIPT}" \
  --real-data-root /home/jl2815/tco/data \
  --output-root "${OUTPUT_ROOT}" \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --smooth 0.5 \
  --target-chunk-size 256 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 40 \
  --grad-tol 1e-5 \
  --slq-probes 8 \
  --slq-steps 256 \
  --ritz-candidate-steps 1536 \
  --ritz-replicates 4 \
  --band-mode-counts=170,170,172 \
  --spectrum-grid 400 \
  --random-seed 20260907 \
  --ritz-relative-residual-tolerance 0.05 \
  --quality-warning-fraction 0.85 \
  --precision-identity-tolerance 1e-8 \
  --coefficient-drop-tolerance 0 \
  --confidence-z 1.96 \
  --suppress-fit-prints

echo "Finished: $(date)"
echo "Re-submit this same file if interrupted; completed date/method caches are reused."
