#!/bin/bash
#SBATCH --job-name=vecc10_e512l
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/vecc10_e512l_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/vecc10_e512l_%j.err
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
SCRIPT="${REMOTE_DIR}/vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_lag643.py"
OUTPUT_ROOT="/home/jl2815/tco/exercise_output/summer/vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_light_tol1e4_lag643"

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
echo "Dates: July 3, 5, 7, 12, 15 in 2024 and 2025"
echo "Methods: adapted and fixed lag-6/4/3"
echo "Primary statistic: z_j=sqrt(omega_j)*(u_j' residual), e_j=z_j^2"
echo "Curve: x_k=k/512, y_k=sum_{j<=k} e_j/512 (same dense full-eigen logic)"
echo "Eigenpairs: one approximate set, 170 smallest + 170 median-nearest + 172 largest"
echo "SLQ: boundary/median estimation only; no residual quadrature"
echo "Random-start averaging: none"
echo "eigsh tolerance: 1e-4; residual quality threshold: 1e-3 (reported, not hard gate)"
echo "Hard checks: finite positive modes, 512 modes available, max |U'U-I| <=1e-5"
echo "Output: ${OUTPUT_ROOT}"
echo "SLURM job/node: ${SLURM_JOB_ID:-unset} / $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"${PYTHON_BIN}" -c 'import numpy, scipy, torch; print("numpy", numpy.__version__, "scipy", scipy.__version__, "torch", torch.__version__); print("CUDA", torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

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
  --slq-probes 4 \
  --slq-steps 128 \
  --band-mode-counts=170,170,172 \
  --spectrum-grid 400 \
  --eigen-oversample 32 \
  --eigsh-tolerance 1e-4 \
  --eigsh-maxiter 2000 \
  --eigsh-ncv-factor 2.0 \
  --eigsh-ncv-extra 24 \
  --eigenpair-relative-residual-tolerance 1e-3 \
  --eigenvector-orthogonality-tolerance 1e-5 \
  --random-seed 20260907 \
  --precision-identity-tolerance 1e-8 \
  --coefficient-drop-tolerance 0 \
  --suppress-fit-prints

echo "Finished: $(date)"
echo "Re-submit this file if the wall time interrupts it; completed date/method caches are reused."
