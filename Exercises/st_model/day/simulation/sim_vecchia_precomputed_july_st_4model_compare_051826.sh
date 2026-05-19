#!/bin/bash
#SBATCH --job-name=vec4_pre_051826
#SBATCH --output=/home/jl2815/tco/exercise_output/vec4_pre_051826_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec4_pre_051826_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="/home/jl2815/tco/exercise_25/st_model/sim_vecchia_precomputed_july_st_4model_compare_051826.py"
SIM_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day"

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "Python: ${PYTHON}"
echo "Current date and time: $(date)"
echo "Using pre-generated simulation assets from: ${SIM_ROOT}"

srun "${PYTHON}" - <<'PY'
import os
import sys
import torch

print("preflight python:", sys.executable, flush=True)
print("preflight torch:", torch.__version__, flush=True)
print("preflight CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("preflight cuda_available:", torch.cuda.is_available(), flush=True)
print("preflight cuda_device_count:", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed before model run.")
print("preflight device0:", torch.cuda.get_device_name(0), flush=True)
PY

srun "${PYTHON}" "${SCRIPT}" \
    --require-cuda \
    --num-iters 100 \
    --seed 123 \
    --years "2022,2023,2024,2025" \
    --day-idxs all \
    --max-asset-days 100 \
    --sim-data-root "${SIM_ROOT}" \
    --data-kind gridded \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --mm-cond-number 100 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --init-noise 0.25 \
    --column-above-count 3 \
    --column-right-col-count 3 \
    --column-per-lag-count 14 \
    --column-lag-count 2 \
    --column-head-right-cols 0 \
    --column-chunk-size 512 \
    --block-shape 3x3 \
    --cluster-hybrid-chunk-size 128 \
    --cluster-hybrid-lag1-local-blocks 3 \
    --cluster-hybrid-lag2-local-blocks 2 \
    --cluster-hybrid-lag1-lon-offset 0.126 \
    --cluster-hybrid-lag2-lon-offset 0.126 \
    --cluster-column-chunk-size 64 \
    --out-dir "${OUTDIR}" \
    --out-prefix "sim_vecchia_precomputed_july_st_4model_compare_051826"

echo "Current date and time: $(date)"
