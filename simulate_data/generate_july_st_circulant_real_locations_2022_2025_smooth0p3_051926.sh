

#!/bin/bash
#SBATCH --job-name=sim_july_st_s03_051926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_july_st_s03_051926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_july_st_s03_051926_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=main

set -euo pipefail

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.py"
OUTDIR="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3"

echo "Running on: $(hostname)"
echo "Python: ${PYTHON}"
echo "Script: ${SCRIPT}"
echo "Output directory: ${OUTDIR}"
echo "Current date and time: $(date)"

srun "${PYTHON}" - <<'PY'
import sys
import numpy
import pandas
import scipy
import torch

print("preflight python:", sys.executable, flush=True)
print("preflight numpy:", numpy.__version__, flush=True)
print("preflight pandas:", pandas.__version__, flush=True)
print("preflight scipy:", scipy.__version__, flush=True)
print("preflight torch:", torch.__version__, flush=True)
PY

srun "${PYTHON}" "${SCRIPT}" \
    --years "2022,2023,2024,2025" \
    --input-root "/home/jl2815/tco/data" \
    --output-dir "${OUTDIR}" \
    --max-hours 248 \
    --hours-per-day 8 \
    --seed 20240701 \
    --smooth 0.3 \
    --spline-n-points 4000 \
    --spline-r-max 30.0 \
    --sigmasq 10.0 \
    --range-lat 0.2 \
    --range-lon 0.3 \
    --range-time 2.0 \
    --advec-lat 0.08 \
    --advec-lon -0.2 \
    --nugget 1.0 \
    --mean-intercept 260.0 \
    --mean-lat-slope 1.0 \
    --mean-lat-center -0.5 \
    --lat-range="-3,2" \
    --lon-range="121,131" \
    --lat-factor-hr 100 \
    --lon-factor-hr 10 \
    --hr-pad 0.1

echo "Current date and time: $(date)"
