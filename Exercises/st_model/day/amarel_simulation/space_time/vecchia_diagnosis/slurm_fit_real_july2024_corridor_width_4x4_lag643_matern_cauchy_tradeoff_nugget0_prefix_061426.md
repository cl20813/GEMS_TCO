# Real July 2024 Corridor Lag643 Matérn vs Cauchy Nugget-Zero Tradeoff Test

This run focuses only on July 2024, using the same real-data space-time Vecchia
geometry as the earlier 2023-2025 run:

```text
model geometry: corridor-width cluster Vecchia
block shape: 4x4
lag pattern: 6/4/3
ordering: block-center max-min prefixes
prefixes: 100, 200, 400, 600, 800, all
year: 2024 only
days: July day_idx 0..14 by default
nugget: fixed 0
```

Model variants:

```text
matern_s03      baseline Matérn smooth=0.3
gc_a075_b1      main GC candidate from pure-space checks
gc_a07_b1       lower-alpha b=1 sensitivity
gc_a08_b1       higher-alpha b=1 sensitivity
gc_a07_b05      low-beta tradeoff candidate
gc_a075_b05     low-beta tradeoff candidate
```

Main outputs:

```text
real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_all_fits.csv
real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_monthly_param_summary.csv
real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_monthly_loss_summary.csv
monthly_average_plots/real_2024_parameter_median_by_blockmaxmin.png
monthly_average_plots/real_2024_parameter_median_by_blockmaxmin_symlog.png
monthly_average_plots/real_2024_loss_mean_median_by_blockmaxmin.png
monthly_average_plots/real_2024_loss_per_valid_median_heatmap.png
```

The parameter-tracking plot legend includes the mean final negative likelihood
loss for each model. Use `loss_per_valid_median` for prefix comparisons, because
raw final objectives change with the number of valid observations.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426
```

## 1. Upload From Local Mac

Run from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
REMOTE_DIAG_DIR="${REMOTE_DIR}/vecchia_diagnosis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIAG="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_diagnosis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIAG_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_DIAG}/fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426.py" \
  "${LOCAL_DIAG}/slurm_fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIAG_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis
nano run_real_july2024_matern_cauchy_tradeoff_nugget0_prefix_061426.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real24_n0mc
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/real24_n0mc_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/real24_n0mc_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840 || true
module load cuda/12.1.0 || true

if ! command -v conda >/dev/null 2>&1; then
  source "${HOME}/.bashrc" || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found after module load and ~/.bashrc fallback." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis/fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426.py"
OUTDIR="/home/jl2815/tco/exercise_output/summer/real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426"
MONTHLY_OUTDIR="${OUTDIR}/monthly_average_plots"

mkdir -p "${OUTDIR}" "${MONTHLY_OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
which python
nvidia-smi || true
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

python "${SCRIPT}" \
  --real-years 2024 \
  --month 7 \
  --days 0,15 \
  --space 1,1 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --model-variants matern_s03 gc_a075_b1 gc_a07_b1 gc_a08_b1 gc_a07_b05 gc_a075_b05 \
  --block-prefixes 100 200 400 600 800 all \
  --nugget-mode zero \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 32 \
  --min-target-points 1 \
  --spline-n-points 4000 \
  --spline-r-max 30.0 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --grad-tol 1e-5 \
  --keep-exact-loc \
  --cuda-fallback error \
  --skip-existing \
  --summary-every 1 \
  --suppress-fit-prints \
  --out-dir "${OUTDIR}" \
  --monthly-out-dir "${MONTHLY_OUTDIR}"

echo "Finished: $(date)"
```

Submit:

```bash
sbatch run_real_july2024_matern_cauchy_tradeoff_nugget0_prefix_061426.sh
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_july2024_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
