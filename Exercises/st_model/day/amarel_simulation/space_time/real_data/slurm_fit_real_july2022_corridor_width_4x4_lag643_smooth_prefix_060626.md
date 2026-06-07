# 2022 Real July Corridor Lag643 Smooth-Prefix Validation

This run tests only 2022 July real data with the space-time corridor Vecchia
model:

```text
model: corridor-width cluster Vecchia
block shape: 4x4
lag pattern: 6/4/3
smooth candidates: 0.25, 0.30, 0.35
block-center max-min prefixes: 100, 200, 400, 600, 800, all
days: July 1-15 by default
region: lat -3..2, lon 121..131
```

The output includes:

```text
real_july2022_corridor_lag643_smooth_prefix_all_fits.csv
real_july2022_corridor_lag643_smooth_prefix_monthly_summary.csv
real_july2022_corridor_lag643_smooth_prefix_likelihood_table.csv
monthly_average_plots/real_parameter_by_blockmaxmin.png
monthly_average_plots/real_2022_likelihood_heatmap_loss-per-valid-median.png
monthly_average_plots/real_2022_likelihood_heatmap_loss-median.png
```

For comparing across different prefix sizes, use `loss_per_valid_median`.  The
raw `loss_median` is also saved, but raw likelihood/objective values change with
the number of valid observations.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/real_july2022_corridor_lag643_smooth_prefix_060626
```

## 1. Upload From Local Mac

Run from the local Mac, not from Amarel:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
REMOTE_REAL_DIR="${REMOTE_DIR}/real_data"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_ST="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time"
LOCAL_REAL="${LOCAL_ST}/real_data"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} ${REMOTE_REAL_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_ST}/fit_july2024_st_corridor_density_sweep_060426.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"

scp \
  "${LOCAL_REAL}/fit_real_july2022_corridor_width_4x4_lag643_smooth_prefix_060626.py" \
  "${LOCAL_REAL}/slurm_fit_real_july2022_corridor_width_4x4_lag643_smooth_prefix_060626.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_REAL_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data
nano run_real_july2022_corridor_lag643_smooth_prefix_060626.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real22_smoothpref
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/real22_smoothpref_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/real22_smoothpref_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data/fit_real_july2022_corridor_width_4x4_lag643_smooth_prefix_060626.py"
OUTDIR="/home/jl2815/tco/exercise_output/summer/real_july2022_corridor_lag643_smooth_prefix_060626"
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
  --datasets real \
  --real-years 2022 \
  --month 7 \
  --days 0,15 \
  --space 1,1 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --real-fit-smooths 0.25 0.3 0.35 \
  --real-fit-smooths-by-year "2022:0.25,0.3,0.35" \
  --block-prefixes 100 200 400 600 800 all \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 128 \
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
sbatch run_real_july2022_corridor_lag643_smooth_prefix_060626.sh
```

## 3. Pull Monthly Plots And Tables To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_july2022_corridor_lag643_smooth_prefix_060626" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```

