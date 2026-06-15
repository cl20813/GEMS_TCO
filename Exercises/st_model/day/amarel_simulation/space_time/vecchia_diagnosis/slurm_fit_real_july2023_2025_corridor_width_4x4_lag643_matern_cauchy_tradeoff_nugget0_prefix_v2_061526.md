# Real July 2023-2025 Corridor Lag643 Matérn vs Cauchy Nugget-Zero Tradeoff Test V2

This is the v2 multi-year diagnosis run for the real-data space-time Vecchia
model:

```text
model geometry: corridor-width cluster Vecchia
block shape: 4x4
lag pattern: 6/4/3
ordering: block-center max-min prefixes
prefixes: 100, 200, 400, 600, 800, all
years: 2023, 2024, 2025
region: latitude -3..2, longitude 121..131
nugget: fixed 0
baseline: Matérn smooth=0.3
```

Year-specific model variants:

```text
2023: matern_s03, gc_a075_b05, gc_a075_b1, gc_a08_b1, gc_a08_b05
2024: matern_s03, gc_a08_b05, gc_a08_b1
2025: matern_s03, gc_a075_b1, gc_a075_b05, gc_a08_b1, gc_a08_b05
```

Main outputs per year:

```text
real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_all_fits.csv
real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_monthly_param_summary.csv
real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_monthly_loss_summary.csv
monthly_average_plots/real_YYYY_parameter_median_by_blockmaxmin.png
monthly_average_plots/real_YYYY_parameter_median_by_blockmaxmin_symlog.png
monthly_average_plots/real_YYYY_loss_mean_median_by_blockmaxmin.png
monthly_average_plots/real_2023_2025_loss_per_valid_median_heatmap.png
```

Each SLURM array task writes to its own year subfolder to avoid concurrent
CSV/JSONL writes.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526
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
  "${LOCAL_DIAG}/fit_real_july2023_2025_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526.py" \
  "${LOCAL_DIAG}/slurm_fit_real_july2023_2025_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIAG_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis
nano run_real_july2023_2025_matern_cauchy_tradeoff_nugget0_prefix_v2_061526.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real2325_n0mc_v2
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/real2325_n0mc_v2_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/real2325_n0mc_v2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis/fit_real_july2023_2025_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526"
YEARS=(2023 2024 2025)
YEAR="${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTDIR="${OUTROOT}/year_${YEAR}"
MONTHLY_OUTDIR="${OUTDIR}/monthly_average_plots"
VARIANTS=(matern_s03 gc_a075_b05 gc_a075_b1 gc_a08_b1 gc_a08_b05)

mkdir -p "${OUTDIR}" "${MONTHLY_OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Output root: ${OUTROOT}"
echo "Year output dir: ${OUTDIR}"
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
  --real-years "${YEAR}" \
  --month 7 \
  --days 0,15 \
  --space 1,1 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --model-variants "${VARIANTS[@]}" \
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
sbatch run_real_july2023_2025_matern_cauchy_tradeoff_nugget0_prefix_v2_061526.sh
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
