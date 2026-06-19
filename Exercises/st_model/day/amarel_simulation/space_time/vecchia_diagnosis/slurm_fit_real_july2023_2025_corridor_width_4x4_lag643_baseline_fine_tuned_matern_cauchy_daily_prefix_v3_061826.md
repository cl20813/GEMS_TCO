# Real July 2023-2025 Corridor Lag643 Baseline vs Fine Tuned Daily Diagnosis V3

This is the 2026-06-18 update of the real-data space-time Vecchia diagnosis.

```text
baseline: Vecchia lag 6/4/3, 4x4 = 16-cell block geometry, Matérn smooth=0.3
fine tuned candidates: day-table GC beta values with alpha swept over 0.75, 0.8, 0.9, 1.0
ordering: block-center max-min prefixes
prefixes: 100, 200, 400, 600, 800, all
years: 2023, 2024, 2025
days: July 1-30, i.e. day_idx 0..29
region: latitude -3..2, longitude 121..131
nugget: fixed 0
main question: as data gets denser by block-prefix, how do parameters and loss/observation behave?
```

Model variants:

```text
baseline: matern_s03 is always included
GC sweep token: gc_day_table_sweep
GC alpha values: 0.75, 0.8, 0.9, 1.0
GC beta value: selected day-by-day from the tuning table
Matern-tuned days use fallback GC beta=0.5 for the alpha sweep
```

Main outputs per year:

```text
real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826_all_fits.csv
real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826_monthly_param_summary.csv
real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826_monthly_loss_summary.csv
real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826_missing.csv
monthly_average_plots/real_YYYY_parameter_median_by_blockmaxmin.png
monthly_average_plots/real_YYYY_parameter_median_by_blockmaxmin_symlog.png
monthly_average_plots/real_YYYY_loss_mean_median_by_blockmaxmin.png
monthly_average_plots/real_2023_2025_loss_per_valid_median_heatmap.png
daily_plots/year_YYYY/real_YYYY_dayDD_parameter_loss_by_blockmaxmin.png
```

Loss shown in legends and heatmap cells is the final Vecchia objective per
observation returned by the engine. The CSV keeps the legacy column name
`loss_per_valid`, but it is now an alias of `loss`, not `loss / n_valid_o3`.

The symlog monthly parameter plot is a secondary diagnostic only. It helps when
range or variance estimates differ by orders of magnitude while advection values
remain near zero. The ordinary linear monthly plot and the daily block-prefix
plots are the primary comparisons.

Each SLURM array task writes to its own year subfolder to avoid concurrent
CSV/JSONL writes.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826
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
  "${LOCAL_DIAG}/fit_real_july2023_2025_corridor_width_4x4_lag643_baseline_fine_tuned_matern_cauchy_daily_prefix_v3_061826.py" \
  "${LOCAL_DIAG}/slurm_fit_real_july2023_2025_corridor_width_4x4_lag643_baseline_fine_tuned_matern_cauchy_daily_prefix_v3_061826.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIAG_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis
nano run_real_july2023_2025_baseline_fine_tuned_daily_v3_061826.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real2325_bft_dv3
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/real2325_bft_dv3_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/real2325_bft_dv3_%A_%a.err
#SBATCH --time=12:00:00
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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis/fit_real_july2023_2025_corridor_width_4x4_lag643_baseline_fine_tuned_matern_cauchy_daily_prefix_v3_061826.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826"
YEARS=(2023 2024 2025)
YEAR="${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTDIR="${OUTROOT}/year_${YEAR}"
MONTHLY_OUTDIR="${OUTDIR}/monthly_average_plots"
VARIANTS=(matern_s03 gc_day_table_sweep)

mkdir -p "${OUTDIR}" "${MONTHLY_OUTDIR}" "${OUTDIR}/daily_plots"
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
  --days 0,30 \
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
  --summary-every 30 \
  --suppress-fit-prints \
  --out-dir "${OUTDIR}" \
  --monthly-out-dir "${MONTHLY_OUTDIR}"

echo "Finished: $(date)"
```

Submit:

```bash
sbatch run_real_july2023_2025_baseline_fine_tuned_daily_v3_061826.sh
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_baseline_fine_tuned_daily_prefix_v3_061826" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
