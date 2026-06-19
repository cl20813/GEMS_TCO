# Simulated July ST Vecchia Conditional Eigen Diagnostic, Smooth 0.3 Nugget 0

This run compares two full-domain daily ST Vecchia conditional eigen diagnostics
on the reusable July simulated data:

```text
data source: smooth=0.3 Matérn ST circulant simulation, nugget=0
input pickle: sim_julyYYYY_st_circulant_gridded.pkl
domain per day: about 18,000 grid cells x 8 hourly slots
Vecchia geometry: corridor-width 4x4 target blocks, lag pattern 6/4/3
models:
  matern_s03: Matérn smooth=0.3, nugget fixed 0
  gc_a075_b05: generalized Cauchy alpha=0.75 beta=0.5, nugget fixed 0
diagnostic: conditional target-block covariance eigenbasis from Vecchia factorization,
            with fitted mean-design column space projected out
loss label: Vecchia objective per target observation, printed to 5 decimals
```

This is not a dense `144000 x 144000` covariance eigendecomposition.  For each
Vecchia target block, the script eigendecomposes the small conditional
covariance

```text
Cov(Y_target | Y_conditioning)
```

and pools the normalized squared scores over the full day.  That keeps the
full-domain missing pattern and ST covariance in the diagnostic without forming
a huge dense matrix.

The plotted curve uses the residual projection

```text
P = I - Z (Z'Z)^-1 Z'
```

in the conditional-eigen score coordinates `z ~ N(Z beta, I)`.  To preserve the
conditional-eigenvalue ordering, the curve accumulates projected squared
coordinate residuals against their leverage-adjusted expected increments
`1 - h_j`.

The diagnostic stage keeps conditional scores, projection/leverage, sorting,
and cumulative sums on the torch device until the final CSV/plot materialization.
The default SLURM command uses `--diag-chunk-size 64`; reduce it if CUDA memory
is tight.

Because the plot preserves the original conditional-eigenvalue ordering after
projecting out the mean design, the coordinate residuals are leverage-adjusted
but not mutually independent.  The dashed bands are therefore diagnostic
reference bands, not exact independent chi-square acceptance bands.

Main outputs:

```text
sim_july_st_s03_n0_vecchia_conditional_eig_matern_gc075b05_061826_summary.csv
daily_plots/year_YYYY/sim_YYYY_dayDD_vecchia_conditional_eig_comparison.png
sim_YYYY_vecchia_conditional_eig_daily_resampled_curves.csv
sim_YYYY_vecchia_conditional_eig_monthly_average_curves.csv
monthly_average_plots/sim_YYYY_monthly_average_vecchia_conditional_eig_comparison.png
run_config.json
```

Use `--save-daily-curves` only if you want the full daily curve CSVs; they can
be large because each day has one score per valid target observation.

## 1. Upload From Local Mac

Run from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/eigen_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/vecchia_conditional_eig_sim_july_st_smooth0p3_matern_gc075b05_nugget0_061826.py" \
  "${LOCAL_DIR}/slurm_vecchia_conditional_eig_sim_july_st_smooth0p3_matern_gc075b05_nugget0_061826.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

Expected simulation inputs on Amarel:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2022_july_st_circulant/sim_july2022_st_circulant_gridded.pkl
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2023_july_st_circulant/sim_july2023_st_circulant_gridded.pkl
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2024_july_st_circulant/sim_july2024_st_circulant_gridded.pkl
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2025_july_st_circulant/sim_july2025_st_circulant_gridded.pkl
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis
nano run_vecchia_conditional_eig_sim_s03_gc075b05_061826.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=st_cond_eig_s03
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/st_cond_eig_s03_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/st_cond_eig_s03_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis/vecchia_conditional_eig_sim_july_st_smooth0p3_matern_gc075b05_nugget0_061826.py"
DATA_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0"
OUTROOT="/home/jl2815/tco/exercise_output/summer/sim_july_st_s03_n0_vecchia_conditional_eig_matern_gc075b05_061826"
YEARS=(2023)
YEAR="${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTDIR="${OUTROOT}/year_${YEAR}"

mkdir -p "${OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTDIR}"
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
  --data-root "${DATA_ROOT}" \
  --sim-kind gridded \
  --years "${YEAR}" \
  --month 7 \
  --days 0,30 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --model-variants matern_s03 gc_a075_b05 \
  --keep-exact-loc \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 32 \
  --diag-chunk-size 64 \
  --min-target-points 1 \
  --spline-n-points 4000 \
  --spline-r-max 30.0 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 8 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --grad-tol 1e-5 \
  --device cuda \
  --cuda-fallback error \
  --resample-grid 200 \
  --suppress-fit-prints \
  --out-dir "${OUTDIR}"

echo "Finished: $(date)"
```

Submit:

```bash
sbatch run_vecchia_conditional_eig_sim_s03_gc075b05_061826.sh
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/sim_july_st_s03_n0_vecchia_conditional_eig_matern_gc075b05_061826" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
