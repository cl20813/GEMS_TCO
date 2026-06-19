# Real July 2023 ST Vecchia Conditional Eigen Diagnostic

This run applies the conditional-eigen Vecchia diagnostic to real July GEMS
data rather than simulation pickles.

```text
years       = 2023
month       = July
days        = day_idx 0..29, i.e. first 30 complete 8-hour days
Vecchia     = corridor-width 4x4 target blocks, lag pattern 6/4/3
nugget      = fixed 0
domains     = full x1 and 2x4 spatial tiles
daily plots = full x1 single plot, plus one 2x4 tile panel per day
monthly     = refreshed after each completed domain/day; full and tile outputs use separate top folders
```

Default model comparison:

```text
1. baseline Matérn smooth=0.3
2. baseline GC a=0.75, b=1
3. fine_tuned_gc: day-specific GC alpha/beta table
```

Fine-tuned GC table:

```csv
day,gc_alpha,gc_beta
1,0.75,0.5
2,0.75,4
3,0.75,0.5
4,0.75,4
5,0.8,0.5
6,0.9,0.5
7,0.9,0.5
8,0.75,1
9,1,0.5
10,0.75,0.5
11,0.9,1
12,0.8,3
13,0.9,0.5
14,1,0.5
15,0.8,0.5
16,0.8,1
17,0.8,0.5
18,0.75,0.5
19,0.8,1
20,0.9,0.5
21,0.9,0.5
22,0.75,0.5
23,0.8,1
24,0.75,0.5
25,0.9,0.5
26,0.75,0.5
27,0.75,0.5
28,0.9,0.5
29,0.8,0.5
30,0.9,0.5
```

The full x1 domain uses every grid cell in the requested lat/lon box for all
8 hourly slots.  The tile domain splits the same box into a `2x4` grid and
fits/diagnoses each tile separately, which is the high-frequency/local check.

Main outputs:

```text
real_july2023_vecchia_conditional_eigen_sort_matern_gc_baselines_finetuned_domains_061926_summary.csv
daily_plots_full/year_2023/real_2023_dayDD_full_vecchia_conditional_eigen_sort_comparison.png
daily_plots_tile_2x4/year_2023/real_2023_dayDD_tile_2x4_vecchia_conditional_eigen_sort_panel.png
monthly_average_full/year_2023/full/full/
monthly_average_tile_2x4/year_2023/tile_2x4/tile_y01_x01/
monthly_average_plots_full/year_2023/full/
monthly_average_plots_tile_2x4/year_2023/real_2023_tile_2x4_monthly_average_vecchia_conditional_eigen_sort_comparison.png
run_config.json
```

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
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_common_engine_061926.py" \
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_real_july2023_matern_gc_baselines_finetuned_domains_061926.py" \
  "${LOCAL_DIR}/slurm_vecc_con_eigen_sort_real_july2023_matern_gc_baselines_finetuned_domains_061926.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

Expected real-data inputs on Amarel:

```text
/home/jl2815/tco/data/pickle_2023/tco_grid_23_07.pkl
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis
nano run_vecc_con_eigen_sort_real_july2023_baselines_finetuned_domains_061926.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real23_eig_ft
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/real23_eig_ft_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/real23_eig_ft_%j.err
#SBATCH --time=12:00:00
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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis/vecchia_conditional_eigen_sort_real_july2023_matern_gc_baselines_finetuned_domains_061926.py"
DATA_ROOT="/home/jl2815/tco/data"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_vecchia_conditional_eigen_sort_matern_gc_baselines_finetuned_domains_061926"
YEAR="2023"
OUTDIR="${OUTROOT}/year_${YEAR}"

mkdir -p "${OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
nvidia-smi || true
python - <<'PY'
import numpy, pandas, scipy, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

python "${SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --years "${YEAR}" \
  --month 7 \
  --days "0,30" \
  --space "1,1" \
  --lat-range="-3,2" \
  --lon-range="121,131" \
  --domain-modes "full,tile_2x4" \
  --tile-grid "2x4" \
  --daily-plot-domains "full" \
  --model-variants year_default \
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
sbatch run_vecc_con_eigen_sort_real_july2023_baselines_finetuned_domains_061926.sh
```

## 3. Monitor

```bash
squeue -u jl2815

tail -f /home/jl2815/tco/exercise_output/summer/logs/real23_eig_ft_<JOBID>.out
```

## 4. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_vecchia_conditional_eigen_sort_matern_gc_baselines_finetuned_domains_061926" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
