# Real July 2022-2025 Implied Spatial Ratio Plot

Canonical implied-spatial ratio runbook.

- years: 2022, 2023, 2024, 2025
- region: latitude `-3..2`, longitude `121..131`
- ST model: daily Vecchia fit, temporal lag-zero implied spatial covariance
- variant: `nugget0` only
- resolution expansion: first `ceil(n_blocks / stride^2)` 4x4 block centers in max-min order, keeping all cells inside selected blocks
- profiles: norm, latitude, longitude, and NE-SW diagonal
- wording: plot axes use `norm frequency`

The Python script is:

```text
/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2022_2025_implied_spatial_ratio_plot.py
```

## Transfer

Run from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_SPECTRAL="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/pure_space/spectral_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} /home/jl2815/tco /home/jl2815/tco/exercise_output/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_SPECTRAL}/real_july2022_2025_implied_spatial_ratio_plot.py" \
  "${LOCAL_SPECTRAL}/slurm_real_july2022_2025_implied_spatial_ratio_plot.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis
nano run_real_july2022_2025_implied_spatial_ratio_plot.sh
sbatch run_real_july2022_2025_implied_spatial_ratio_plot.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=impl_ratio
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/impl_ratio_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/impl_ratio_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2022_2025_implied_spatial_ratio_plot.py"
OUTROOT="/home/jl2815/tco/exercise_output/real_data/real_july2022_2025_implied_spatial_ratio_plot"
YEARS=(2022 2023 2024 2025)
YEAR="${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p "${OUTROOT}" /home/jl2815/tco/exercise_output/logs
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Experiment: implied spatial ratio plot, nugget0, norm/lat/lon/diag profiles"
echo "Output root: ${OUTROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

srun python "${SCRIPT}" \
  --years "${YEAR}" \
  --month 7 \
  --days "1,30" \
  --smooths "0.2,0.25,0.3,0.35,0.4,0.45" \
  --resolutions "8,4,2,1" \
  --resolution-order maxmin \
  --variants "nugget0" \
  --profiles "radial,lat,lon,diag" \
  --data-root "/home/jl2815/tco/data" \
  --output-root "${OUTROOT}" \
  --expanded-bounds \
  --lat-range "-3,2" \
  --lon-range "121,131" \
  --device cuda \
  --cuda-fallback cpu \
  --block-shape "4,4" \
  --n-neighbor-blocks-t 6 \
  --lag1-local-blocks 4 \
  --lag1-shifted-blocks 1 \
  --lag2-local-blocks 3 \
  --lag2-shifted-blocks 1 \
  --daily-stride 2 \
  --lag1-lon-offset 0.126 \
  --target-chunk-size 128 \
  --lbfgs-steps 8 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --radial-bins 70 \
  --radial-qmax 0.985 \
  --skip-existing

echo "Finished: $(date)"
```

## Main Outputs

Remote output root:

```text
/home/jl2815/tco/exercise_output/real_data/real_july2022_2025_implied_spatial_ratio_plot
```

Folder structure:

```text
smooth_0p3/
  2024_07/
    daily_plots/
      20240701_implied_spatial_data_vs_expected_norm.png
      20240701_implied_spatial_data_vs_expected_lat.png
      20240701_implied_spatial_data_vs_expected_lon.png
      20240701_implied_spatial_data_vs_expected_diag.png
    daily_csv/
      20240701_st_fits.csv
      20240701_implied_spatial_spectral_profiles.csv
    monthly_average/
      202407_30day_mean_implied_spatial_data_vs_expected_norm.png
      202407_30day_mean_implied_spatial_data_vs_expected_lat.png
      202407_30day_mean_implied_spatial_data_vs_expected_lon.png
      202407_30day_mean_implied_spatial_data_vs_expected_diag.png
      202407_daily_mean_curves.csv
      202407_30day_mean_curves.csv
```

The plot axes use `norm frequency` wording for the norm profile.

## Pull Results

Run from the local Mac:

```bash
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data/real_july2022_2025_implied_spatial_ratio_plot"
REMOTE_OUT="/home/jl2815/tco/exercise_output/real_data/real_july2022_2025_implied_spatial_ratio_plot"

mkdir -p "${LOCAL_OUT}"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}" \
  "${LOCAL_OUT}/"
```
