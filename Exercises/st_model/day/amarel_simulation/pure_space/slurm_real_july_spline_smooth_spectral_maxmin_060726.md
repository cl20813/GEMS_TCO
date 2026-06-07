# Real July Max-Min Spline-Smooth Spectral Diagnostics, 2026-06-07

This run replaces the older regular thinning sequence:

```text
x8, x4, x2, x1
```

with increasingly dense max-min 4x4 block prefixes:

```text
B200, B400, B800, all
```

Reason: regular thinning can alias high-frequency structure into lower
frequencies.  Here selected cells remain on the original grid, and unselected
cells are treated as missing in the periodogram/window calculation.

```text
years   = 2022, 2023, 2024, 2025
month   = July
days    = July 1..30
smooth  = 0.25, 0.3, 0.35, 0.5
region  = lat -3..2, lon 121..131
Vecchia = 4x4 target blocks, condition on 2 previous max-min cluster blocks
prefix  = first 200, 400, 800 max-min 4x4 block centers, then all blocks
```

Important diagnostic convention:

```text
No Hann/tapering is used.
I and E[I] both use the observed/missing grid directly.
The expected periodogram uses only the missing-data mask/window autocorrelation.
```

Output is organized by year first, then smooth:

```text
/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_maxmin_060726/
  2022_07/
    smooth_0p25/
    smooth_0p3/
    smooth_0p35/
    smooth_0p5/
  2023_07/
  2024_07/
  2025_07/
  monthly_plots_top/
    2022_07/
    2023_07/
    2024_07/
    2025_07/
```

Monthly plots include radial, latitude N-S, longitude E-W, and diagonal NE-SW
profiles.  A separate monthly plot compares fitted finite-sample `E[I]` against
the continuous theoretical spectrum after Whittle-style amplitude profiling.

## Transfer Code

Update the package code:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco"
```

Transfer the run script:

```bash
ssh jl2815@amarel.rutgers.edu \
  'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_maxmin_060726.py" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/"
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_real_july_spline_smooth_spectral_maxmin_060726.sh
sbatch slurm_real_july_spline_smooth_spectral_maxmin_060726.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=real_jul_maxmin
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_jul_maxmin_060726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_jul_maxmin_060726_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SMOOTHS=(0.25 0.3 0.35 0.5)
YEARS=(2022 2023 2024 2025)
MONTH=7

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_maxmin_060726.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_maxmin_060726"
TOPPLOTS="${OUTROOT}/monthly_plots_top"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${TOPPLOTS}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"
echo "Single Slurm job; looping over all smooth/year combinations."
echo "YEARS=${YEARS[*]}"
echo "SMOOTHS=${SMOOTHS[*]}"
echo "Block prefixes: B200 B400 B800 all"
echo "Region: lat -3..2, lon 121..131"
echo "No Hann/tapering; mask/window check only."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

for YEAR in "${YEARS[@]}"; do
  for SMOOTH in "${SMOOTHS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Starting YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
    echo "Current date and time: $(date)"
    echo "============================================================"

    srun python "${SCRIPT}" \
      --years "${YEAR}" \
      --month "${MONTH}" \
      --days "1,30" \
      --smooths "${SMOOTH}" \
      --block-prefixes "200,400,800,all" \
      --variants "nugget0,nugget_free" \
      --cluster-neighbor-blocks 2 \
      --cluster-block-shape 4x4 \
      --mean-design lat \
      --data-root "/home/jl2815/tco/data" \
      --output-root "${OUTROOT}" \
      --top-plot-dir "${TOPPLOTS}" \
      --expanded-bounds \
      --lat-range "-3,2" \
      --lon-range "121,131" \
      --no-hann \
      --device cuda \
      --cuda-fallback cpu \
      --target-chunk-size 512 \
      --lbfgs-steps 8 \
      --lbfgs-eval 20 \
      --lbfgs-history 10 \
      --radial-bins 70 \
      --radial-qmax 0.985 \
      --skip-existing

    echo "Finished YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
    echo "Current date and time: $(date)"
  done
done

echo "All done: $(date)"
```

## Output

Remote output root:

```text
/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_maxmin_060726
```

Important folders:

```text
2022_07/smooth_0p3/
  daily_csv/
    20220701_fits.csv
    20220701_spectral_profiles.csv
  daily_plots/
    20220701_data_vs_expected_periodogram.png
  monthly_average/
    202207_30day_mean_data_vs_expected_periodogram.png
    202207_30day_mean_directional_data_vs_expected_periodogram.png
    202207_expected_periodogram_vs_continuous_theoretical_scaled.png
    202207_daily_mean_curves.csv
    202207_30day_mean_curves.csv

monthly_plots_top/2022_07/
  smooth_0p25_202207_30day_mean_data_vs_expected_periodogram.png
  smooth_0p3_202207_30day_mean_data_vs_expected_periodogram.png
  smooth_0p35_202207_30day_mean_data_vs_expected_periodogram.png
  smooth_0p5_202207_30day_mean_data_vs_expected_periodogram.png
  ...
```

## Transfer Monthly Plots Back

Only copy the monthly plot summary folder:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/spline_smooth_spectral_maxmin_060726"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_maxmin_060726/monthly_plots_top" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/spline_smooth_spectral_maxmin_060726/"
```

Full output fallback:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_maxmin_060726" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
