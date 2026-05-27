# Real July Spline-Smooth Spectral Diagnostics

This run fits real July GEMS data with pure-space Vecchia Matérn models using
spline-evaluated smoothness values:

```text
smooth = 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
years  = 2022, 2023, 2024, 2025
days   = July 1..30
```

For each smooth/year/day it keeps the reference plot structure:

```text
columns: x8, x4, x2, x1
rows:    nugget fixed 0, nugget free
```

All hourly pure-space fits use cluster/group Vecchia with `4x4` grid-cell
target blocks and conditioning on `2` previous max-min cluster blocks.

The black line is the mean of the 8 hourly residual periodograms. The red line
is the mean of the 8 fitted finite-sample expected periodograms. The blue line
is the ratio of those two means, not the mean of 8 hourly ratios.

The fitted parameter label is `fit median`; it uses the median over the 8 hourly
fits for `sigma^2`, `range`, and `nugget` when nugget is free.

## Expanded July Data

This run uses the expanded July grid:

```text
lat -3..7, lon 111..131
```

Expected Amarel files:

```text
/home/jl2815/tco/data/pickle_2022/tco_grid_lat-3to7_lon111to131_22_07.pkl
/home/jl2815/tco/data/pickle_2023/tco_grid_lat-3to7_lon111to131_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat-3to7_lon111to131_25_07.pkl
```

If the expanded files are not already on Amarel, transfer them from the Mac:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2022 2023 2024 2025 --months 7 --only-extra-bounds
```

Verify on Amarel:

```bash
ssh jl2815@amarel.rutgers.edu '
for y in 2022 2023 2024 2025; do
  yy=${y:2:2}
  ls -lh /home/jl2815/tco/data/pickle_${y}/tco_grid_lat-3to7_lon111to131_${yy}_07.pkl
done
'
```

## Transfer Code

Update the package code, including the independent spline Vecchia wrapper:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

Transfer the run script:

```bash
ssh jl2815@amarel.rutgers.edu \
    'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_052126.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_real_july_spline_smooth_spectral_052126.sh
sbatch slurm_real_july_spline_smooth_spectral_052126.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=real_jul_spl
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_jul_spl_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_jul_spl_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu021

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SMOOTHS=(0.2 0.3 0.4 0.5 0.6 0.7)
YEARS=(2022 2023 2024 2025)
MONTH=7

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_052126.py"
OUTROOT="/home/jl2815/tco/exercise_output/real_data/spline_smooth_spectral_052126"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"
echo "Single Slurm job; looping over all smooth/year combinations."
echo "YEARS=${YEARS[*]}"
echo "SMOOTHS=${SMOOTHS[*]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

for SMOOTH in "${SMOOTHS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
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
            --resolutions "8,4,2,1" \
            --variants "nugget0,nugget_free" \
            --cluster-neighbor-blocks 2 \
            --cluster-block-shape 4x4 \
            --mean-design lat \
            --data-root "/home/jl2815/tco/data" \
            --output-root "${OUTROOT}" \
            --expanded-bounds \
            --lat-range "-3,7" \
            --lon-range "111,131" \
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

echo "Current date and time: $(date)"
```

Then submit:

```bash
sbatch slurm_real_july_spline_smooth_spectral_052126.sh
```

The sbatch file launches one Slurm job. Inside that one job, it loops through:

```text
6 smooth values x 4 years = 24 smooth/year runs
```

Resource request:

```text
time: 48:00:00
cpu:  8
mem:  64G
gpu:  1
jobs visible in squeue: 1
```

The single log files are:

```text
/home/jl2815/tco/exercise_output/logs/real_jul_spl_<jobid>.out
/home/jl2815/tco/exercise_output/logs/real_jul_spl_<jobid>.err
```

If the one job hits the 48-hour walltime, resubmit the same sbatch file. The
script passes `--skip-existing`, so completed daily outputs are skipped on the
next run.

## Output

Output root:

```text
/home/jl2815/tco/exercise_output/real_data/spline_smooth_spectral_052126
```

Folder structure:

```text
smooth_0p2/
  2022_07/
    daily_plots/
      20220701_data_vs_expected_periodogram.png
      ...
    daily_csv/
      20220701_fits.csv
      20220701_radial_spectrum.csv
      ...
    monthly_average/
      202207_30day_mean_data_vs_expected_periodogram.png
      202207_daily_mean_curves.csv
      202207_30day_mean_curves.csv
```

The same structure is repeated for every smooth/year combination.

## Transfer Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/real_data/spline_smooth_spectral_052126 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data/"
```

## Notes

The binning is locked to the first full July grid for each year. Every hourly
dataframe is reindexed onto that same `(lat, lon)` grid before thinning, so a
missing cell becomes `NaN` instead of shifting the x8/x4/x2/x1 grid positions.

The daily plot first averages over the 8 hours. The 30-day plot then averages
the 30 daily mean curves, so each day has equal weight.
