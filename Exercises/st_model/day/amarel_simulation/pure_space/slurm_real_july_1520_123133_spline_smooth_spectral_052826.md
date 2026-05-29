# Real July 1520_123133 Spline-Smooth Spectral Diagnostics

This run fits real July GEMS data for the `1520_123133` open-water region:

```text
lat    = 15..20
lon    = 123..133
smooth = 0.2, 0.3, 0.5, 1.0
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

## Region Data

Expected Amarel files:

```text
/home/jl2815/tco/data/pickle_2022/tco_grid_lat15to20_lon123to133_22_07.pkl
/home/jl2815/tco/data/pickle_2023/tco_grid_lat15to20_lon123to133_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat15to20_lon123to133_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat15to20_lon123to133_25_07.pkl
```

The Python script also accepts the already-prepared wider West Pacific files as
a fallback and filters them internally to `lon=123..133`:

```text
/home/jl2815/tco/data/pickle_2022/tco_grid_lat15to20_lon123to145_22_07.pkl
/home/jl2815/tco/data/pickle_2023/tco_grid_lat15to20_lon123to145_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat15to20_lon123to145_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat15to20_lon123to145_25_07.pkl
```

Verify on Amarel after uploading:

```bash
ssh jl2815@amarel.rutgers.edu '
for y in 2022 2023 2024 2025; do
  yy=${y:2:2}
  ls -lh /home/jl2815/tco/data/pickle_${y}/tco_grid_lat15to20_lon123to133_${yy}_07.pkl
done
'
```

If the exact `123133` files are not present yet, the wider `123145` files are
fine as long as they are already uploaded. The loaded dataframe is still bounded
by `--lat-range=15,20` and `--lon-range=123,133` before fitting.

## Transfer Code

Update the package code:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

Transfer the run script:

```bash
ssh jl2815@amarel.rutgers.edu \
    'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_1520_123133_spline_smooth_spectral_052826.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_real_july_1520_123133_spline_smooth_spectral_052826.sh
sbatch slurm_real_july_1520_123133_spline_smooth_spectral_052826.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=rj1520spl
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/rj1520spl_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/rj1520spl_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

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

SMOOTHS=(0.2 0.3 0.5 1.0)
YEARS=(2022 2023 2024 2025)
MONTH=7

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_1520_123133_spline_smooth_spectral_052826.py"
OUTROOT="/home/jl2815/tco/exercise_output/real_data/real_july_1520_123133_spline_smooth_spectral_052826"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"
echo "Single Slurm job; looping over all smooth/year combinations."
echo "YEARS=${YEARS[*]}"
echo "SMOOTHS=${SMOOTHS[*]}"
echo "REGION=lat15to20_lon123to133"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

for SMOOTH in "${SMOOTHS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        echo ""
        echo "============================================================"
        echo "Starting YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}, REGION=1520_123133"
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
            --lat-range=15,20 \
            --lon-range=123,133 \
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
sbatch slurm_real_july_1520_123133_spline_smooth_spectral_052826.sh
```

The sbatch file launches one Slurm job. Inside that one job, it loops through:

```text
4 smooth values x 4 years = 16 smooth/year runs
```

If the one job hits the 48-hour walltime, resubmit the same sbatch file. The
script passes `--skip-existing`, so completed daily outputs are skipped on the
next run.

## Output

Output root:

```text
/home/jl2815/tco/exercise_output/real_data/real_july_1520_123133_spline_smooth_spectral_052826
```

Monthly average plots are also copied into one top-level quicklook folder:

```text
/home/jl2815/tco/exercise_output/real_data/real_july_1520_123133_spline_smooth_spectral_052826/monthly_average_plots
```

Example quicklook filename:

```text
real_july_1520_123133_202207_smooth0p3_30day_mean_data_vs_expected_periodogram.png
```

Folder structure:

```text
monthly_average_plots/
  real_july_1520_123133_202207_smooth0p2_30day_mean_data_vs_expected_periodogram.png
  real_july_1520_123133_202207_smooth0p3_30day_mean_data_vs_expected_periodogram.png
  ...
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

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/real_data/real_july_1520_123133_spline_smooth_spectral_052826 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data/"
```
