# Real July Implied-Spatial Spectral Diagnostics

This run fits one daily spatio-temporal Vecchia model to the 8 hourly fields,
then compares hourly spatial residual periodograms with the finite-sample
expected periodogram from the fitted ST model's temporal-lag-zero implied
spatial covariance.

Plot layout:

```text
columns: x8, x4, x2, x1
row 1:  nugget fixed 0
row 2:  nugget free
```

Each day/resolution/variant uses one ST fit, not 8 separate pure-space fits.

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

If needed, transfer the expanded files from the Mac:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2022 2023 2024 2025 --months 7 --only-extra-bounds
```

## Transfer Code

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

ssh jl2815@amarel.rutgers.edu \
    'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_implied_spatial_spectral_052126.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_real_july_implied_spatial_spectral_052126.sh
sbatch slurm_real_july_implied_spatial_spectral_052126.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=real_jul_implsp
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_jul_implsp_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_jul_implsp_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu033

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

SMOOTHS=(0.2 0.25 0.3 0.35 0.4 0.45)
YEARS=(2022 2023 2024 2025)
MONTH=7

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_implied_spatial_spectral_052126.py"
OUTROOT="/home/jl2815/tco/exercise_output/real_data/implied_spatial_spectral_052126"
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
        echo "Starting implied-spatial ST run: YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
        echo "Current date and time: $(date)"
        echo "============================================================"

        srun python "${SCRIPT}" \
            --years "${YEAR}" \
            --month "${MONTH}" \
            --days "1,30" \
            --smooths "${SMOOTH}" \
            --resolutions "8,4,2,1" \
            --variants "nugget0,nugget_free" \
            --data-root "/home/jl2815/tco/data" \
            --output-root "${OUTROOT}" \
            --expanded-bounds \
            --lat-range=-3,7 \
            --lon-range=111,131 \
            --device cuda \
            --cuda-fallback cpu \
            --block-shape "3,3" \
            --n-neighbor-blocks-t 6 \
            --lag1-local-blocks 3 \
            --lag1-shifted-blocks 1 \
            --lag2-local-blocks 2 \
            --lag2-shifted-blocks 1 \
            --daily-stride 2 \
            --lag1-lon-offset 0.063 \
            --target-chunk-size 128 \
            --lbfgs-steps 8 \
            --lbfgs-eval 20 \
            --lbfgs-history 10 \
            --radial-bins 70 \
            --radial-qmax 0.985 \
            --skip-existing

        echo "Finished implied-spatial ST run: YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
        echo "Current date and time: $(date)"
    done
done

echo "Current date and time: $(date)"
```

Then submit:

```bash
sbatch slurm_real_july_implied_spatial_spectral_052126.sh
```

This is one visible Slurm job. If it hits walltime, submit the same sbatch file
again; `--skip-existing` skips completed daily outputs.

## Output

Output root:

```text
/home/jl2815/tco/exercise_output/real_data/implied_spatial_spectral_052126
```

Folder structure:

```text
smooth_0p2/
  2022_07/
    daily_plots/
      20220701_implied_spatial_data_vs_expected_periodogram.png
      ...
    daily_csv/
      20220701_st_fits.csv
      20220701_implied_spatial_radial_spectrum.csv
      ...
    monthly_average/
      202207_30day_mean_implied_spatial_data_vs_expected_periodogram.png
      202207_daily_mean_curves.csv
      202207_30day_mean_curves.csv
```

## Transfer Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/real_data/implied_spatial_spectral_052126 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data/"
```
