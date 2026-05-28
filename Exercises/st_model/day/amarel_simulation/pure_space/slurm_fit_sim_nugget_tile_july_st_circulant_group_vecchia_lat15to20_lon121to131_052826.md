# Pure-Space Group Vecchia Tile Nugget Fit For July Simulation Assets

This is the Philippine Sea version of the July pure-space tile-nugget run.  It
fits the July 2022-2025 Philippine Sea GEMS pkl files using pure-space
isotropic **cluster/group Vecchia**, not pointwise Vecchia.

Key settings:

- fitted data domain: focused region, latitude `[15, 20]`, longitude `[121, 131]`
- source pkl files: Philippine Sea step3 pkl files:
  `/home/jl2815/tco/data/pickle_{year}/tco_grid_lat15to20_lon121to131_YY_07.pkl`
- years: `2022`, `2023`, `2024`, `2025`
- smoothness values: `nu=0.2` and `nu=0.5`
- execution style: **one Slurm job, one GPU node, sequential loops**, no job array
- global fit: isotropic Matern with `log(phi1), log(phi2), log(nugget)`
  where `sigma^2 = phi1 / phi2` and `range = 1 / phi2`
- tile fit: hold global `sigma^2` and range fixed, profile only tile nugget
- mean design: `lat`, i.e. GLS-profiled `beta0 + beta1 * centered latitude`
- tile plots: longitude/latitude axes plus GEMS nadir marker at `(lon=128.2, lat=17.5)`

Using `mean_design=lat` is the better default here.  Each hour is fitted
separately, so time dummies are unnecessary, but the focused `[15, 20]`
latitude domain can carry a broad meridional mean gradient.  A GLS latitude
term removes that mean component before covariance parameters try to explain
it.  The code uses centered latitude, so it is mathematically the same as
`beta0 + beta1 * latitude` after reparameterizing the intercept.

The global-then-tile strategy is still a reasonable diagnostic: global
`sigma^2` and range are hard to identify inside small tiles, so fixing them
keeps the tile comparison focused on local nugget-like short-scale variance.
Treat tile nuggets as relative local diagnostics, not independent full
covariance fits.

## 1. Transfer Files

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_lat15to20_lon121to131_052826.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

The Philippine Sea July pkl files should already be on Amarel after running the
SCP commands in:

```text
/Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st/philippine_sea_step5_scp_commands.md
```

If you prefer the Python transfer helper instead, use:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2022 2023 2024 2025 --months 7 --only-extra-bounds --lat-lon-bounds 15,20,121,131
```

## 2. Single Sequential Slurm Job

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano fit_sim_july_group_vecchia_tiles_lat15to20_lon121to131_seq_052826.sh
sbatch fit_sim_july_group_vecchia_tiles_lat15to20_lon121to131_seq_052826.sh
```


#SBATCH --nodelist=gpu021

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=phsea_gv_seq
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/phsea_gv_seq_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/phsea_gv_seq_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1


set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEARS=(2022 2023 2024 2025)
SMOOTHS=(0.2 0.5)

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_lat15to20_lon121to131_052826.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/eda/real_data/july_lat15to20_lon121to131_group_vecchia_tiles_052826

expected_hours_for_year() {
  case "$1" in
    2022) echo 240 ;;
    2023) echo 248 ;;
    2024) echo 248 ;;
    2025) echo 247 ;;
    *) echo "Unsupported year: $1" >&2; return 1 ;;
  esac
}

mkdir -p /home/jl2815/tco/exercise_output/logs "${BASE_ROOT}"

echo "Host: $(hostname)"
echo "Started: $(date)"
nvidia-smi
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

for YEAR in "${YEARS[@]}"; do
  EXPECTED_HOURS=$(expected_hours_for_year "${YEAR}")
  YY=$(printf "%02d" $((YEAR % 100)))
  DATA_PATH=${DATA_ROOT}/pickle_${YEAR}/tco_grid_lat15to20_lon121to131_${YY}_07.pkl
  YEAR_OUT=${BASE_ROOT}/${YEAR}_july_lat15to20_lon121to131
  MANIFEST=${YEAR_OUT}/manifest_hours.csv

  mkdir -p "${YEAR_OUT}"

  echo "============================================================"
  echo "Year=${YEAR}, expected_hours=${EXPECTED_HOURS}"
  echo "Data=${DATA_PATH}"
  echo "Output=${YEAR_OUT}"
  echo "Time=$(date)"
  echo "============================================================"

  echo "Building manifest for ${YEAR}..."
  python "${SCRIPT}" \
    --mode manifest \
    --input "${DATA_PATH}" \
    --output-dir "${YEAR_OUT}" \
    --manifest "${MANIFEST}" \
    --month "${YEAR}-07" \
    --expected-hours "${EXPECTED_HOURS}" \
    --time-col hour \
    --x-col Longitude \
    --y-col Latitude \
    --value-col ColumnAmountO3 \
    --coords raw \
    --lat-range=15,20 \
    --lon-range=121,131 \
    --require-region-cover

  echo "Manifest row count for ${YEAR}:"
  wc -l "${MANIFEST}"

  for SMOOTH in "${SMOOTHS[@]}"; do
    SMOOTH_TAG=${SMOOTH/./p}
    OUTDIR=${YEAR_OUT}/nu${SMOOTH_TAG}
    mkdir -p "${OUTDIR}"
    export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
    mkdir -p "${MPLCONFIGDIR}"

    echo "============================================================"
    echo "Sequential fit start: year=${YEAR}, smooth=${SMOOTH}, output=${OUTDIR}"
    echo "Time: $(date)"
    echo "============================================================"

    for HOUR_IDX in $(seq 0 $((EXPECTED_HOURS - 1))); do
      echo "---- year=${YEAR} smooth=${SMOOTH} hour_idx=${HOUR_IDX}/${EXPECTED_HOURS} $(date) ----"
      srun python "${SCRIPT}" \
        --mode fit \
        --input "${DATA_PATH}" \
        --output-dir "${OUTDIR}" \
        --manifest "${MANIFEST}" \
        --month "${YEAR}-07" \
        --expected-hours "${EXPECTED_HOURS}" \
        --array-index "${HOUR_IDX}" \
        --time-col hour \
        --x-col Longitude \
        --y-col Latitude \
        --value-col ColumnAmountO3 \
        --coords raw \
        --lat-range=15,20 \
        --lon-range=121,131 \
        --require-region-cover \
        --smooth "${SMOOTH}" \
        --tile-y 4 \
        --tile-x 8 \
        --cluster-block-shape 4x4 \
        --cluster-neighbor-blocks 2 \
        --target-chunk-size 96 \
        --min-target-points 1 \
        --min-tile-points 80 \
        --mean-design lat \
        --sigmasq-init 10 \
        --range-init 0.2 \
        --nugget-init 1.0 \
        --nadir-lat 17.5 \
        --nadir-lon 128.2 \
        --device cuda
    done

    echo "Summarizing year=${YEAR}, smooth=${SMOOTH}: $(date)"
    python "${SCRIPT}" \
      --mode summarize \
      --input "${DATA_PATH}" \
      --output-dir "${OUTDIR}" \
      --manifest "${MANIFEST}" \
      --month "${YEAR}-07" \
      --expected-hours "${EXPECTED_HOURS}" \
      --time-col hour \
      --x-col Longitude \
      --y-col Latitude \
      --value-col ColumnAmountO3 \
      --coords raw \
      --lat-range=15,20 \
      --lon-range=121,131 \
      --require-region-cover \
      --smooth "${SMOOTH}" \
      --tile-y 4 \
      --tile-x 8 \
      --cluster-block-shape 4x4 \
      --cluster-neighbor-blocks 2 \
      --target-chunk-size 96 \
      --min-target-points 1 \
      --min-tile-points 80 \
      --mean-design lat \
      --nadir-lat 17.5 \
      --nadir-lon 128.2

    echo "Completed year=${YEAR}, smooth=${SMOOTH}: $(date)"
  done
done

echo "Finished all years and smooth values: $(date)"
```

Submit:

```bash
sbatch fit_sim_july_group_vecchia_tiles_lat15to20_lon121to131_seq_052826.sh
```

## 3. Monitor

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/logs/phsea_gv_seq_<JOBID>.out

ls -R /home/jl2815/tco/exercise_output/eda/real_data/july_lat15to20_lon121to131_group_vecchia_tiles_052826
```

Expected output folders:

```text
july_lat15to20_lon121to131_group_vecchia_tiles_052826/
  2022_july_lat15to20_lon121to131/
    manifest_hours.csv
    nu0p2/
      hourly/
      summary/
    nu0p5/
      hourly/
      summary/
  2023_july_lat15to20_lon121to131/
    ...
  2024_july_lat15to20_lon121to131/
    ...
  2025_july_lat15to20_lon121to131/
    ...
```

## 4. Copy Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/real_data/july_lat15to20_lon121to131_group_vecchia_tiles_052826 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data/"
```
