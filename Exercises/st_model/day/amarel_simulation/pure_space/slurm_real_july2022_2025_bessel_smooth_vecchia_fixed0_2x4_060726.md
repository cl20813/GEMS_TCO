# Real July 2022-2025 Bessel Smooth Vecchia, Nugget Fixed 0, 2x4 Tiles

This run fits real July GEMS TCO data for 2022, 2023, 2024, and 2025 with the
pure-space anisotropic Matern Bessel model.

- final model: cluster Vecchia, target clusters `4x4`, conditioning on two previous cluster blocks
- spatial tiles: `2x4`
- region used after loading the expanded pickle: latitude `-3..2`, longitude `121..131`
- estimated parameters: `sigmasq, range_lat, range_lon, smooth`
- fixed parameter: `nugget = 0`
- smooth is estimated with bounded logit transform from `0.05` to `2.5`, initialized at `0.5`
- output root does not overlap the older nugget-free/free-vs-fixed run

The Python fitter is:

```text
/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_2x4.py
```

The file name says `july2024`, but the script is controlled by `--month` and
`--input`, so this runbook uses it for 2022-2025.

## Transfer

From the local machine:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space"
LOCAL_PURE="/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} /home/jl2815/tco"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_PURE}/fit_july2024_bessel_smooth_full_likelihood_tiles_2x4.py" \
  "${LOCAL_PURE}/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_2x4.py" \
  "${LOCAL_PURE}/slurm_real_july2022_2025_bessel_smooth_vecchia_fixed0_2x4_060726.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

The full-likelihood file is copied because the Vecchia script imports shared
data-loading, manifest, tiling, and monthly-plot helpers from it.

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano run_real_july2022_2025_bessel_vecchia_fixed0_2x4_060726.sh
sbatch run_real_july2022_2025_bessel_vecchia_fixed0_2x4_060726.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real_snu0_v2x4
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_snu0_v2x4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_snu0_v2x4_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=mem-redhat

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_2x4.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_july_bessel_smooth_vecchia_2x4_fixed0_2022_2025_060726"
MONTHLY_OUTDIR="${OUTROOT}/monthly_output"
NUGGET_MODE="fixed0"

mkdir -p /home/jl2815/tco/exercise_output/logs "${OUTROOT}" "${MONTHLY_OUTDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
which python
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
PY

for YEAR in 2022 2023 2024 2025; do
  YY=$(printf "%02d" $((YEAR % 100)))
  MONTH="${YEAR}-07"
  DATA_PATH="/home/jl2815/tco/data/pickle_${YEAR}/tco_grid_lat-3to7_lon111to131_${YY}_07.pkl"
  OUTDIR="${OUTROOT}/${YEAR}_07/vecchia_cluster_4x4_cond2_2x4"
  MANIFEST="${OUTDIR}/manifest_hours.csv"

  echo "============================================================"
  echo "Year=${YEAR}, nugget_mode=${NUGGET_MODE}, time=$(date)"
  echo "Data: ${DATA_PATH}"
  echo "Output: ${OUTDIR}/${NUGGET_MODE}"
  echo "Monthly output: ${MONTHLY_OUTDIR}"
  echo "============================================================"

  mkdir -p "${OUTDIR}"

  python "${SCRIPT}" \
    --mode manifest \
    --input "${DATA_PATH}" \
    --output-dir "${OUTDIR}" \
    --monthly-output-dir "${MONTHLY_OUTDIR}" \
    --manifest "${MANIFEST}" \
    --month "${MONTH}" \
    --max-hours 240 \
    --expected-hours 240 \
    --time-col hour \
    --x-col Longitude \
    --y-col Latitude \
    --value-col ColumnAmountO3 \
    --coords raw \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --tile-y 2 \
    --tile-x 4

  for HOUR_IDX in $(seq 0 239); do
    echo "---- year=${YEAR} fixed0 hour_idx=${HOUR_IDX}/239 $(date) ----"
    srun --exclusive -N1 -n1 -c8 python "${SCRIPT}" \
      --mode fit \
      --input "${DATA_PATH}" \
      --output-dir "${OUTDIR}" \
      --monthly-output-dir "${MONTHLY_OUTDIR}" \
      --manifest "${MANIFEST}" \
      --month "${MONTH}" \
      --max-hours 240 \
      --expected-hours 240 \
      --array-index "${HOUR_IDX}" \
      --time-col hour \
      --x-col Longitude \
      --y-col Latitude \
      --value-col ColumnAmountO3 \
      --coords raw \
      --lat-range=-3,2 \
      --lon-range=121,131 \
      --tile-y 2 \
      --tile-x 4 \
      --min-tile-points 200 \
      --tile-max-points 0 \
      --tile-workers 4 \
      --cluster-block-shape 4x4 \
      --cluster-neighbor-blocks 2 \
      --target-chunk-size 128 \
      --min-target-points 1 \
      --nugget-mode "${NUGGET_MODE}" \
      --mean-design lat \
      --range-lat-init 0.35 \
      --range-lon-init 0.35 \
      --smooth-init 0.5 \
      --smooth-min 0.05 \
      --smooth-max 2.5 \
      --range-min 0.03 \
      --range-max 5.0 \
      --jitter 1e-6 \
      --n-restarts 1 \
      --maxiter 80 \
      --maxfun 0 \
      --maxls 20 \
      --maxcor 20 \
      --optimizer-method L-BFGS-B \
      --outlier-whitened-threshold 10
  done

  python "${SCRIPT}" \
    --mode summarize \
    --input "${DATA_PATH}" \
    --output-dir "${OUTDIR}" \
    --monthly-output-dir "${MONTHLY_OUTDIR}" \
    --manifest "${MANIFEST}" \
    --month "${MONTH}" \
    --max-hours 240 \
    --expected-hours 240 \
    --nugget-mode "${NUGGET_MODE}"

  echo "Completed year=${YEAR}: $(date)"
done

echo "Finished all years: $(date)"
```

Expected monthly files:

```text
/home/jl2815/tco/exercise_output/summer/real_july_bessel_smooth_vecchia_2x4_fixed0_2022_2025_060726/monthly_output/
  202207_vecc_cluster_4x4_cond2_fixed0_tile_monthly_summary.csv
  202207_vecc_cluster_4x4_cond2_fixed0_tile_monthly_parameter_maps.png
  202207_vecc_cluster_4x4_cond2_fixed0_tile_monthly_nugget_nu_maps.png
  ...
  202507_vecc_cluster_4x4_cond2_fixed0_tile_monthly_summary.csv
  202507_vecc_cluster_4x4_cond2_fixed0_tile_monthly_parameter_maps.png
  202507_vecc_cluster_4x4_cond2_fixed0_tile_monthly_nugget_nu_maps.png
```

## Copy Monthly Output To Local

After the job finishes, from the local machine:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/real_july_bessel_smooth_vecchia_2x4_fixed0_2022_2025_060726"

scp -r \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_july_bessel_smooth_vecchia_2x4_fixed0_2022_2025_060726/monthly_output" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/real_july_bessel_smooth_vecchia_2x4_fixed0_2022_2025_060726/"
```
