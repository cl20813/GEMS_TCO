# Real July 2022-2025 Bessel Smooth Full Likelihood, 2x4 Tiles

This is the canonical full-likelihood runbook for real July GEMS TCO data.

- years: 2022, 2023, 2024, 2025
- region: latitude `-3..2`, longitude `121..131`
- spatial tiles: `2x4`
- model: exact full likelihood, anisotropic Bessel Matern, smooth estimated
- default nugget mode: `fixed0`

The Python fitter is:

```text
/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_real_july2022_2025_bessel_smooth_full_likelihood_tiles_2x4.py
```

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
  "${LOCAL_PURE}/fit_real_july2022_2025_bessel_smooth_full_likelihood_tiles_2x4.py" \
  "${LOCAL_PURE}/slurm_real_july2022_2025_bessel_smooth_full_likelihood_tiles_2x4.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano run_real_july2022_2025_bessel_full_likelihood_tiles_2x4.sh
sbatch run_real_july2022_2025_bessel_full_likelihood_tiles_2x4.sh
```

To run the nugget-estimated version without creating another runbook:

```bash
sbatch --export=ALL,NUGGET_MODE=free run_real_july2022_2025_bessel_full_likelihood_tiles_2x4.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real_full2x4
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_full2x4_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_full2x4_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --partition=mem-redhat
#SBATCH --array=0-3

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_real_july2022_2025_bessel_smooth_full_likelihood_tiles_2x4.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_july2022_2025_bessel_smooth/full_likelihood_tiles_2x4"
MONTHLY_OUTDIR="${OUTROOT}/monthly_output"
NUGGET_MODE="${NUGGET_MODE:-fixed0}"
YEARS=(2022 2023 2024 2025)

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
YEAR="${YEARS[${ARRAY_INDEX}]}"
YY="$(printf "%02d" $((YEAR % 100)))"
MONTH="${YEAR}-07"
DATA_PATH="/home/jl2815/tco/data/pickle_${YEAR}/tco_grid_lat-3to7_lon111to131_${YY}_07.pkl"
OUTDIR="${OUTROOT}/${YEAR}_07"
MANIFEST="${OUTDIR}/manifest_hours.csv"

mkdir -p /home/jl2815/tco/exercise_output/logs "${OUTDIR}" "${MONTHLY_OUTDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Nugget mode: ${NUGGET_MODE}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTDIR}/${NUGGET_MODE}"
which python
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
PY

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

N_HOURS="$(python - "${MANIFEST}" <<'PY'
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1])))
PY
)"

if [[ "${N_HOURS}" -le 0 ]]; then
  echo "No manifest hours found for ${MONTH}"
  exit 1
fi

for HOUR_IDX in $(seq 0 $((N_HOURS - 1))); do
  echo "---- full year=${YEAR} nugget=${NUGGET_MODE} hour_idx=${HOUR_IDX}/$((N_HOURS - 1)) $(date) ----"
  srun --exclusive -N1 -n1 -c12 python "${SCRIPT}" \
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
    --tile-max-points 2400 \
    --tile-workers 4 \
    --qc-tile-y 2 \
    --qc-tile-x 4 \
    --qc-tile-max-points 0 \
    --qc-tile-workers 4 \
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

echo "Finished full likelihood year=${YEAR}, nugget=${NUGGET_MODE}: $(date)"
```

Expected monthly output:

```text
/home/jl2815/tco/exercise_output/summer/real_july2022_2025_bessel_smooth/full_likelihood_tiles_2x4/monthly_output/
  202207_full_likelihood_fixed0_tile_monthly_summary.csv
  ...
  202507_full_likelihood_fixed0_tile_monthly_summary.csv
```

## Copy Monthly Output To Local

After the array finishes, from the local machine:

```bash
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/real_july2022_2025_bessel_smooth/full_likelihood_tiles_2x4"
REMOTE_OUT="/home/jl2815/tco/exercise_output/summer/real_july2022_2025_bessel_smooth/full_likelihood_tiles_2x4"

mkdir -p "${LOCAL_OUT}"

scp -r \
  "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/monthly_output" \
  "${LOCAL_OUT}/"
```
