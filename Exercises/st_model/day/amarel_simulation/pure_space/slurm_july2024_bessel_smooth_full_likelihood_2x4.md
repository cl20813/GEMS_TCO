# July 2024 Bessel Smooth Full Likelihood 4x4 Tiles

This run fits the first 240 observed hours of July 2024.  Each hour is split
into `4x4` tiles and each tile is fitted independently with a direct-Bessel
anisotropic Matern model:

`sigmasq, range_lat, range_lon, smooth, nugget`

Both nugget settings are run in the same Slurm job:

- `free`: nugget estimated
- `fixed0`: nugget fixed at zero, with numerical jitter still added

When `--outlier-whitened-threshold` is positive, outlier screening is two-stage:
the first fit is a 4x4-cond2 cluster Vecchia fit using the same anisotropic
`phi1, phi2, phi3` reparameterization.  The QC whitening mask is computed once
per hour on the same `4x4` tile grid, then reused by the final `4x4` dense
full-likelihood tile fits.

Monthly summaries are written to:

```text
/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_4x4
```

## Transfer

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_full_likelihood_tiles_2x4.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/

ssh jl2815@amarel.rutgers.edu "cd /home/jl2815/tco && python -c 'import GEMS_TCO.matern_bessel_anisotropic; print(\"GEMS_TCO Bessel module import ok\")'"
```

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano run_july2024_bessel_full_likelihood_4x4.sh
sbatch run_july2024_bessel_full_likelihood_4x4.sh
```

The current RedHat high-memory partitions can be checked with:

```bash
sinfo -p mem-redhat,cmem-redhat -N --format="%N %P %t %c %m %f"
```

In the 2026-06-03 snapshot, the idle high-memory RedHat nodes were:

| Partition | Idle nodes | CPUs/node | Slurm memory | Features |
| --- | --- | ---: | ---: | --- |
| `mem-redhat` | `mem003-005` | 40 | 1,000,000 MB (~1 TB) | Piscataway, EDR, Cascade Lake |
| `mem-redhat` | `mem010` | 64 | 2,063,978 MB (~2.06 TB) | Piscataway, NDR, Sapphire Rapids |
| `mem-redhat` | `memk001-002` | 64 | 2,000,000 MB (~2 TB) | Piscataway, EDR, Ice Lake |
| `mem-redhat` | `memk003` | 64 | 2,000,000 MB (~2 TB) | Piscataway, EDR, Sapphire Rapids |
| `cmem-redhat` | `memc001` | 52 | 1,500,000 MB (~1.5 TB) | Camden, EDR, Ampere |

This job only requests 12 CPUs and 128 GB, so any of the idle `mem-redhat` nodes
can satisfy the request.  The `cmem-redhat` node is not included in the submit
script because its `ampere` feature may require a different software build than
the existing x86 conda/Python environment.

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=bess_full4x4
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/bess_full4x4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/bess_full4x4_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=mem-redhat

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

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_full_likelihood_tiles_2x4.py
DATA_PATH=/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
OUTDIR=/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/full_likelihood_4x4
MONTHLY_OUTDIR=/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_4x4
MANIFEST=${OUTDIR}/manifest_hours.csv

mkdir -p /home/jl2815/tco/exercise_output/logs "${OUTDIR}" "${MONTHLY_OUTDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
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
  --month 2024-07 \
  --max-hours 240 \
  --expected-hours 240 \
  --time-col hour \
  --x-col Longitude \
  --y-col Latitude \
  --value-col ColumnAmountO3 \
  --coords raw \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --tile-y 4 \
  --tile-x 4

for NUGGET_MODE in free fixed0; do
  echo "============================================================"
  echo "Full likelihood start: nugget_mode=${NUGGET_MODE}, time=$(date)"
  echo "Output: ${OUTDIR}/${NUGGET_MODE}"
  echo "============================================================"

  for HOUR_IDX in $(seq 0 239); do
    echo "---- full nugget=${NUGGET_MODE} hour_idx=${HOUR_IDX}/239 $(date) ----"
    srun --exclusive -N1 -n1 -c12 python "${SCRIPT}" \
      --mode fit \
      --input "${DATA_PATH}" \
      --output-dir "${OUTDIR}" \
      --monthly-output-dir "${MONTHLY_OUTDIR}" \
      --manifest "${MANIFEST}" \
      --month 2024-07 \
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
      --tile-y 4 \
      --tile-x 4 \
      --min-tile-points 200 \
      --tile-max-points 2400 \
      --tile-workers 4 \
      --qc-tile-y 4 \
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
    --month 2024-07 \
    --max-hours 240 \
    --expected-hours 240 \
    --nugget-mode "${NUGGET_MODE}"

  echo "Completed full likelihood nugget_mode=${NUGGET_MODE}: $(date)"
done

echo "Finished full likelihood run: $(date)"
```

Expected monthly output files:

```text
/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_4x4/
  202407_full_likelihood_free_tile_monthly_summary.csv
  202407_full_likelihood_fixed0_tile_monthly_summary.csv
  202407_full_likelihood_free_tile_monthly_parameter_maps.png
  202407_full_likelihood_fixed0_tile_monthly_parameter_maps.png
  202407_full_likelihood_free_tile_monthly_nugget_nu_maps.png
  202407_full_likelihood_fixed0_tile_monthly_nugget_nu_maps.png
  202407_full_likelihood_free_tile_monthly_{sigmasq,sigma,range_lat,range_lon,nu,nugget,phi1,phi2,phi3}_map.png
  202407_full_likelihood_fixed0_tile_monthly_{sigmasq,sigma,range_lat,range_lon,nu,nugget,phi1,phi2,phi3}_map.png
```

## Recover Partial Hourly Outputs

If the Slurm job stops before the summarize step, copy the already-written
hourly tile CSVs from Amarel to the local machine:

```bash
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/july2024_bessel_smooth/full_likelihood_2x4"


REMOTE_ROOT="/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/full_likelihood_2x4"




mkdir -p "${LOCAL_ROOT}/free" "${LOCAL_ROOT}/fixed0"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_ROOT}/manifest_hours.csv" \
  "${LOCAL_ROOT}/"

mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/july2024_bessel_smooth/full_likelihood_2x4/free"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/full_likelihood_2x4/free/hourly" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/july2024_bessel_smooth/full_likelihood_2x4/free/"

  
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_ROOT}/fixed0/hourly" \
  "${LOCAL_ROOT}/fixed0/"
```

Optional monthly output recovery, if the summarize step finished:

```bash
LOCAL_MONTHLY="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/july2024_bessel_smooth/monthly_output"
REMOTE_MONTHLY="/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output"

mkdir -p "${LOCAL_MONTHLY}"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_MONTHLY}/"* \
  "${LOCAL_MONTHLY}/"
```

If only one nugget mode had started, run only the corresponding `scp -r`
command for `free/hourly` or `fixed0/hourly`.
