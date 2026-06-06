# July 2024 Bessel Smooth Vecchia Cluster 4x4 Cond2, 4x4 Tiles

This run fits the first 240 observed hours of July 2024.  Each hour is split
into `4x4` tiles.  Inside each tile, the Vecchia likelihood uses:

- target clusters: `4x4` grid cells
- conditioning: two previous max-min cluster blocks
- covariance: direct-Bessel anisotropic Matern
- estimated parameters: `sigmasq, range_lat, range_lon, smooth, nugget`
- optimizer scale: `phi1, phi2, phi3`, bounded smooth-logit, and nugget, with
  `phi2 = 1/range_lon`, `phi3 = (range_lon/range_lat)^2`, and
  `sigmasq = phi1/phi2`

Both nugget settings are run in the same Slurm job:

- `free`: nugget estimated
- `fixed0`: nugget fixed at zero, with numerical jitter still added

Monthly summaries are written to:

```text
/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_vecchia_4x4
```

## Transfer

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_full_likelihood_tiles_4x4.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_4x4.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/

ssh jl2815@amarel.rutgers.edu "cd /home/jl2815/tco && python -c 'import GEMS_TCO.matern_bessel_anisotropic; import GEMS_TCO.kernels_space_iso_cluster_052426; print(\"GEMS_TCO Bessel + Vecchia imports ok\")'"
```

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano run_july2024_bessel_vecchia_cluster_4x4_cond2_tiles_4x4.sh
sbatch run_july2024_bessel_vecchia_cluster_4x4_cond2_tiles_4x4.sh
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

This job only requests 8 CPUs and 64 GB, so any of the idle `mem-redhat` nodes
can satisfy the request.  The `cmem-redhat` node is not included in the submit
script because its `ampere` feature may require a different software build than
the existing x86 conda/Python environment.

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=bess_vecc4x4
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/bess_vecc4x4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/bess_vecc4x4_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_4x4.py
DATA_PATH=/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
OUTDIR=/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/vecchia_cluster_4x4_cond2_tiles_4x4
MONTHLY_OUTDIR=/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_vecchia_4x4
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
  echo "Vecchia start: nugget_mode=${NUGGET_MODE}, time=$(date)"
  echo "Output: ${OUTDIR}/${NUGGET_MODE}"
  echo "============================================================"

  for HOUR_IDX in $(seq 0 239); do
    echo "---- vecc nugget=${NUGGET_MODE} hour_idx=${HOUR_IDX}/239 $(date) ----"
    srun --exclusive -N1 -n1 -c8 python "${SCRIPT}" \
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
    --month 2024-07 \
    --max-hours 240 \
    --expected-hours 240 \
    --nugget-mode "${NUGGET_MODE}"

  echo "Completed Vecchia nugget_mode=${NUGGET_MODE}: $(date)"
done

echo "Finished Vecchia run: $(date)"
```

Expected monthly output files:

```text
/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/monthly_output_vecchia_4x4/
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_summary.csv
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_summary.csv
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_parameter_maps.png
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_parameter_maps.png
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_nugget_nu_maps.png
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_nugget_nu_maps.png
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_{sigmasq,sigma,range_lat,range_lon,nu,nugget,phi1,phi2,phi3}_map.png
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_{sigmasq,sigma,range_lat,range_lon,nu,nugget,phi1,phi2,phi3}_map.png
```
