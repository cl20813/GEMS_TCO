# Smooth 0.3 Nugget 0 Simulation, Pure-Space Vecchia Bessel Smooth Estimation, 2x4 Tiles

This test uses the reusable Amarel simulation data generated with true Matérn
smoothness `nu=0.3` and `nugget=0`.  It fits pure-space Vecchia models for the
first 10 days:

```text
10 days x 8 hours/day = 80 hourly pure-space fits
```

Each hour is split into `2x4` spatial tiles over:

```text
lat: -3 to 2
lon: 121 to 131
```

Within each tile we estimate:

```text
sigmasq, range_lat, range_lon, smooth, nugget
```

Estimator:

- `vecchia_cluster_4x4_cond2_tiles_2x4`: cluster Vecchia, 4x4 target blocks, 2 previous conditioning blocks

Simulation data root on Amarel:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0
```

Output root on Amarel:

```text
/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_nugget0_purespace_bessel_vecchia_2x4_fixed0_060726
```

For `year=2024`, the wrapper expects:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2024_july_st_circulant/sim_july2024_st_circulant_real_locations.pkl
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0/2024_july_st_circulant/sim_july2024_st_circulant_truth.json
```

Because `real_locations` values are simulated at
`Source_Latitude/Source_Longitude`, the wrapper fits using
`Source_Longitude` and `Source_Latitude` as the spatial coordinates.

## 1. Upload From Local Mac

Run this block from the local Mac, not from Amarel:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space"
REMOTE_SIM_DIR="${REMOTE_DIR}/simulation"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_PURE="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/pure_space"
LOCAL_SIM="${LOCAL_PURE}/simulation"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} ${REMOTE_SIM_DIR} /home/jl2815/tco"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp "${LOCAL_PURE}/fit_july2024_bessel_smooth_vecchia_cluster_4x4_cond2_tiles_2x4.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"

scp \
  "${LOCAL_SIM}/fit_sim_smooth0p3_nugget0_bessel_smooth_vecchia_tiles_2x4.py" \
  "${LOCAL_SIM}/slurm_sim_smooth0p3_nugget0_bessel_smooth_vecchia_tiles_2x4_060726.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SIM_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/simulation
nano run_sim_smooth0p3_nugget0_bessel_smooth_vecchia_tiles_2x4_060726.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=sim03n0_v2x4
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/sim03n0_v2x4_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/sim03n0_v2x4_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=mem-redhat

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840 || true

if ! command -v conda >/dev/null 2>&1; then
  source "${HOME}/.bashrc" || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found after module load and ~/.bashrc fallback." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/simulation/fit_sim_smooth0p3_nugget0_bessel_smooth_vecchia_tiles_2x4.py"
SIM_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0"
OUTROOT="/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_nugget0_purespace_bessel_vecchia_2x4_fixed0_060726"
YEAR="${YEAR:-2024}"
MAX_HOURS="${MAX_HOURS:-80}"
NUGGET_MODE="${NUGGET_MODE:-fixed0}"

mkdir -p /home/jl2815/tco/exercise_output/summer/logs "${OUTROOT}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "YEAR=${YEAR}"
echo "MAX_HOURS=${MAX_HOURS}"
echo "NUGGET_MODE=${NUGGET_MODE}"
echo "SIM_ROOT=${SIM_ROOT}"
echo "OUTROOT=${OUTROOT}"
which python
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY

python "${SCRIPT}" \
  --mode all \
  --year "${YEAR}" \
  --sim-root "${SIM_ROOT}" \
  --sim-kind real_locations \
  --out-root "${OUTROOT}" \
  --max-hours "${MAX_HOURS}" \
  --expected-hours "${MAX_HOURS}" \
  --hour-start 0 \
  --hour-end "${MAX_HOURS}" \
  --nugget-mode "${NUGGET_MODE}" \
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
  --outlier-whitened-threshold 0

echo "Finished: $(date)"
```

Submit:

```bash
sbatch run_sim_smooth0p3_nugget0_bessel_smooth_vecchia_tiles_2x4_060726.sh
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_nugget0_purespace_bessel_vecchia_2x4_fixed0_060726" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```

Expected summary files:

```text
.../monthly_output/vecchia_cluster_4x4_cond2_tiles_2x4/
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_summary.csv
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_parameter_maps.png
  202407_vecc_cluster_4x4_cond2_fixed0_tile_monthly_nugget_nu_maps.png
```
