# Smooth 0.3 Simulation, Pure-Space Bessel Smooth Estimation, 4x4 Tiles

This test uses the reusable Amarel simulation data generated with true Matérn
smoothness `nu=0.3` and fits pure-space models for the first 10 days:

```text
10 days x 8 hours/day = 80 hourly pure-space fits
```

Each hour is split into `4x4` spatial tiles.  Within each tile we estimate:

```text
sigmasq, range_lat, range_lon, smooth, nugget
```

Two estimators are run into separate folders:

- `vecchia_cluster_4x4_cond2_tiles_4x4`: cluster Vecchia, 4x4 target blocks, 2 previous conditioning blocks
- `full_likelihood_4x4`: dense full likelihood with torch Bessel smooth optimization

The submit script requests one GPU.  This is mostly for the dense torch full
likelihood; the Vecchia/SciPy path is still largely CPU-bound.  Full likelihood
uses one tile worker to avoid several Python processes competing for the same
GPU, while Vecchia keeps four tile workers.

Simulation output root on Amarel:

```text
/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_purespace_bessel_4x4_060626
```

The assumed simulation root is:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3
```

For `year=2024`, the wrapper expects:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2024_july_st_circulant/sim_july2024_st_circulant_real_locations.pkl
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2024_july_st_circulant/sim_july2024_st_circulant_truth.json
```

If your Amarel smooth0.3 simulation folder has a different name, edit `SIM_ROOT`
in the Slurm script below.

Because `real_locations` values are simulated at
`Source_Latitude/Source_Longitude`, the wrapper automatically fits using
`Source_Longitude` and `Source_Latitude` as the spatial coordinates.  If
`--sim-kind gridded` is used, it switches back to `Longitude` and `Latitude`.

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

scp \
  "${LOCAL_PURE}/fit_real_july2022_2025_bessel_smooth_full_likelihood_tiles_2x4.py" \
  "${LOCAL_PURE}/fit_real_july2022_2025_bessel_smooth_vecchia_tiles_2x4.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"

scp \
  "${LOCAL_SIM}/fit_sim_smooth0p3_bessel_smooth_tiles_4x4.py" \
  "${LOCAL_SIM}/slurm_sim_smooth0p3_bessel_smooth_tiles_4x4_060626.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SIM_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/simulation
nano run_sim_smooth0p3_bessel_smooth_tiles_4x4_060626.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=sim03_pure_bess
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/sim03_pure_bess_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/sim03_pure_bess_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

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

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_FULL_DEVICE=auto
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/simulation/fit_sim_smooth0p3_bessel_smooth_tiles_4x4.py"
SIM_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3"
OUTROOT="/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_purespace_bessel_4x4_060626"
YEAR="${YEAR:-2024}"
MAX_HOURS="${MAX_HOURS:-80}"
NUGGET_MODE="${NUGGET_MODE:-free}"

METHODS=(vecchia full)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p /home/jl2815/tco/exercise_output/summer/logs "${OUTROOT}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "METHOD=${METHOD}"
echo "YEAR=${YEAR}"
echo "MAX_HOURS=${MAX_HOURS}"
echo "SIM_ROOT=${SIM_ROOT}"
echo "OUTROOT=${OUTROOT}"
which python
nvidia-smi || true
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

python "${SCRIPT}" \
  --mode all \
  --method "${METHOD}" \
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
  --tile-y 4 \
  --tile-x 4 \
  --min-tile-points 200 \
  --tile-workers 4 \
  --full-tile-workers 1 \
  --vecchia-tile-workers 4 \
  --full-tile-max-points 2400 \
  --vecchia-tile-max-points 0 \
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

echo "Finished METHOD=${METHOD}: $(date)"
```

Submit:

```bash
sbatch run_sim_smooth0p3_bessel_smooth_tiles_4x4_060626.sh
```

This creates two array tasks:

```text
task 0: vecchia
task 1: full likelihood
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/sim_smooth0p3_purespace_bessel_4x4_060626" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```

Expected summary files:

```text
.../monthly_output/vecchia_cluster_4x4_cond2_tiles_4x4/
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_summary.csv
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_parameter_maps.png
  202407_vecc_cluster_4x4_cond2_free_tile_monthly_nugget_nu_maps.png

.../monthly_output/full_likelihood_4x4/
  202407_full_likelihood_free_tile_monthly_summary.csv
  202407_full_likelihood_free_tile_monthly_parameter_maps.png
  202407_full_likelihood_free_tile_monthly_nugget_nu_maps.png
```
