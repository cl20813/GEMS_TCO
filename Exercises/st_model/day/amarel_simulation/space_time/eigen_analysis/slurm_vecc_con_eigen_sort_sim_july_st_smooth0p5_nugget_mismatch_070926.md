# One-Day Simulated July ST Vecchia Conditional Eigen Diagnostic, Smooth 0.5 Nugget 0

This run creates or reuses one simulated July day from a Matern ST DGP:

```text
DGP: Matern smooth=0.5, nugget=0
truth:
  sigmasq    = 10
  range_lat  = 0.2
  range_lon  = 0.3
  range_time = 2.0
  advec_lat  = 0.08
  advec_lon  = -0.2
data amount: one year, one July day, 8 hourly slots
diagnostic: Vecchia conditional target-block covariance eigenbasis
loss label: Vecchia objective per target observation, printed in plot legends
```

The Python driver checks the simulation truth JSON before generating anything.
If a valid smooth=0.5, nugget=0 pickle already exists, it reuses it.  If not,
it runs the reusable July ST circulant generator with `--max-hours 8`.

Two comparison folders are written under the same summer output root:

```text
smoothness_mismatch:
  Matern smooth=0.5, nugget fixed 0
  Matern smooth=0.3, nugget fixed 0
  Matern smooth=1.0, nugget fixed 0

nugget_mismatch:
  Matern smooth=0.5, nugget fixed 0
  Matern smooth=0.5, nugget fixed 2
```

The common engine is not modified.  The new driver patches model specs and a
fixed-nugget Matern wrapper at runtime, then calls
`vecchia_conditional_eigen_sort_common_engine_061926.py`.

Main outputs:

```text
/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926/experiment_manifest.json
/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926/smoothness_mismatch/*_summary.csv
/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926/smoothness_mismatch/daily_plots/year_2023/sim_2023_day01_vecchia_conditional_eigen_sort_comparison.png
/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926/nugget_mismatch/*_summary.csv
/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926/nugget_mismatch/daily_plots/year_2023/sim_2023_day01_vecchia_conditional_eigen_sort_comparison.png
```

## 1. Upload From Local Mac

Run from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis"
REMOTE_SIM_DIR="/home/jl2815/tco/simulate_data"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/eigen_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} ${REMOTE_SIM_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_ROOT}/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SIM_DIR}/"

scp \
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_common_engine_061926.py" \
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_sim_smooth0p5_nugget_mismatch_070926.py" \
  "${LOCAL_DIR}/slurm_vecc_con_eigen_sort_sim_july_st_smooth0p5_nugget_mismatch_070926.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

The one-day generator needs the July real-location template pickle for 2023. If
it is not already on Amarel, upload it once:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/data/pickle_2023"

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_23_07.pkl" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/"
```

## 2. Submit On Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis
nano run_vecc_con_eigen_sort_sim_s05_n0_mismatch_070926.sh
sbatch run_vecc_con_eigen_sort_sim_s05_n0_mismatch_070926.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=st_s05_mismatch
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/st_s05_mismatch_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/st_s05_mismatch_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

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

export PYTHONPATH="/home/jl2815/tco:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis/vecchia_conditional_eigen_sort_sim_smooth0p5_nugget_mismatch_070926.py"
GENERATOR="/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py"
DATA_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926"
OUTROOT="/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926"

mkdir -p "${OUTROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Script: ${SCRIPT}"
echo "Generator: ${GENERATOR}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTROOT}"
which python
nvidia-smi || true
python - <<'PY'
import numpy, pandas, scipy, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

python "${SCRIPT}" \
  --experiment both \
  --data-root "${DATA_ROOT}" \
  --generator-script "${GENERATOR}" \
  --generator-input-root "/home/jl2815/tco/data" \
  --generate-if-missing \
  --out-root "${OUTROOT}" \
  --years 2023 \
  --month 7 \
  --days 0 \
  --hours-per-day 8 \
  --sim-kind gridded \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --keep-exact-loc \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 32 \
  --diag-chunk-size 64 \
  --min-target-points 1 \
  --spline-n-points 4000 \
  --spline-r-max 30.0 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 8 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --grad-tol 1e-5 \
  --device cuda \
  --cuda-fallback error \
  --resample-grid 200 \
  --suppress-fit-prints

echo "Finished: $(date)"
```

Submit:

```bash
sbatch run_vecc_con_eigen_sort_sim_s05_n0_mismatch_070926.sh
```

Monitor:

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/summer/logs/st_s05_mismatch_<JOBID>.out
tail -f /home/jl2815/tco/exercise_output/summer/logs/st_s05_mismatch_<JOBID>.err
```

## 3. Pull Results To Local

Run from the local Mac:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_n0_vecchia_conditional_eigen_sort_smooth_nugget_mismatch_070926" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
