# One-Day Simulated July ST Vecchia Conditional Eigen Diagnostic, Advection Mismatch

This is the advection-mismatch companion to the smoothness/nugget mismatch
driver.  The data-generating process is still Matern smooth=0.5.  The fitted
models keep smoothness and nugget fixed, but force the advection parameters to
three different values:

```text
advection_mismatch_dgp1:
  true:  advec_lat=0.08, advec_lon=-0.2, nugget fixed 1
  zero:  advec_lat=0,    advec_lon=0,    nugget fixed 1
  large: advec_lat=0.5,  advec_lon=0.5,  nugget fixed 1
```

The default Python driver run is `advection_mismatch_dgp1`, matching the local
smooth0p5 simulated asset with nugget=1.

## Upload From Local Mac

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/eigen_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_common_engine_061926.py" \
  "${LOCAL_DIR}/vecchia_conditional_eigen_sort_sim_smooth0p5_advection_mismatch_071126.py" \
  "${LOCAL_DIR}/slurm_vecc_con_eigen_sort_sim_july_st_smooth0p5_advection_mismatch_071126.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit On Amarel

Create `run_vecc_con_eigen_sort_sim_s05_advec_mismatch_cpu_071126.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=st_s05_advec_cpu
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/st_s05_advec_cpu_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/st_s05_advec_cpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=mem-redhat
#SBATCH --nodelist=mem010

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

export PYTHONPATH="/home/jl2815/tco:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis/vecchia_conditional_eigen_sort_sim_smooth0p5_advection_mismatch_071126.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/sim_july_st_s05_vecchia_conditional_eigen_sort_advection_mismatch_071126"

mkdir -p "${OUTROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Script: ${SCRIPT}"
echo "Data root: auto-resolve validated smooth=0.5 nugget=1 simulation asset"
echo "Output root: ${OUTROOT}"
which python
python - <<'PY'
import numpy, pandas, scipy, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
print("torch threads", torch.get_num_threads())
PY

python "${SCRIPT}" \
  --experiment advection_mismatch_dgp1 \
  --isolate-models \
  --split-fit-diagnostic \
  --out-root "${OUTROOT}" \
  --years 2023 \
  --month 7 \
  --days 0 \
  --hours-per-day 8 \
  --truth-nugget 1.0 \
  --sim-kind gridded \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --keep-exact-loc \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 16 \
  --diag-chunk-size 64 \
  --min-target-points 1 \
  --spline-n-points 4000 \
  --spline-r-max 30.0 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --grad-tol 1e-5 \
  --device cpu \
  --cuda-fallback cpu \
  --resample-grid 200 \
  --save-daily-curves \
  --suppress-fit-prints

echo "Finished: $(date)"
```

Submit:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/eigen_analysis
sbatch run_vecc_con_eigen_sort_sim_s05_advec_mismatch_cpu_071126.sh
```

