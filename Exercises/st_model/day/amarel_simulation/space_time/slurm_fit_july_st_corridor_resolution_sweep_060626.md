# July ST Corridor Block Max-Min Sweep, 2026-06-06

This runbook fits the space-time corridor Vecchia model while increasing the
number of max-min ordered 4x4 target blocks:

```text
first 200 -> 400 -> 600 -> 1000 block centers
```

The script builds regular 4x4 blocks on the spatial grid, orders their centers
by max-min, and keeps all grid cells inside the first K blocks.  That gives the
wide-coverage-to-dense progression you wanted while keeping the plot x-axis
categorical and stable.

Experiments:

- real data, year-specific refinement:
  - 2022: fit smooth `0.2`, `0.25`, `0.3`, `0.35`
  - 2023: fit smooth `0.3`, `0.35`
  - 2024: fit smooth `0.3`, `0.35`
  - 2025: fit smooth `0.3`, `0.35`
- region: `lat [-3, 2]`, `lon [121, 131]`
- model: corridor Vecchia cluster, `4x4`, lag `6/4/3`
- output root on Amarel: `/home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626`

The default Slurm block below uses `DAYS=0,15` to match the earlier first-15-day
monthly-average tests.  Set `DAYS=0,30` before `sbatch` if you want the full
July month.

## 1. Upload Scripts To Amarel

Run this from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
LOCAL_DIR="/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR}"

scp \
  "${LOCAL_DIR}/fit_july2024_st_corridor_density_sweep_060426.py" \
  "${LOCAL_DIR}/fit_july_st_corridor_resolution_sweep_060626.py" \
  "${LOCAL_DIR}/slurm_fit_july_st_corridor_resolution_sweep_060626.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## 2. Submit On Amarel

On Amarel:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_july_st_corridor_resolution_sweep_060626.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=st_res643
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/slurm_logs/st_res643_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/slurm_logs/st_res643_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi
echo "Current date and time: $(date)"

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_july_st_corridor_resolution_sweep_060626.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626"
MONTHLY_ROOT="${OUTROOT}/monthly_average_plots_top"

mkdir -p "${OUTROOT}" "${MONTHLY_ROOT}" /home/jl2815/tco/exercise_output/summer/slurm_logs
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

DAYS="${DAYS:-0,15}"
BLOCK_PREFIXES="${BLOCK_PREFIXES:-200 400 600 1000}"

YEARS=(2022 2023 2024 2025)
DATASET="real"

YEAR_INDEX="${SLURM_ARRAY_TASK_ID}"
YEAR="${YEARS[$YEAR_INDEX]}"

case "${YEAR}" in
  2022)
    REAL_FIT_SMOOTHS=(0.2 0.25 0.3 0.35)
    ;;
  2023|2024|2025)
    REAL_FIT_SMOOTHS=(0.3 0.35)
    ;;
  *)
    echo "ERROR: unsupported YEAR=${YEAR}" >&2
    exit 3
    ;;
esac

REAL_FIT_SMOOTH_MAP="2022:0.2,0.25,0.3,0.35;2023:0.3,0.35;2024:0.3,0.35;2025:0.3,0.35"

OUTDIR="${OUTROOT}/${DATASET}/year_${YEAR}"
MONTHLY_OUTDIR="${MONTHLY_ROOT}/${DATASET}/year_${YEAR}"
mkdir -p "${OUTDIR}" "${MONTHLY_OUTDIR}"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "DATASET=${DATASET}"
echo "YEAR=${YEAR}"
echo "REAL_FIT_SMOOTHS=${REAL_FIT_SMOOTHS[*]}"
echo "DAYS=${DAYS}"
echo "BLOCK_PREFIXES=${BLOCK_PREFIXES}"
echo "OUTDIR=${OUTDIR}"
echo "MONTHLY_OUTDIR=${MONTHLY_OUTDIR}"

srun python "${SCRIPT}" \
  --datasets "${DATASET}" \
  --real-years "${YEAR}" \
  --real-fit-smooths "${REAL_FIT_SMOOTHS[@]}" \
  --real-fit-smooths-by-year "${REAL_FIT_SMOOTH_MAP}" \
  --block-prefixes ${BLOCK_PREFIXES} \
  --days "${DAYS}" \
  --month 7 \
  --space 1,1 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --spline-n-points 4000 \
  --spline-r-max 30.0 \
  --real-reference-advec-lon-abs 0.126 \
  --daily-stride 2 \
  --target-chunk-size 128 \
  --min-target-points 1 \
  --lbfgs-lr 1.0 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --grad-tol 1e-5 \
  --keep-exact-loc \
  --center-response \
  --require-cuda \
  --cuda-fallback error \
  --skip-existing \
  --summary-every 1 \
  --suppress-fit-prints \
  --out-dir "${OUTDIR}" \
  --monthly-out-dir "${MONTHLY_OUTDIR}"

echo "Current date and time: $(date)"
```

Submit:

```bash
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_july_st_corridor_resolution_sweep_060626.sh
```

Full July option:

```bash
DAYS=0,30 sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_july_st_corridor_resolution_sweep_060626.sh
```

## 3. Pull Monthly Plots Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_blockmaxmin_refine_smooth2022_060626"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626/monthly_average_plots_top" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_blockmaxmin_refine_smooth2022_060626/"
```

If you also want the CSVs/log summaries:

```bash
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```

If a monthly plot looks truncated while the job is still running, check:

```bash
cat /home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626/real/year_2025/running_summary.txt

column -s, -t < /home/jl2815/tco/exercise_output/summer/st_corridor_blockmaxmin_refine_smooth2022_060626/real/year_2025/st_corridor_blockmaxmin_missing_prefixes.csv | less -S
```

The plot only shows block prefixes completed for every smooth/year series in
that panel. Missing prefixes are written to `st_corridor_blockmaxmin_missing_prefixes.csv`.
