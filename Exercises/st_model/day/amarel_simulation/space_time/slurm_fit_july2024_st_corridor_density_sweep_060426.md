# July 2022-2025 ST Corridor Density Sweep

Created 2026-06-04.

This Amarel runbook fits the fixed corridor-width 4x4 lag-643 space-time
Vecchia model while increasing the spatial data density by max-min order
prefixes.

The experiment is:

- data: real July 2022-2025, and smooth-0.3 simulation July 2022-2025
- fitted days: July days 1--15
- fit smooth fixed at `0.3` and `0.5`
- max-min spatial prefixes: `1000`, `2000`, `4000`, `18000`
- model geometry: corridor-width `4x4`, lag pattern `6/4/3`
- smooth kernel: ST Matérn spline, needed because the legacy closed-form
  corridor kernel fails for arbitrary smooth such as `0.3`

The total fit count is:

```text
4 years * 2 data kinds * 2 smooth values * 4 density prefixes * 15 days = 960 fits
```

The Slurm script below uses one plain task:

```text
task1: real/sim, smooth=0.3/0.5
```

The single job loops over years and writes fit outputs to:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_2022_2025_smooth03sim/fit_outputs/{YEAR}
```

Monthly/aggregate summaries and plots are written outside the fit-output tree:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_2022_2025_smooth03sim/monthly_average/{YEAR}
```

The task folder contains:

```text
st_corridor_density_sweep_all_fits.csv
st_corridor_density_sweep_all_fits.jsonl
parameter_by_nfirst_summary.csv
parameter_shift_vs_densest_summary.csv
parameter_by_nfirst.png
parameter_shift_vs_densest.png
running_summary.txt
run_config.json
```

## Transfer Files

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time /home/jl2815/tco/exercise_output/logs'

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_july2024_st_corridor_density_sweep_060426.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/
```

The simulation assets are expected to already exist at:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3
```

## Submit Slurm Job

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano fit_july2024_st_corridor_density_sweep_060426.sh
sbatch fit_july2024_st_corridor_density_sweep_060426.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=st_den643
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/st_den643_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/st_den643_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_july2024_st_corridor_density_sweep_060426.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_2022_2025_smooth03sim/fit_outputs"
MONTHLY_ROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_2022_2025_smooth03sim/monthly_average"
SIMROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3"

mkdir -p "${OUTROOT}" "${MONTHLY_ROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "DATA_KINDS=real sim"
echo "SMOOTHS=0.3 0.5"
echo "OUTROOT=${OUTROOT}"
echo "MONTHLY_ROOT=${MONTHLY_ROOT}"
echo "SIMROOT=${SIMROOT}"

for YEAR in 2022 2023 2024 2025; do
    OUTDIR="${OUTROOT}/${YEAR}"
    MONTHLY_OUTDIR="${MONTHLY_ROOT}/${YEAR}"
    mkdir -p "${OUTDIR}" "${MONTHLY_OUTDIR}"

    echo "============================================================"
    echo "YEAR=${YEAR}"
    echo "OUTDIR=${OUTDIR}"
    echo "MONTHLY_OUTDIR=${MONTHLY_OUTDIR}"
    echo "============================================================"

    srun python "${SCRIPT}" \
        --data-kinds real sim \
        --smooths 0.3 0.5 \
        --n-first-values 1000 2000 4000 18000 \
        --days 0,15 \
        --real-years "${YEAR}" \
        --sim-years "${YEAR}" \
        --month 7 \
        --space 1,1 \
        --lat-range=-3,2 \
        --lon-range=121,131 \
        --sim-data-root "${SIMROOT}" \
        --sim-pickle-kind real_locations \
        --smooth-kernel spline \
        --spline-n-points 4000 \
        --spline-r-max 30.0 \
        --real-reference-advec-lon-abs 0.126 \
        --sim-reference-advec-lon-abs 0.2 \
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
        --sim-init truth \
        --require-cuda \
        --cuda-fallback error \
        --skip-existing \
        --summary-every 1 \
        --suppress-fit-prints \
        --out-dir "${OUTDIR}" \
        --monthly-out-dir "${MONTHLY_OUTDIR}"
done

echo "Current date and time: $(date)"
```

## Pull Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_2022_2025_smooth03sim \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer/"
```
