# Corridor Final Cluster Sweep

Created 2026-05-23.

This is the final focused Amarel comparison after the cluster offset-taper
sweeps.  It runs exactly five 4x4 candidates:

- corridor `6/4/3`: `t-1` covers `0.5x--1.5x |advec_lon|`, `t-2` covers `0x--2x |advec_lon|`
- corridor `6/5/3`: same corridor, one extra `t-1` block
- half-offset reuse `6/5/3`: `t-1=0.5x`, `t-2=0.5x`
- half-offset reuse `6/5/4`: `t-1=0.5x`, `t-2=0.5x`, one extra `t-2` block
- half-offset step2 `6/5/3`: `t-1=0.5x`, `t-2=1.0x`

Default reference magnitude is `|advec_lon| = 0.2`, so the actual defaults are:

- `t-1` corridor: `0.10--0.30`
- `t-2` corridor: `0.00--0.40`
- half offset: `0.10`
- full one-step offset for the added `t-2` variant: `0.20`

Only this Markdown runbook is needed locally; do not keep a duplicate local
`.sh` unless you specifically want a separate scratch copy.

Outputs are written under:

```text
/home/jl2815/tco/exercise_output/estimates/day/corridor_final_sweep_052326
```

The main fit row file is:

```text
all_corridor_final_fits_summary.csv
```

## Transfer Files

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time /home/jl2815/tco/exercise_output/logs'

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_sim_july_cluster_strategy_sweep_052226.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_sim_july_offset_taper_geometry_sweep_052226.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_sim_july_corridor_final_sweep_052326.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/
```

## Create And Submit Slurm Script

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano fit_sim_july_corridor_final_sweep_052326.sh
sbatch fit_sim_july_corridor_final_sweep_052326.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=corr_final
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/corr_final_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/corr_final_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu021

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_sim_july_corridor_final_sweep_052326.py"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day/corridor_final_sweep_052326"
SIMROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"

mkdir -p "${OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

srun python "${SCRIPT}" \
    --num-sims 200 \
    --years 2022,2023,2024,2025 \
    --day-idxs all \
    --asset-sampling cycle \
    --sim-data-root "${SIMROOT}" \
    --data-kind real_locations \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --reference-advec-lon-abs 0.2 \
    --lag1-corridor-low-mult 0.5 \
    --lag1-corridor-high-mult 1.5 \
    --lag2-corridor-low-mult 0.0 \
    --lag2-corridor-high-mult 2.0 \
    --daily-stride 2 \
    --target-chunk-size 128 \
    --min-target-points 1 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --summary-every 5 \
    --require-cuda \
    --resume \
    --out-dir "${OUTDIR}"

echo "Current date and time: $(date)"
```

## Pull Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/corridor_final_sweep_052326 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/"
```
