# Cluster Strategy Sweep On Pre-Generated July ST Simulation Assets

Created 2026-05-22.

This runbook transfers the cluster strategy wrapper and the fitting script to
Amarel, then launches one visible Slurm job.  It does not generate simulation
data.  It fits 200 simulation iterations across:

- strategies: `center_full`, `center_tapered`, `offset_full`, `offset_tapered`, `offset_tapered_force_center`
- cluster shapes: `3x3`, `4x4`
- lag depth fixed through `t-2`
- offset convention: lag 1 uses `0.063*2=0.126`, lag 2 uses `0.063*4=0.252`

The fitting script writes one compact row per fit to:

```text
/home/jl2815/tco/exercise_output/estimates/day/cluster_strategy_sweep_052226/all_fits_summary.csv
```

Truth parameters are written once to `truth_params.json`, not repeated in every
fit row.  Running summaries are refreshed during the job.

## Transfer Files

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time /home/jl2815/tco/exercise_output/logs'

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_sim_july_cluster_strategy_sweep_052226.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/
```

## Create And Submit Slurm Script

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano fit_sim_july_cluster_strategy_sweep_052226.sh
sbatch fit_sim_july_cluster_strategy_sweep_052226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=clust_st_sweep
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/clust_st_sweep_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/clust_st_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu043

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_sim_july_cluster_strategy_sweep_052226.py"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day/cluster_strategy_sweep_052226"
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
    --strategies center_full,center_tapered,offset_full,offset_tapered,offset_tapered_force_center \
    --block-shapes 3x3,4x4 \
    --lag0-block-count 6 \
    --lag1-keep-fraction 0.80 \
    --lag2-keep-fraction 0.50 \
    --lag1-lon-offset 0.126 \
    --lag2-lon-offset 0.252 \
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

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/cluster_strategy_sweep_052226 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/"
```
