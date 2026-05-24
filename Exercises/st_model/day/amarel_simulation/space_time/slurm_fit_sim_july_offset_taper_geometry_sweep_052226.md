# Offset-Taper Cluster Geometry Sweep

Created 2026-05-22.

This is the second-stage cluster Vecchia test.  The first sweep suggested:

- target-center forcing at `t-1/t-2` is not helpful
- `offset_tapered` is the most promising strategy

This run focuses only on offset-taper variants.  It tests:

- block geometry:
  - `3x3_default`
  - `4x4_default`
  - `4x4_lon_right_o1`
  - `4x4_lon_right_o2`
  - `3x5_lon_wide`
- lag block counts:
  - `t/t-1/t-2 = 6/5/3` baseline
  - `6/4/3`
  - `6/4/2`
  - `6/3/2`
- selected lag-2 center variants where `t-2` reuses the one-step `0.126`
  shifted center instead of the two-step `0.252` shifted center.

No target-center cluster is forced at `t-1/t-2`.

Outputs are written under:

```text
/home/jl2815/tco/exercise_output/estimates/day/offset_taper_geometry_sweep_052226
```

The main fit row file is:

```text
all_offset_taper_fits_summary.csv
```

## Transfer Files

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time /home/jl2815/tco/exercise_output/logs'

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/


scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/fit_sim_july_offset_taper_geometry_sweep_052226.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/
```

## Create And Submit Slurm Script

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano fit_sim_july_offset_taper_geometry_sweep_052226.sh
sbatch fit_sim_july_offset_taper_geometry_sweep_052226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=offtap_geom
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/offtap_geom_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/offtap_geom_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu035

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/fit_sim_july_offset_taper_geometry_sweep_052226.py"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day/offset_taper_geometry_sweep_052226"
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
    --lag1-lon-offset 0.126 \
    --lag2-lon-offset 0.252 \
    --daily-stride 2 \
    --target-chunk-size 128 \
    --min-target-points 1 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --summary-every 7 \
    --require-cuda \
    --resume \
    --out-dir "${OUTDIR}"

echo "Current date and time: $(date)"
```

## Pull Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/offset_taper_geometry_sweep_052226 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/"
```
