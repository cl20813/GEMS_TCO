# Real July Corridor-Width Cluster Vecchia, 4x4 Lag643

Created 2026-05-25.

This runbook launches one Amarel job, no array.  It fits 2022--2025 July real
data with the selected real-data corridor-width cluster Vecchia model:

- module: `GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643`
- block shape: `4x4`
- lag pattern: `6/4/3`
- reference one-step `|advec_lon|`: `0.126`
- `t-1` corridor: `[0.5 delta, 1.5 delta]`
- `t-2` corridor: `[0, 2 delta]`

Outputs are written on Amarel under:

```text
/home/jl2815/tco/exercise_output/estimates/day/real_july_corridor_width_4x4_lag643_052526
```

The main running files are:

```text
real_july_corridor_width_4x4_lag643_parameter_fits.csv
real_july_corridor_width_4x4_lag643_all_fits.json
real_july_corridor_width_4x4_lag643_all_fits.jsonl
running_summary.txt
run_config.json
```

The JSON/JSONL files keep the full per-fit telemetry.  The CSV is intentionally
clean: identifiers, status/error, loss, `time_s`, and estimated physical
parameters only.

## Transfer Files

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting /home/jl2815/tco/exercise_output/logs'

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting/fit_real_july_corridor_width_4x4_lag643_052526.py" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting/slurm_fit_real_july_corridor_width_4x4_lag643_052526.md" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting/
```

## Submit Slurm Job

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting
nano fit_real_july_corridor_width_4x4_lag643_052526.sh
sbatch fit_real_july_corridor_width_4x4_lag643_052526.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=real_corr643
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_corr643_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_corr643_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu023

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data/st_model_fiitting/fit_real_july_corridor_width_4x4_lag643_052526.py"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day/real_july_corridor_width_4x4_lag643_052526"

mkdir -p "${OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

srun python "${SCRIPT}" \
    --years 2022 2023 2024 2025 \
    --month 7 \
    --days 0,31 \
    --space 1,1 \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --smooth 0.5 \
    --reference-advec-lon-abs 0.126 \
    --daily-stride 2 \
    --target-chunk-size 128 \
    --min-target-points 1 \
    --lbfgs-lr 1.0 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-history 10 \
    --grad-tol 1e-5 \
    --keep-exact-loc \
    --cuda-fallback error \
    --skip-existing \
    --output-root "${OUTDIR}"

echo "Current date and time: $(date)"
```

## Pull Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/real_july_corridor_width_4x4_lag643_052526 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```
