# Refit Implied-Spatial x1 Smooth 0.3, 2024 July Days 02-10

This is only for filling the missing `x1`, `nugget_free`, `smooth=0.3`
parameter rows used by the semivariogram notebook.

It does not recompute x8/x4/x2 and uses `--fit-only`, so it writes daily
`*_st_fits.csv` files only.

## Update Code

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco

ssh jl2815@amarel.rutgers.edu \
    'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_implied_spatial_spectral_052126.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_refit_implied_spatial_x1_smooth03_2024_july_days02_10_052926.sh
sbatch slurm_refit_implied_spatial_x1_smooth03_2024_july_days02_10_052926.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=refit_x1_s03_24
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/refit_x1_s03_24_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/refit_x1_s03_24_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_implied_spatial_spectral_052126.py"
OUTROOT="/home/jl2815/tco/exercise_output/real_data/implied_spatial_x1_refit_smooth0p3_2024_july_days02_10_052926"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"
echo "Refit only: 2024 July days 02-10, smooth=0.3, x1, nugget_free"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

srun python "${SCRIPT}" \
    --years "2024" \
    --month 7 \
    --days "2,10" \
    --smooths "0.3" \
    --resolutions "1" \
    --variants "nugget_free" \
    --data-root "/home/jl2815/tco/data" \
    --output-root "${OUTROOT}" \
    --expanded-bounds \
    --lat-range=-3,7 \
    --lon-range=121,131 \
    --device cuda \
    --cuda-fallback cpu \
    --block-shape "4,4" \
    --n-neighbor-blocks-t 6 \
    --lag1-local-blocks 4 \
    --lag1-shifted-blocks 1 \
    --lag2-local-blocks 3 \
    --lag2-shifted-blocks 1 \
    --daily-stride 2 \
    --lag1-lon-offset 0.126 \
    --target-chunk-size 32 \
    --lbfgs-steps 8 \
    --lbfgs-eval 20 \
    --lbfgs-history 10 \
    --fit-only

echo "Current date and time: $(date)"
```

## Transfer Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/real_data/implied_spatial_x1_refit_smooth0p3_2024_july_days02_10_052926 \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real_data/"
```

After this transfer, rerun the semivariogram notebook. It is already set to
look in this refit folder first for `smooth=0.3` x1 rows, then fall back to the
original `lag643` output folder.
