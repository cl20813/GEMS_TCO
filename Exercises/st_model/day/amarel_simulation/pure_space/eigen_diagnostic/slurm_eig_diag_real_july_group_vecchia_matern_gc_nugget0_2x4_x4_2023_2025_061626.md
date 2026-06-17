# Real July Pure-Space Eigen Diagnostic: Generalized Cauchy Beta Comparison, Nugget 0

This run compares pure-space group-Vecchia eigen diagnostics for July real data.
It uses no QC refit and fixes the nugget at 0 for every model.

Settings:

```text
years: 2023, 2024, 2025
models:
  2023: gc_a075_b1 vs gc_a075_b05
  2024: gc_a08_b1 vs gc_a08_b05
  2025: gc_a075_b1 vs gc_a075_b05
domain: lat -3..7, lon 111..131 input files
regions: 2x4 coordinate tiles + whole-domain sparse x4 stride
Vecchia: 4x4 target blocks, condition on 2 previous max-min cluster blocks
mean: lat, GLS-profiled in fitting
nugget: fixed 0
QC: off
loss label: raw negative Vecchia objective / valid observations, shown to 5 decimals
```

## Transfer

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic /home/jl2815/tco"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_matern_gc_nugget0_2x4_x4_061626.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/

ssh jl2815@amarel.rutgers.edu "cd /home/jl2815/tco && python -c 'import GEMS_TCO.kernels_space_aniso_cauchy_cluster_060326; print(\"GEMS_TCO Cauchy group Vecchia imports ok\")'"
```

Expected inputs:

```text
/home/jl2815/tco/data/pickle_2023/tco_grid_lat-3to7_lon111to131_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat-3to7_lon111to131_25_07.pkl
```

## Submit

On Amarel:

```bash
cd ./jobscript/tco/gp_exercise
nano eig_diag_real_july_gv_gc_beta_nugget0_2x4_x4_061626.sh
sbatch eig_diag_real_july_gv_gc_beta_nugget0_2x4_x4_061626.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=eig_gv_gcbeta
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/eig_gv_gcbeta_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/eig_gv_gcbeta_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_matern_gc_nugget0_2x4_x4_061626.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_gc_beta_nugget0_2x4_x4_061626

YEARS=(2023 2024 2025)
MONTH=7
DAY_RANGE="1,30"
HOURS="all"

mkdir -p /home/jl2815/tco/exercise_output/logs
mkdir -p "${BASE_ROOT}"

echo "Running on: $(hostname)"
nvidia-smi || true
echo "Current date and time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

for YEAR in "${YEARS[@]}"; do
  YY=$(printf "%02d" $((YEAR % 100)))
  DATA_PATH=${DATA_ROOT}/pickle_${YEAR}/tco_grid_lat-3to7_lon111to131_${YY}_07.pkl
  YEAR_OUT=${BASE_ROOT}/${YEAR}

  mkdir -p "${YEAR_OUT}"
  export MPLCONFIGDIR="${YEAR_OUT}/.mplconfig"
  mkdir -p "${MPLCONFIGDIR}"

  echo ""
  echo "============================================================"
  echo "Starting YEAR=${YEAR}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "YEAR_OUT=${YEAR_OUT}"
  echo "Current date and time: $(date)"
  echo "============================================================"

  srun python "${SCRIPT}" \
    --input "${DATA_PATH}" \
    --output-root "${YEAR_OUT}" \
    --year "${YEAR}" \
    --month "${MONTH}" \
    --days "${DAY_RANGE}" \
    --hours "${HOURS}" \
    --hour-match slot \
    --model-variants auto \
    --regions "tiles,sparse" \
    --tile-y 2 \
    --tile-x 4 \
    --sparse-strides "4" \
    --cluster-neighbor-blocks 2 \
    --cluster-block-shape 4x4 \
    --target-chunk-size 96 \
    --min-target-points 1 \
    --mean-design lat \
    --x-col auto \
    --y-col auto \
    --value-col ColumnAmountO3 \
    --coords raw \
    --device cuda \
    --eig-device same \
    --cuda-fallback error \
    --lbfgs-steps 8 \
    --lbfgs-eval 20 \
    --sigmasq-init 10 \
    --range-init 0.2 \
    --range-lat-init 0.2 \
    --range-lon-init 0.3 \
    --min-points 80 \
    --max-points 0 \
    --max-eig-points 0 \
    --cov-jitter 1e-8

done

echo "Current date and time: $(date)"
```
