# Real-data Eigenvalue Diagnostics: July 2024, Pure Space, nu=0.3 / 0.5 / 1.0

This runbook launches the real-data eigenvalue diagnostic:

```text
Exercises/st_model/day/real_data/eda/eig_diag_real_july_pure_space.py
```

It uses the actual July 2024 GEMS `tco_grid` pickle, not the simulated
circulant-embedding asset.

Default first pass:

1. July 1, first observed hour only.
2. 4x4 geographic tiles.
3. Whole-domain sparse x8 and x4.
4. `nu=0.3`, `nu=0.5`, and `nu=1.0`, `nugget0`, pure-space isotropic covariance.

---

## Update package code

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

## Transfer run file

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/real_data/eda'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/eda/eig_diag_real_july_pure_space.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/real_data/eda/
```

## Real data location

Expected Amarel path:

```bash
/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
```

If it is missing, transfer the expanded-bound July file first:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2024 --months 7 --only-extra-bounds
```

---

## Submit on Amarel

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano eig_diag_real24_purespace_nu03_05_051426.sh
sbatch eig_diag_real24_purespace_nu03_05_051426.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=real24_eig05
#SBATCH --output=/home/jl2815/tco/exercise_output/real24_eig05_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/real24_eig05_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu018

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi
echo "Current date and time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEAR=2024
MONTH=7
DAY_RANGE="1,1"
HOURS="first"
SMOOTHS=(0.3 0.5 1.0)
VARIANTS="nugget0"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/real_data/eda/eig_diag_real_july_pure_space.py"
DATA_PATH="/home/jl2815/tco/data/pickle_${YEAR}/tco_grid_lat-3to7_lon111to131_24_07.pkl"
OUTROOT="/home/jl2815/tco/exercise_output/eda/real/eigdiag_${YEAR}_july"

mkdir -p "${OUTROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

for SMOOTH in "${SMOOTHS[@]}"; do
    echo ""
    echo "Running REAL eigen diagnostics: days=${DAY_RANGE}, hours=${HOURS}, smooth=${SMOOTH}, variants=${VARIANTS}"

    srun python "${SCRIPT}" \
        --input "${DATA_PATH}" \
        --output-root "${OUTROOT}" \
        --year "${YEAR}" \
        --month "${MONTH}" \
        --days "${DAY_RANGE}" \
        --hours "${HOURS}" \
        --hour-match slot \
        --smooth "${SMOOTH}" \
        --regions "tiles4x4,sparse" \
        --tile-y 4 \
        --tile-x 4 \
        --sparse-strides "8,4" \
        --variants "${VARIANTS}" \
        --neighbors 8 \
        --mean-design base \
        --x-col Source_Longitude \
        --y-col Source_Latitude \
        --value-col ColumnAmountO3 \
        --coords raw \
        --device cuda \
        --eig-device same \
        --target-chunk-size 1024 \
        --lbfgs-steps 8 \
        --lbfgs-eval 20 \
        --min-points 80 \
        --cov-jitter 1e-8
done

echo "Current date and time: $(date)"
```

To run all observed slots for July 1:

```bash
HOURS="all"
DAY_RANGE="1,1"
```

To run the first observed slot for every July day:

```bash
HOURS="first"
DAY_RANGE="1,31"
```

For larger runs such as all July days or `VARIANTS="nugget0,nugget_free"`,
raise the resources, for example:

```bash
#SBATCH --time=24:00:00
#SBATCH --mem=96G
```

To run both fixed-nugget and free-nugget models:

```bash
VARIANTS="nugget0,nugget_free"
```

---

## Output

```text
/home/jl2815/tco/exercise_output/eda/real/eigdiag_2024_july/nu0p3/
  real_eigen_diagnostic_math_notes.txt
  real_eigen_fit_rows.csv
  real_eigen_diagnostic_summary.csv
  20240701_h0053/
    nugget0_x8_eigdiag.png
    nugget0_x4_eigdiag.png
    nugget0_tiles4x4_overview.png
/home/jl2815/tco/exercise_output/eda/real/eigdiag_2024_july/nu0p5/
  real_eigen_diagnostic_math_notes.txt
  real_eigen_fit_rows.csv
  real_eigen_diagnostic_summary.csv
  20240701_h0053/
    nugget0_tile_r1c1_of_4x4_eigdiag.png
    ...
    nugget0_tiles4x4_overview.png
    nugget0_x8_eigdiag.png
    nugget0_x4_eigdiag.png
```

The plot title starts with `real`, so it is easy to distinguish from the
simulation output.

## Transfer results back to local

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real/eigdiag_2024_july"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/real/eigdiag_2024_july \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real/"
```
