# Simulation Eigenvalue Diagnostics: July 2024, Pure Space, nu=0.5

This runbook launches the eigenvalue-decomposition diagnostic in:

```text
Exercises/st_model/day/pure_space/simulation/eig_diag_sim_july_pure_space.py
```

The first-pass job is intentionally small: July 1, first observed hour only,
with both requested spatial reductions:

1. 4x4 geographic tiles: 16 separate covariance eigendecompositions.
2. Whole-domain sparse grids: x8 and x4 only.

The default fit is pure-space isotropic Matérn with `smooth=0.5`, `neighbors=8`,
and `variant=nugget0`. To also test nugget fitting, change `VARIANTS` to
`"nugget0,nugget_free"`.

---

## Update package code

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

## Transfer run file

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/pure_space/simulation/eig_diag_sim_july_pure_space.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation/
```

## Simulation asset location

Expected gridded simulation pickle:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2024_july_st_circulant/sim_july2024_st_circulant_gridded.pkl
```

If missing, first run:

```text
Exercises/st_model/simulate_data/slurm_generate_july_st_circulant_real_locations_2022_2025.md
```

---

## Submit on Amarel

```bash
ssh jl2815@amarel.rutgers.edu
cd ./jobscript/tco/gp_exercise
nano eig_diag_sim24_purespace_nu05_051426.sh
sbatch eig_diag_sim24_purespace_nu05_051426.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim24_eig05
#SBATCH --output=/home/jl2815/tco/exercise_output/sim24_eig05_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim24_eig05_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
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
SMOOTH="0.5"
VARIANTS="nugget0"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation/eig_diag_sim_july_pure_space.py"
DATA_PATH="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/${YEAR}_july_st_circulant/sim_july${YEAR}_st_circulant_gridded.pkl"
OUTROOT="/home/jl2815/tco/exercise_output/eda/simulation/eigdiag_${YEAR}_july"

mkdir -p "${OUTROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Running eigen diagnostics: days=${DAY_RANGE}, hours=${HOURS}, smooth=${SMOOTH}, variants=${VARIANTS}"

srun python "${SCRIPT}" \
    --input "${DATA_PATH}" \
    --output-root "${OUTROOT}" \
    --year "${YEAR}" \
    --month "${MONTH}" \
    --days "${DAY_RANGE}" \
    --hours "${HOURS}" \
    --smooth "${SMOOTH}" \
    --regions "tiles4x4,sparse" \
    --tile-y 4 \
    --tile-x 4 \
    --sparse-strides "8,4" \
    --variants "${VARIANTS}" \
    --neighbors 8 \
    --mean-design base \
    --x-col Longitude \
    --y-col Latitude \
    --value-col ColumnAmountO3 \
    --device cuda \
    --eig-device same \
    --target-chunk-size 1024 \
    --lbfgs-steps 8 \
    --lbfgs-eval 20 \
    --min-points 80 \
    --cov-jitter 1e-8 \
    --skip-existing

echo "Current date and time: $(date)"
```

To run more hours after the first pass:

```bash
DAY_RANGE="1,1"
HOURS="all"
```

To run all July days, expect a much larger job:

```bash
DAY_RANGE="1,31"
HOURS="first"
```

To fit both nugget-fixed and nugget-free covariance models:

```bash
VARIANTS="nugget0,nugget_free"
```

---

## Output

```text
/home/jl2815/tco/exercise_output/eda/simulation/eigdiag_2024_july/nu0p5/
  eigen_diagnostic_math_notes.txt
  eigen_fit_rows.csv
  eigen_diagnostic_summary.csv
  20240701_h0000/
    nugget0_tile_r1c1_of_4x4_eigdiag.png
    ...
    nugget0_tiles4x4_overview.png
    nugget0_x8_eigdiag.png
    nugget0_x4_eigdiag.png
```

Each PNG is the Figure 13.5-style cumulative-sum diagnostic:

| line/symbol | meaning |
|---|---|
| open black circles | cumulative sums of whitened eigenbasis scores, `Y_1^2 + ... + Y_k^2` |
| solid gray line | ideal 1-to-1 line |
| dashed gray lines | Brownian-bridge 95% approximate band, `k +/- 1.358 sqrt(2m)` |

## Transfer results back to local

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/simulation/eigdiag_2024_july \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/"
```
