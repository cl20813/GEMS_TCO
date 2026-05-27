### Update packages (mac -> Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac -> Amarel)
```
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_111131_052726.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/
```

### Transfer data (mac -> Amarel)
If the expanded July files are not already on Amarel:
```
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2022 2023 2024 2025 --months 7 --only-extra-bounds
```

Expected files:
```
/home/jl2815/tco/data/pickle_2022/tco_grid_lat-3to7_lon111to131_22_07.pkl
/home/jl2815/tco/data/pickle_2023/tco_grid_lat-3to7_lon111to131_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat-3to7_lon111to131_25_07.pkl
```

### Transfer estimate results (Amarel -> mac)
```
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/real_data/eigen_diag_july_group_vecchia_111131_052726 "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/real/"
```

---

### Connect & setup
```
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

### Check data on Amarel
```
for y in 2022 2023 2024 2025; do
  yy=${y:2:2}
  ls -lh /home/jl2815/tco/data/pickle_${y}/tco_grid_lat-3to7_lon111to131_${yy}_07.pkl
done
```

---

### Real July eigen diagnostic, group Vecchia (sbatch)

Settings:
```
years: 2022, 2023, 2024, 2025
smooth: 0.3, 0.5
data: expanded lat -3..7, lon 111..131
Vecchia: 4x4 block, condition on 2 previous blocks
plots: 4x4 tile monthly average, x4 sparse monthly average
mean: lat, GLS-profiled in fitting
eigen diagnostic: residual projection before R Sigma R eigendecomposition
```

```
cd ./jobscript/tco/gp_exercise
nano eig_diag_real_july_gv_111131_052726.sh
sbatch eig_diag_real_july_gv_111131_052726.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=real_eig_gv_july
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_eig_gv_july_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_eig_gv_july_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

#### Load Modules
module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_111131_052726.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/eda/real_data/eigen_diag_july_group_vecchia_111131_052726

YEARS=(2022 2023 2024 2025)
SMOOTHS=(0.3 0.5)
MONTH=7
DAY_RANGE="1,31"
HOURS="all"

mkdir -p /home/jl2815/tco/exercise_output/logs
mkdir -p "${BASE_ROOT}"

echo "Running on: $(hostname)"
nvidia-smi
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

  for SMOOTH in "${SMOOTHS[@]}"; do
    echo ""
    echo "Running eigen diagnostic: year=${YEAR}, smooth=${SMOOTH}"
    echo "Current date and time: $(date)"

    srun python "${SCRIPT}" \
      --input "${DATA_PATH}" \
      --output-root "${YEAR_OUT}" \
      --year "${YEAR}" \
      --month "${MONTH}" \
      --days "${DAY_RANGE}" \
      --hours "${HOURS}" \
      --hour-match slot \
      --smooth "${SMOOTH}" \
      --regions "tiles4x4,sparse" \
      --tile-y 4 \
      --tile-x 4 \
      --sparse-strides "4" \
      --variants "nugget0" \
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
      --nugget-init 1.0 \
      --min-points 80 \
      --max-points 0 \
      --max-eig-points 0 \
      --cov-jitter 1e-8
  done
done

echo "Current date and time: $(date)"
```

### Monitor
```
tail -f /home/jl2815/tco/exercise_output/logs/real_eig_gv_july_<JOBID>.out
```

### Output
```
/home/jl2815/tco/exercise_output/eda/real_data/eigen_diag_july_group_vecchia_111131_052726/{YEAR}/nu0p3
/home/jl2815/tco/exercise_output/eda/real_data/eigen_diag_july_group_vecchia_111131_052726/{YEAR}/nu0p5
```
