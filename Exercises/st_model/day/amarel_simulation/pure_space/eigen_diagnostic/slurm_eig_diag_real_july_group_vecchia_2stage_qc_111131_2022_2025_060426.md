# Two-Stage QC Real July Eigen Diagnostic, Group Vecchia, 2022-2025

This run fits real July GEMS data with pure-space group Vecchia, performs a
two-stage extreme whitened-residual QC refit, and writes monthly-average eigen
diagnostic outputs only.

Settings:

```text
years: 2022, 2023, 2024, 2025
smooth values: 0.2, 0.3, 0.4, 0.5
domain: lat -3..7, lon 111..131 input files
diagnostic regions: 2x4 tiles + whole-domain sparse x4 stride
Vecchia: 4x4 target blocks, condition on 2 previous max-min cluster blocks
covariance: anisotropic pure-space Matern, estimating sigmasq, range_lat, range_lon, nugget
variant: nugget_free
mean: lat, GLS-profiled in fitting
QC: fit once, compute detrended Vecchia whitened residuals, set |w|>10 to missing, refit
outputs: monthly average curves/plots plus monthly QC/eigen summaries only
```

## Transfer

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic /home/jl2815/tco"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_2stage_qc_111131_060426.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/

ssh jl2815@amarel.rutgers.edu "cd /home/jl2815/tco && python -c 'import GEMS_TCO.kernels_space_iso_cluster_052426; print(\"GEMS_TCO group Vecchia imports ok\")'"
```

If the expanded July files are not already on Amarel:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2022 2023 2024 2025 --months 7 --only-extra-bounds
```

Expected inputs:

```text
/home/jl2815/tco/data/pickle_2022/tco_grid_lat-3to7_lon111to131_22_07.pkl
/home/jl2815/tco/data/pickle_2023/tco_grid_lat-3to7_lon111to131_23_07.pkl
/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl
/home/jl2815/tco/data/pickle_2025/tco_grid_lat-3to7_lon111to131_25_07.pkl
```

## Submit

On Amarel:

```bash
cd ./jobscript/tco/gp_exercise
nano eig_diag_real_july_gv_2stage_qc_111131_060426.sh
sbatch eig_diag_real_july_gv_2stage_qc_111131_060426.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=eig_gv_qc_july
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/eig_gv_qc_july_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/eig_gv_qc_july_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu024

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

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/eigen_diagnostic/eig_diag_real_july_group_vecchia_2stage_qc_111131_060426.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131

YEARS=(2022 2023 2024 2025)
SMOOTHS=(0.2 0.3 0.4 0.5)
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
    echo "Running 2-stage QC eigen diagnostic: year=${YEAR}, smooth=${SMOOTH}"
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
      --regions "tiles,sparse" \
      --tile-y 2 \
      --tile-x 4 \
      --sparse-strides "4" \
      --variants "nugget_free" \
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
      --nugget-init 1.0 \
      --min-points 80 \
      --max-points 0 \
      --max-eig-points 0 \
      --cov-jitter 1e-8 \
      --qc-whitened-threshold 10
  done
done

echo "Current date and time: $(date)"
```

## Monitor

```bash
tail -f /home/jl2815/tco/exercise_output/logs/eig_gv_qc_july_<JOBID>.out
```

## Outputs

The run writes into:

```text
/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131/{YEAR}/nu0p2
/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131/{YEAR}/nu0p3
/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131/{YEAR}/nu0p4
/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131/{YEAR}/nu0p5
```

Expected files in each `nu...` folder:

```text
real_eigen_monthly_average_curves.csv
real_eigen_monthly_qc_summary.csv
nugget_free_tiles2x4_monthly_average_overview.png
nugget_free_x4_monthly_average_eigdiag.png
real_eigen_diagnostic_math_notes.txt
```

The script does not save per-hour diagnostic plots unless `--save-hourly-plots`
is added. It also does not save per-hour row CSVs unless `--save-hourly-rows`
is added.

## Retrieve

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer/eigen_analysis"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/eigen_analysis/group_vecchia_aniso_nuggetfree_2stage_qc_w10_111131 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer/eigen_analysis/"
```
