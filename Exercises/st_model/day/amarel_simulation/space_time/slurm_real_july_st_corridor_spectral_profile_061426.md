# Space-Time Corridor Spectral Profile Diagnostic, Matérn vs Cauchy Nugget-Zero Refine

This runs the full-grid real-data July space-time spectral diagnostic for the
June 14 nugget-zero Matérn/Cauchy candidate set:

```text
model geometry: corridor Vecchia cluster
block shape: 4x4
lag pattern: 6/4/3
spatial thinning: none, full grid
nugget: fixed 0
data: real GEMS TCO July, lat [-3, 2], lon [121, 131]
diagnostics: directional I / E[I], E[I] / continuous-like spectrum, whitened profile
```

Model variants:

```text
2023: matern_s03, gc_a07_b1, gc_a075_b1, gc_a065_b1
2024: matern_s03, gc_a07_b05, gc_a09_b05, gc_a08_b1
2025: matern_s03, gc_a075_b1, gc_a075_b07, gc_a075_b05
```

Important outputs:

```text
st_corridor_spectral_all_fits.csv
st_corridor_spectral_profiles.csv
st_corridor_spectral_monthly_summary.csv
st_corridor_spectral_ratio_band_table.csv
st_corridor_spectral_monthly_I_over_EI_profile.png
st_corridor_spectral_monthly_EI_over_continuous_profile.png
st_corridor_spectral_monthly_whitened_profile.png
st_corridor_parameter_monthly_summary.png
```

The band table is also copied into each monthly/top plot folder. It summarizes
representative `dc_first_bin`, `low_bins_1_5`, `mid_band`, and `high_band`
ratios, especially `ratio_I_over_EI_*` and `ratio_EI_over_continuous_*`.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426
```

## 1. Upload Scripts To Amarel

Run this from the local Mac before submitting the job:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
REMOTE_REAL_DIR="${REMOTE_DIR}/real_data"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time"
LOCAL_REAL="${LOCAL_DIR}/real_data"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} ${REMOTE_REAL_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_DIR}/real_july_st_corridor_spectral_profile_061426.py" \
  "${LOCAL_DIR}/slurm_real_july_st_corridor_spectral_profile_061426.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"

scp \
  "${LOCAL_REAL}/fit_real_july2023_2025_corridor_width_4x4_lag643_matern_cauchy_nugget0_prefix_061426.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_REAL_DIR}/"
```

## 2. Submit On Amarel

On Amarel, create the Slurm script:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/slurm_real_july_st_corridor_spectral_profile_061426.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=stspec_ref
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stspec_ref_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stspec_ref_%A_%a.err

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840 || true
module load cuda/12.1.0 || true

if ! command -v conda >/dev/null 2>&1; then
  source "${HOME}/.bashrc" || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found after module load and ~/.bashrc fallback." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate faiss_env

mkdir -p /home/jl2815/tco/exercise_output/summer/logs
export PYTHONPATH="/home/jl2815/tco:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_data:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_july_st_corridor_spectral_profile_061426.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426"

CASES=(
  "2023:matern_s03"
  "2023:gc_a07_b1"
  "2023:gc_a075_b1"
  "2023:gc_a065_b1"
  "2024:matern_s03"
  "2024:gc_a07_b05"
  "2024:gc_a09_b05"
  "2024:gc_a08_b1"
  "2025:matern_s03"
  "2025:gc_a075_b1"
  "2025:gc_a075_b07"
  "2025:gc_a075_b05"
)

CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
YEAR="${CASE%%:*}"
MODEL_VARIANT="${CASE##*:}"
OUTDIR="${OUTROOT}/year_${YEAR}/${MODEL_VARIANT}"
TOPPLOTS="${OUTROOT}/monthly_plots_top/year_${YEAR}/${MODEL_VARIANT}"

mkdir -p "${OUTDIR}" "${TOPPLOTS}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "YEAR=${YEAR}"
echo "MODEL_VARIANT=${MODEL_VARIANT}"
echo "OUTDIR=${OUTDIR}"
which python
nvidia-smi || true
python - <<'PY'
import numpy, scipy, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

python "${SCRIPT}" \
  --real-years "${YEAR}" \
  --model-variants "${MODEL_VARIANT}" \
  --days 0,30 \
  --lat-range=-3,2 \
  --lon-range=121,131 \
  --space "1,1" \
  --out-dir "${OUTDIR}" \
  --monthly-out-dir "${TOPPLOTS}" \
  --device cuda \
  --cuda-fallback error \
  --spline-n-points 4000 \
  --spline-r-max 30 \
  --target-chunk-size 128 \
  --lbfgs-steps 5 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --summary-every 1 \
  --skip-existing \
  --suppress-fit-prints

echo "Finished: $(date)"
```

Submit:

```bash
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/slurm_real_july_st_corridor_spectral_profile_061426.sh
```

Check status/logs:

```bash
squeue -u jl2815

tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_ref_<JOBID>_0.out
```

## 3. Pull Results To Local

After the array finishes, copy the top plots and ratio tables:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426/monthly_plots_top" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426/"
```

To copy the full CSV/log output:

```bash
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_matern_cauchy_nugget0_refine_061426" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
