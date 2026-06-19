# Space-Time Corridor Spectral Profile: Fine-Tuned Beta, July 2023-2025

Runbook for comparing the fixed Matérn baseline against one day-specific
fine-tuned beta model for each July day in 2023, 2024, and 2025.

```text
model geometry: corridor Vecchia cluster
block shape: 4x4
lag pattern: 6/4/3
spatial thinning: none, full grid
nugget: fixed 0
baseline: Matérn smooth=0.3 nugget0
alternative: fine_tuned_beta, selected by year and day_idx
diagnostic: 8x8 finite-sample cross-periodogram whitening
directions: norm, lat, lon, diag
```

Fine-tuned beta rules:

```text
2023: GC a=0.75, day-specific b, with selected Matérn s=0.3 days
2024: GC a=0.8,  day-specific b, with selected Matérn s=0.3 days
2025: GC a=0.75, day-specific b, with selected Matérn s=0.3 days
```

Main outputs:

```text
st_corridor_spectral_all_fits.csv
st_corridor_spectral_profiles.csv
st_corridor_spectral_monthly_summary.csv
st_corridor_spectral_representative_frequency_band_table.csv
st_corridor_spectral_baseline_comparison.csv
monthly_average_plots/year_2023/ratio_triptych_norm.png
monthly_average_plots/year_2024/ratio_triptych_norm.png
monthly_average_plots/year_2025/ratio_triptych_norm.png
monthly_average_plots/year_<year>/daily_norm_ratio_plots/dayidx_XX_norm_I_over_EI_profile_ratio.png
monthly_average_plots/year_<year>/daily_norm_ratio_plots/daily_norm_I_over_EI_profile_ratio_rows.csv
```

The comparison table reports tuned-minus-baseline loss and spectral-ratio
improvements.  Positive `ratio_abslog_improvement`,
`ratio_rmslog_improvement`, or `low_bin0_5_ratio_abslog_improvement` means the
fine-tuned beta model is closer to target 1 than the baseline for that
day/direction.

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_fine_tuned_beta_nugget0_061726
```

## 1. Upload Scripts To Amarel

Run this from the local Mac before submitting the job:

```bash
REMOTE_SPACE_TIME="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
REMOTE_SPECTRAL="${REMOTE_SPACE_TIME}/spectral_analysis"
REMOTE_DIAG="${REMOTE_SPACE_TIME}/vecchia_diagnosis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_SPACE_TIME="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time"
LOCAL_SPECTRAL="${LOCAL_SPACE_TIME}/spectral_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_SPECTRAL} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_SPECTRAL}/st_corridor_common.py" \
  "${LOCAL_SPECTRAL}/real_july_st_corridor_spectral_profile_fine_tuned_beta_061726.py" \
  "${LOCAL_SPECTRAL}/slurm_real_july_st_corridor_spectral_profile_fine_tuned_beta_061726.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SPECTRAL}/"
```

## 2. Submit On Amarel

On Amarel, create the Slurm script:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_fine_tuned_beta_061726.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=stspec_ftbeta
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stspec_ftbeta_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stspec_ftbeta_%A_%a.err

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
export PYTHONPATH="/home/jl2815/tco:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_diagnosis:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/real_july_st_corridor_spectral_profile_fine_tuned_beta_061726.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_fine_tuned_beta_nugget0_061726"
YEARS=(2023 2024 2025)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR="${OUTROOT}/year_${YEAR}"
TOPPLOTS="${OUTROOT}/monthly_average_plots"
MODEL_VARIANTS=(matern_s03 fine_tuned_beta)

mkdir -p "${OUTDIR}" "${TOPPLOTS}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "YEAR=${YEAR}"
echo "MODEL_VARIANTS=${MODEL_VARIANTS[*]}"
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
  --model-variants "${MODEL_VARIANTS[@]}" \
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
  --target-chunk-size 64 \
  --oom-retry-target-chunk-sizes 32,16 \
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
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_fine_tuned_beta_061726.sh
```

Check status/logs:

```bash
squeue -u jl2815

tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_ftbeta_<JOBID>_0.out
tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_ftbeta_<JOBID>_1.out
tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_ftbeta_<JOBID>_2.out
```

## 3. Pull Results To Local

After the array finishes:

```bash
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_2023_2025_fine_tuned_beta_nugget0_061726"
REMOTE_OUT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_fine_tuned_beta_nugget0_061726"

mkdir -p "${LOCAL_OUT}"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/monthly_average_plots" \
  "${LOCAL_OUT}/"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/year_2023/st_corridor_spectral_baseline_comparison.csv" \
  "${LOCAL_OUT}/st_corridor_spectral_baseline_comparison_2023.csv"
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/year_2024/st_corridor_spectral_baseline_comparison.csv" \
  "${LOCAL_OUT}/st_corridor_spectral_baseline_comparison_2024.csv"
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/year_2025/st_corridor_spectral_baseline_comparison.csv" \
  "${LOCAL_OUT}/st_corridor_spectral_baseline_comparison_2025.csv"
```

To copy the full CSV/log output:

```bash
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
