# Space-Time Corridor Spectral Profile Diagnostic, 2023-2025 Matérn vs GC

Canonical runbook for the real July space-time spectral diagnostic.

```text
model geometry: corridor Vecchia cluster
block shape: 4x4
lag pattern: 6/4/3
spatial thinning: none, full grid
nugget: fixed 0
data: real GEMS TCO July, lat [-3, 2], lon [121, 131]
diagnostic: 8x8 finite-sample cross-periodogram whitening, pooled by frequency direction
directions: norm, lat, lon, diag
```

Model variants:

```text
2023: matern_s03, gc_a075_b1
2024: matern_s03, gc_a08_b1
2025: matern_s03, gc_a075_b1
```

Main outputs:

```text
st_corridor_spectral_all_fits.csv
st_corridor_spectral_profiles.csv
st_corridor_spectral_monthly_summary.csv
st_corridor_spectral_representative_frequency_band_table.csv
monthly_average_plots/year_2023/marginal_timeavg_spatial_I_vs_Ediag_profile_sigma_norm.png
monthly_average_plots/year_2023/marginal_timeavg_spatial_Ediag_vs_continuous_profile_sigma_norm.png
monthly_average_plots/year_2023/whitened_8x8_I_vs_EI_target1_norm.png
monthly_average_plots/year_2023/st_corridor_spectral_monthly_summary.csv
monthly_average_plots/year_2023/st_corridor_spectral_representative_frequency_band_table.csv
```

The direction suffixes are `norm`, `lat`, `lon`, and `diag`.  The y labels are
fixed by plot family so the Matérn-vs-GC comparison is visually consistent:

```text
marginal timeavg I vs Ediag: marginal time-averaged spatial spectrum with profile sigma, observed I and diagonal fitted E[I]
marginal timeavg Ediag vs continuous: marginal time-averaged spatial spectrum with profile sigma, diagonal fitted E[I] and theoretical continuous spectrum
whitened 8x8 I vs EI: 8x8-whitened I / E[I] quadratic power, target = 1
```

No per-hour spectrum files are written.  The detailed profile CSV is already
binned by day, direction, model, and frequency bin; the research-facing outputs
are the monthly summary, representative frequency-band table, and monthly plots.
The representative table reports:

```text
lowest_frequency_bin
low_frequency_bins_1_5
mid_frequency_band
high_frequency_band
```

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_061426
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
LOCAL_DIAG="${LOCAL_SPACE_TIME}/vecchia_diagnosis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_SPECTRAL} ${REMOTE_DIAG} /home/jl2815/tco/exercise_output/summer/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_SPECTRAL}/real_july_st_corridor_spectral_profile_061426.py" \
  "${LOCAL_SPECTRAL}/slurm_real_july_st_corridor_spectral_profile_061426.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SPECTRAL}/"

scp \
  "${LOCAL_DIAG}/fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIAG}/"
```

## 2. Submit On Amarel

On Amarel, create the Slurm script:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_061426.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=stspec_y
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stspec_y_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stspec_y_%A_%a.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/real_july_st_corridor_spectral_profile_061426.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_061426"

YEARS=(2023 2024 2025)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"
if [[ "${YEAR}" == "2024" ]]; then
  MODEL_VARIANTS=(matern_s03 gc_a08_b1)
else
  MODEL_VARIANTS=(matern_s03 gc_a075_b1)
fi

OUTDIR="${OUTROOT}/year_${YEAR}"
TOPPLOTS="${OUTROOT}/monthly_average_plots"

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
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_061426.sh
```

Check status/logs:

```bash
squeue -u jl2815

tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_y_<JOBID>_0.out
```

## 3. Pull Results To Local

After the array finishes, copy the research-facing monthly plots and summary
tables:

```bash
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_061426"
REMOTE_OUT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_061426"

mkdir -p "${LOCAL_OUT}"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/monthly_average_plots" \
  "${LOCAL_OUT}/"
```

This brings back the year folders, for example:

```text
monthly_average_plots/year_2023/
  marginal_timeavg_spatial_I_vs_Ediag_profile_sigma_norm.png
  marginal_timeavg_spatial_Ediag_vs_continuous_profile_sigma_norm.png
  whitened_8x8_I_vs_EI_target1_norm.png
  st_corridor_spectral_monthly_summary.csv
  st_corridor_spectral_representative_frequency_band_table.csv
```

To copy the full CSV/log output, including daily binned profiles and fit logs:

```bash
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
