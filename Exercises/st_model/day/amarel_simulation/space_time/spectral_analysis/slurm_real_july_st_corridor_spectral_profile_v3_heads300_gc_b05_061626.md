# Space-Time Corridor Spectral Profile Diagnostic v3: GC b=0.5 Standard vs Heads300

Testing runbook for the real July space-time spectral diagnostic v3.

```text
model geometry: corridor Vecchia cluster
block shape: 4x4
lag pattern: 6/4/3
spatial thinning: none, full grid
nugget: fixed 0
data: real GEMS TCO July, lat [-3, 2], lon [121, 131]
diagnostic: 8x8 finite-sample cross-periodogram whitening, pooled by frequency direction
directions: norm, lat, lon, diag
heads variant: add one exact full-GP likelihood on 300 early max-min head points per time slot
```

Model variants:

```text
2023: gc_a075_b05_standard, gc_a075_b05_heads300
2024: gc_a08_b05_standard,  gc_a08_b05_heads300
2025: gc_a075_b05_standard, gc_a075_b05_heads300
```

The heads300 objective keeps the original Vecchia tail objective and adds an
exact likelihood contribution on up to `300 * 8 = 2400` single head points per
day.  The head points are not removed from the Vecchia tail, so loss values are
pseudo-likelihood diagnostics; compare spectral ratios and parameter stability.

Main outputs:

```text
st_corridor_spectral_all_fits.csv
st_corridor_spectral_profiles.csv
st_corridor_spectral_monthly_summary.csv
st_corridor_spectral_representative_frequency_band_table.csv
monthly_average_plots/year_2023/marginal_timeavg_spatial_I_over_Ediag_profile_sigma_target1_norm.png
monthly_average_plots/year_2023/marginal_timeavg_spatial_Ediag_over_continuous_ratio_norm.png
monthly_average_plots/year_2023/whitened_8x8_I_vs_EI_target1_norm.png
monthly_average_plots/year_2023/ratio_triptych_norm.png
monthly_average_plots/year_2023/ratio_triptych_lat.png
monthly_average_plots/year_2023/ratio_triptych_lon.png
monthly_average_plots/year_2023/ratio_triptych_diag.png
monthly_average_plots/year_2023/daily_norm_ratio_plots/dayidx_00_norm_I_over_EI_profile_ratio.png
monthly_average_plots/year_2023/st_corridor_spectral_monthly_summary.csv
monthly_average_plots/year_2023/st_corridor_spectral_representative_frequency_band_table.csv
```

Amarel output root:

```text
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_gc_b05_heads300_v3_061626
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
  "${LOCAL_SPECTRAL}/real_july_st_corridor_spectral_profile_v3_heads300_gc_b05_061626.py" \
  "${LOCAL_SPECTRAL}/slurm_real_july_st_corridor_spectral_profile_v3_heads300_gc_b05_061626.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SPECTRAL}/"

scp \
  "${LOCAL_DIAG}/fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIAG}/"
```

## 2. Submit On Amarel

On Amarel, create the Slurm script:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_v3_heads300_gc_b05_061626.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=stspec_v3h
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/logs/stspec_v3h_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/logs/stspec_v3h_%A_%a.err

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

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/real_july_st_corridor_spectral_profile_v3_heads300_gc_b05_061626.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_gc_b05_heads300_v3_061626"

YEARS=(2023 2024 2025)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"
if [[ "${YEAR}" == "2024" ]]; then
  MODEL_VARIANTS=(gc_a08_b05_standard gc_a08_b05_heads300)
else
  MODEL_VARIANTS=(gc_a075_b05_standard gc_a075_b05_heads300)
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
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/spectral_analysis/slurm_real_july_st_corridor_spectral_profile_v3_heads300_gc_b05_061626.sh
```

Check status/logs:

```bash
squeue -u jl2815

tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_v3h_<JOBID>_0.out
tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_v3h_<JOBID>_1.out
tail -n 80 /home/jl2815/tco/exercise_output/summer/logs/stspec_v3h_<JOBID>_2.out
```
