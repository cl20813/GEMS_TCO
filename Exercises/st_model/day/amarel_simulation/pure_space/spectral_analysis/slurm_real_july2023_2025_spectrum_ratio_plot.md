# Real July 2023-2025 Pure-Space Spectrum Ratio Plot

Canonical pure-space spectrum-ratio runbook.

- years: 2023, 2024, 2025
- region: latitude `-3..2`, longitude `121..131`
- nugget: fixed `0`
- baseline: Matérn smooth `0.3`
- model candidates:
  - 2023: `matern_s03`, `gc_a075_b1`
  - 2024: `matern_s03`, `gc_a08_b1`
  - 2025: `matern_s03`, `gc_a075_b1`
- averaging: pure-space hourly mean over July, usually `30 days x 8 hours = 240 hours`
- no whitening: temporal 8x8 whitening is only for the spatio-temporal diagnostic
- profiles: norm, latitude, longitude, and NE-SW diagonal
- wording: plot axes use `norm frequency`

The Python script is:

```text
/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2023_2025_spectrum_ratio_plot.py
```

## Transfer

Run from the local Mac:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis"
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_SPECTRAL="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/pure_space/spectral_analysis"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR} /home/jl2815/tco /home/jl2815/tco/exercise_output/logs"

scp -r "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp \
  "${LOCAL_SPECTRAL}/real_july2023_2025_spectrum_ratio_plot.py" \
  "${LOCAL_SPECTRAL}/slurm_real_july2023_2025_spectrum_ratio_plot.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis
nano run_real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot.sh
sbatch run_real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=ps_spec_gc
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/ps_spec_gc_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/ps_spec_gc_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

module purge || true
module use /projects/community/modulefiles || true
module load anaconda/2024.06-ts840 || true
module load cuda/12.1.0 || true

if ! command -v conda >/dev/null 2>&1; then
  source "${HOME}/.bashrc" || true
fi

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2023_2025_spectrum_ratio_plot.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot"
TOPPLOTS="${OUTROOT}/monthly_plots_top"
YEARS=(2023 2024 2025)
YEAR="${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}"
VARIANTS="matern_s03,gc_a075_b1,gc_a08_b1"

mkdir -p "${OUTROOT}" "${TOPPLOTS}" /home/jl2815/tco/exercise_output/logs
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Started: $(date)"
echo "Year: ${YEAR}"
echo "Experiment: pure-space spectrum ratio, Matern s=0.3 baseline plus final year-specific GC, nugget0"
echo "Output root: ${OUTROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

srun python "${SCRIPT}" \
  --years "${YEAR}" \
  --month 7 \
  --days "1,30" \
  --smooths "0.3" \
  --block-prefixes "all" \
  --variants "${VARIANTS}" \
  --domain-modes "full" \
  --cluster-neighbor-blocks 2 \
  --cluster-block-shape 4x4 \
  --mean-design lat \
  --data-root "/home/jl2815/tco/data" \
  --output-root "${OUTROOT}" \
  --top-plot-dir "${TOPPLOTS}" \
  --expanded-bounds \
  --lat-range "-3,2" \
  --lon-range "121,131" \
  --combined-profiles "radial,lat,lon,diag" \
  --combined-ratio-normalize \
  --no-hann \
  --device cuda \
  --cuda-fallback cpu \
  --target-chunk-size 512 \
  --lbfgs-steps 8 \
  --lbfgs-eval 20 \
  --lbfgs-history 10 \
  --radial-bins 70 \
  --radial-qmax 0.985 \
  --skip-existing

echo "Finished: $(date)"
```

## Main Outputs

Remote output root:

```text
/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot
```

Most important comparison plots:

```text
monthly_plots_top/YYYY_07/
  smooth_0p3_combined_YYYY07_full_model_compare_profile_out_I_over_EI_ratio_norm.png
  smooth_0p3_combined_YYYY07_full_model_compare_profile_out_I_over_EI_ratio_lat.png
  smooth_0p3_combined_YYYY07_full_model_compare_profile_out_I_over_EI_ratio_lon.png
  smooth_0p3_combined_YYYY07_full_model_compare_profile_out_I_over_EI_ratio_diag.png
```

Monthly numeric summaries are written beside the domain outputs:

```text
YYYY_07/smooth_0p3/<domain_group>/<domain_label>/monthly_average/
  YYYY07_hourly_mean_curves.csv
  YYYY07_30day_mean_curves.csv
```

`YYYY07_30day_mean_curves.csv` is kept as a compatibility alias, but its values are now the pure-space hourly mean over all available July hours.

## Pull Results

Run from the local Mac:

```bash
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot"
REMOTE_OUT="/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot"

mkdir -p "${LOCAL_OUT}"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/monthly_plots_top" \
  "${LOCAL_OUT}/"

scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/2023_07/smooth_0p3/combined_domain_plots" \
  "${LOCAL_OUT}/2023_07_combined_domain_plots"
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/2024_07/smooth_0p3/combined_domain_plots" \
  "${LOCAL_OUT}/2024_07_combined_domain_plots"
scp -r "jl2815@amarel.rutgers.edu:${REMOTE_OUT}/2025_07/smooth_0p3/combined_domain_plots" \
  "${LOCAL_OUT}/2025_07_combined_domain_plots"
```
