# Real July 2024 x1 Slice/Tile Spectral Ratio Diagnostics, 2026-06-09

Legacy 2024-only runbook.  The maintained script now lives in this
`spectral_analysis` folder as `real_july2022_2025_spectrum_ratio_plot.py`; this
runbook pins `--years "2024"` to reproduce the old 2024-only run.

This run checks whether the strange latitude-direction `I / E[I]` ratio is a
large-scale trend problem by fitting independent x1/full-resolution domains.

```text
year    = 2024 only
month   = July
days    = July 1..30
smooth  = 0.25, 0.3, 0.35, 0.5
variant = nugget0 only
prefix  = all only, so no resolution sequence
region  = lat -3..2, lon 121..131
domains = 5 latitude slices, 5 longitude slices, and 2x4 tiles
```

Default longitude slices are five 2-degree intervals over `121..131`:

```text
121..123, 123..125, 125..127, 127..129, 129..131
```

Slice/tile domains are half-open on internal upper bounds and inclusive on the
last upper bound, so boundary grid cells are not duplicated across neighboring
domains.

The combined plots replace the old `2 variants x 4 block-prefixes` view.  For
`tile_2x4`, each panel is one fitted spatial tile.  Two combined plot families
are written for every selected profile (`radial`, `lat`, `lon`, `diag`):

```text
I vs E[I] with gray daily spectra, black monthly mean I, red E[I], and blue ratio
E[I] vs continuous theoretical spectrum
```

## Transfer Code

Update the package code:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco"
```

Transfer the run script:

```bash
ssh jl2815@amarel.rutgers.edu \
  'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2022_2025_spectrum_ratio_plot.py" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/"
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis
nano slurm_real_july2024_spline_smooth_spectral_x1_slice_tile_060926.sh
sbatch slurm_real_july2024_spline_smooth_spectral_x1_slice_tile_060926.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=jul24_x1_dom
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/jul24_x1_dom_060926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/jul24_x1_dom_060926_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu037

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

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/spectral_analysis/real_july2022_2025_spectrum_ratio_plot.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_x1_slice_tile_060926"
TOPPLOTS="${OUTROOT}/monthly_plots_top"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${TOPPLOTS}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Started: $(date)"
echo "Experiment: 2024 July x1/all, nugget0, lat/lon slices plus 2x4 tiles"
echo "Output root: ${OUTROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

srun python "${SCRIPT}" \
  --years "2024" \
  --month 7 \
  --days "1,30" \
  --smooths "0.25,0.3,0.35,0.5" \
  --block-prefixes "all" \
  --variants "nugget0" \
  --domain-modes "lat_slices,lon_slices,tile_2x4" \
  --lat-slice-count 5 \
  --lon-slice-count 5 \
  --tile-grid "2x4" \
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
/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_x1_slice_tile_060926
```

Per-domain folders:

```text
2024_07/smooth_0p3/lat_slices/lat_m3tom2/
2024_07/smooth_0p3/lon_slices/lon_121to123/
2024_07/smooth_0p3/tile_2x4/tile_y01_x01/
```

Combined domain plots:

```text
2024_07/smooth_0p3/combined_domain_plots/
  202407_lat_slices_nugget0_data_vs_expected_lat.png
  202407_lat_slices_nugget0_expected_vs_continuous_lat.png
  202407_lon_slices_nugget0_data_vs_expected_lon.png
  202407_lon_slices_nugget0_expected_vs_continuous_lon.png
  202407_tile_2x4_nugget0_data_vs_expected_radial.png
  202407_tile_2x4_nugget0_data_vs_expected_lat.png
  202407_tile_2x4_nugget0_data_vs_expected_lon.png
  202407_tile_2x4_nugget0_data_vs_expected_diag.png
  202407_tile_2x4_nugget0_expected_vs_continuous_radial.png
  202407_tile_2x4_nugget0_expected_vs_continuous_lat.png
  202407_tile_2x4_nugget0_expected_vs_continuous_lon.png
  202407_tile_2x4_nugget0_expected_vs_continuous_diag.png
```

Top-level copies:

```text
monthly_plots_top/2024_07/
```

## Pull Results To Local

Run from the local Mac after the Amarel job finishes:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_x1_slice_tile_060926" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```

Local result folder:

```text
/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/spline_smooth_spectral_x1_slice_tile_060926
```
