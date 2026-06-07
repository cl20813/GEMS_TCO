# Real July Spline-Smooth Spectral Diagnostics, Updated 2026-06-06

This run fits real July GEMS pure-space slices with cluster/group Vecchia and
then compares residual periodograms against fitted finite-sample expected
periodograms.

```text
years   = 2022, 2023, 2024, 2025
month   = July
days    = July 1..30
smooth  = 0.25, 0.3
region  = lat -3..2, lon 121..131
Vecchia = 4x4 target blocks, condition on 2 previous max-min cluster blocks
```

Important diagnostic convention:

```text
No Hann/tapering is used.
I and E[I] both use the observed/missing grid directly.
The expected periodogram uses only the missing-data mask/window autocorrelation.
```

Monthly plots include radial, latitude N-S, longitude E-W, and diagonal NE-SW
profiles. A separate monthly plot compares fitted finite-sample `E[I]` against
the continuous theoretical spectrum after Whittle-style sigma profiling, so the
shape is what matters. The profile scale uses the reference notebook convention:

```text
profile scale = mean(E[I] / S_continuous)
S_profiled    = profile scale * S_continuous
```

That plot separates the latent Matérn spectrum from the observed continuous
spectrum with the nugget added:

```text
latent continuous S:          Matérn spectrum only, no nugget
observed continuous S+nugget: Matérn spectrum plus flat nugget contribution
```

For the latent comparison, both sides exclude nugget:

```text
finite-sample E[I] latent vs latent continuous S
```

For the observed comparison, both sides include nugget:

```text
finite-sample E[I] observed vs observed continuous S+nugget
```

This avoids hiding the fact that nugget is not part of the latent theoretical
smooth spectrum.

The blue `I/E[I]` ratio curves use the same sigma-profile idea as the reference
notebook: first compute the raw ratio, estimate its mean scale, then plot
`raw ratio / mean(raw ratio)`. The legend reports both the mean scale and
`profile sigma^2 = fitted sigma^2 * mean(raw ratio)`. With a free nugget this is
best read as a global amplitude profile diagnostic, not a re-fit of only
`sigma^2`.

The script first looks for the exact `lat-3to2_lon121to131` pickle. If that is
not present, it can fall back to the older expanded `lat-3to7_lon111to131`
pickle and then filters the dataframe to `lat -3..2, lon 121..131`.

## Transfer Code

Update the package code:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

Transfer the run script:

```bash
ssh jl2815@amarel.rutgers.edu \
    'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space /home/jl2815/tco/exercise_output/logs'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_060626.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## Submit

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_real_july_spline_smooth_spectral_060626.sh
sbatch slurm_real_july_spline_smooth_spectral_060626.sh
```

Paste this sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=real_jul_spl
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_jul_spl_060626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_jul_spl_060626_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu042

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SMOOTHS=(0.25 0.3)
YEARS=(2022 2023 2024 2025)
MONTH=7

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/real_july_spline_smooth_spectral_060626.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_060626"
TOPPLOTS="${OUTROOT}/monthly_plots_top"
LOGROOT="/home/jl2815/tco/exercise_output/logs"

mkdir -p "${OUTROOT}" "${TOPPLOTS}" "${LOGROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"
echo "Single Slurm job; looping over all smooth/year combinations."
echo "YEARS=${YEARS[*]}"
echo "SMOOTHS=${SMOOTHS[*]}"
echo "Region: lat -3..2, lon 121..131"
echo "No Hann/tapering; mask/window check only."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

for SMOOTH in "${SMOOTHS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        echo ""
        echo "============================================================"
        echo "Starting YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
        echo "Current date and time: $(date)"
        echo "============================================================"

        srun python "${SCRIPT}" \
            --years "${YEAR}" \
            --month "${MONTH}" \
            --days "1,30" \
            --smooths "${SMOOTH}" \
            --resolutions "8,4,2,1" \
            --variants "nugget0,nugget_free" \
            --cluster-neighbor-blocks 2 \
            --cluster-block-shape 4x4 \
            --mean-design lat \
            --data-root "/home/jl2815/tco/data" \
            --output-root "${OUTROOT}" \
            --top-plot-dir "${TOPPLOTS}" \
            --expanded-bounds \
            --lat-range "-3,2" \
            --lon-range "121,131" \
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

        echo "Finished YEAR=${YEAR}, MONTH=${MONTH}, SMOOTH=${SMOOTH}"
        echo "Current date and time: $(date)"
    done
done

echo "All done: $(date)"
```

The sbatch file launches one Slurm job:

```text
2 smooth values x 4 years = 8 smooth/year runs
```

## Output

Remote output root:

```text
/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_060626
```

Important folders:

```text
smooth_0p35/
  2022_07/
    daily_csv/
      20220701_fits.csv
      20220701_spectral_profiles.csv
    daily_plots/
      20220701_data_vs_expected_periodogram.png
    monthly_average/
      202207_30day_mean_data_vs_expected_periodogram.png
      202207_30day_mean_directional_data_vs_expected_periodogram.png
      202207_expected_periodogram_vs_continuous_theoretical_scaled.png
      202207_daily_mean_curves.csv
      202207_30day_mean_curves.csv

monthly_plots_top/
  smooth_0p35_202207_30day_mean_data_vs_expected_periodogram.png
  smooth_0p35_202207_30day_mean_directional_data_vs_expected_periodogram.png
  smooth_0p35_202207_expected_periodogram_vs_continuous_theoretical_scaled.png
  ...
```

## Transfer Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/real_data/spline_smooth_spectral_060626" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer/"
```

## Notes

The binning is locked to the first full July grid for each year. Every hourly
dataframe is reindexed onto that same `(lat, lon)` grid before thinning, so a
missing cell becomes `NaN` instead of shifting the x8/x4/x2/x1 grid positions.

The daily plot first averages over the 8 hours. The monthly plot then averages
daily mean curves, so each day has equal weight.
