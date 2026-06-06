# Space-Time Corridor Vecchia Spectral Profile Diagnostic, 2026-06-06

This runs the full-grid real-data July space-time diagnostic:

- model: corridor Vecchia cluster, 4x4 target blocks, lag pattern 6/4/3
- data: real GEMS TCO July, region `lat [-3, 2]`, `lon [121, 131]`
- spatial thinning: none
- spectral diagnostic: raw residual field, no taper, missing-window correction, 8x8 time whitening, global profile-out
- output root on Amarel: `/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626`

## 1. Upload Scripts To Amarel

Run this from the local Mac before submitting the job:

```bash
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time"
LOCAL_DIR="/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time"

ssh jl2815@amarel.rutgers.edu "mkdir -p ${REMOTE_DIR}"

scp \
  "${LOCAL_DIR}/real_july_st_corridor_spectral_profile_060626.py" \
  "${LOCAL_DIR}/slurm_real_july_st_corridor_spectral_profile_060626.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## 2. Submit On Amarel

On Amarel, create the Slurm script with nano:

```bash
nano /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/slurm_real_july_st_corridor_spectral_profile_060626.sh
```

Paste this bash block into nano, then save:

```bash
#!/bin/bash
#SBATCH --job-name=st_spec_060626
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/jl2815/tco/exercise_output/summer/slurm_logs/st_spec_060626_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/summer/slurm_logs/st_spec_060626_%A_%a.err

set -euo pipefail

source ~/.bashrc
conda activate faiss_env

mkdir -p /home/jl2815/tco/exercise_output/summer/slurm_logs
export MPLCONFIGDIR="/tmp/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/real_july_st_corridor_spectral_profile_060626.py"
OUTROOT="/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626"

YEARS=(2022 2023 2024 2025)
SMOOTHS=(0.35 0.4 0.5)

YEAR_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SMOOTHS[@]}))
SMOOTH_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SMOOTHS[@]}))

YEAR="${YEARS[$YEAR_INDEX]}"
SMOOTH="${SMOOTHS[$SMOOTH_INDEX]}"
SMOOTH_TAG=$(echo "${SMOOTH}" | sed 's/-/m/g; s/\./p/g')
OUTDIR="${OUTROOT}/year_${YEAR}/smooth_${SMOOTH_TAG}"
TOPPLOTS="${OUTROOT}/monthly_plots_top/year_${YEAR}/smooth_${SMOOTH_TAG}"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "YEAR=${YEAR}"
echo "SMOOTH=${SMOOTH}"
echo "OUTDIR=${OUTDIR}"

python "${SCRIPT}" \
  --real-years "${YEAR}" \
  --smooths "${SMOOTH}" \
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
  --skip-existing
```

Then submit:

```bash
sbatch /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/slurm_real_july_st_corridor_spectral_profile_060626.sh
```

Check status/logs:

```bash
squeue -u jl2815

tail -n 80 /home/jl2815/tco/exercise_output/summer/slurm_logs/st_spec_060626_<JOBID>_0.out
```

## 3. Output Locations

Full Amarel output:

```bash
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626
```

Task outputs:

```bash
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626/year_2022/smooth_0p35
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626/year_2022/smooth_0p4
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626/year_2022/smooth_0p5
...
```

Top monthly plot folder:

```bash
/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626/monthly_plots_top
```

## 4. Copy Monthly Plots Back To Local

After the array finishes, copy only the monthly/top plots to local:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_060626"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626/monthly_plots_top" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_060626/"
```

If you want the full CSV/log output too:

```bash
scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626" \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
