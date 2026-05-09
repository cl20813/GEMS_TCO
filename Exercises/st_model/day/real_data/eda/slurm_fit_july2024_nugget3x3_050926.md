# July 2024 spatial nugget 3x3 EDA

### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```


### Transfer run file (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/real_data/eda"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/eda/fit_july2024_spatial_nugget_tiles.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/real_data/eda/
```

### Transfer estimate results (Amarel -> mac)
```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/july2024_nugget_3x3_compact \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/"
```

---

### Connect & setup
```bash
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
conda activate faiss_env
```

### Quick manifest test
```bash
python /home/jl2815/tco/exercise_25/st_model/real_data/eda/fit_july2024_spatial_nugget_tiles.py \
    --mode manifest \
    --input /home/jl2815/tco/data/pickle_2024/tco_grid_24_07.pkl \
    --output-dir /home/jl2815/tco/exercise_output/eda/july2024_nugget_3x3_compact \
    --month 2024-07 \
    --expected-hours 248 \
    --time-col hour \
    --x-col Source_Longitude \
    --y-col Source_Latitude \
    --value-col ColumnAmountO3
```

---

### July 2024 nugget 3x3 single-node (sbatch)

```bash
cd ./jobscript/tco/gp_exercise
nano fit_july2024_nugget3x3_050926.sh
sbatch fit_july2024_nugget3x3_050926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=nug3x3
#SBATCH --output=/home/jl2815/tco/exercise_output/nug3x3_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/nug3x3_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

#### Load Modules
module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi
echo "Current date and time: $(date)"

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/real_data/eda/fit_july2024_spatial_nugget_tiles.py"
DATA_PATH="/home/jl2815/tco/data/pickle_2024/tco_grid_24_07.pkl"
OUTDIR="/home/jl2815/tco/exercise_output/eda/july2024_nugget_3x3_compact"

mkdir -p "${OUTDIR}"
export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

common_args=(
    --input "${DATA_PATH}"
    --output-dir "${OUTDIR}"
    --month 2024-07
    --expected-hours 248
    --time-col hour
    --x-col Source_Longitude
    --y-col Source_Latitude
    --value-col ColumnAmountO3
    --coords raw
    --smooth 0.5
    --neighbors 10
    --max-points 0
    --min-tile-points 80
    --tiles 3
    --n-restarts 5
    --device cuda
)

# Step 1: manifest
python "${SCRIPT}" --mode manifest "${common_args[@]}"

# Step 2: fit each hour (writes per-hour files to hourly/)
MANIFEST="${OUTDIR}/manifest_hours.csv"
N_HOURS=$(tail -n +2 "${MANIFEST}" | wc -l)
echo "Fitting ${N_HOURS} hours..."
for i in $(seq 0 $((N_HOURS - 1))); do
    python "${SCRIPT}" --mode fit --array-index "${i}" "${common_args[@]}"
done

# Step 3: summarize (full diagnostics: day×hour heatmaps, tile boxplot, timeseries)
python "${SCRIPT}" --mode summarize "${common_args[@]}"

echo "Current date and time: $(date)"
```

```bash
sbatch fit_july2024_nugget3x3_050926.sh
```

---

### Expected outputs

```bash
ls /home/jl2815/tco/exercise_output/eda/july2024_nugget_3x3_compact
```

Key files:

| file | meaning |
|---|---|
| `hourly/*_global.csv` | per-hour global fit: sigmasq/range/nugget/nll |
| `hourly/*_tiles.csv` | per-hour 3x3 tile nuggets |
| `summary/global_results_all.csv` | all hours concatenated |
| `summary/global_params_by_day.csv` | daily mean sigmasq/range/nugget |
| `summary/global_params_by_hour_slot.csv` | hour-slot mean sigmasq/range/nugget across 31 days |
| `summary/global_params_timeseries.png` | sigma/range/nugget time series |
| `summary/global_params_daily_mean.png` | daily mean parameters with ±1 sd band |
| `summary/global_params_hour_slot_mean.png` | hour-slot mean parameters with error bars |
| `summary/global_{param}_day_hour_heatmap.png` | day × hour slot heatmap per parameter |
| `summary/tile_nugget_summary_3x3.csv` | 9-row tile nugget mean/median/sd |
| `summary/tile_nugget_mean_heatmap_3x3.png` | 3x3 mean tile nugget heatmap |
| `summary/tile_nugget_median_heatmap_3x3.png` | 3x3 median tile nugget heatmap |
| `summary/tile_to_global_nugget_ratio_mean_heatmap_3x3.png` | tile/global nugget ratio heatmap |
| `summary/tile_nugget_boxplot_3x3.png` | hourly distribution per tile |
