# July 2025 spatial nugget 3x3 EDA

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

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/2025_july_summary \
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
    --input /home/jl2815/tco/data/pickle_2025/tco_grid_25_07.pkl \
    --output-dir /home/jl2815/tco/exercise_output/eda/2025_july_summary \
    --month 2025-07 \
    --expected-hours 248 \
    --time-col hour \
    --x-col Source_Longitude \
    --y-col Source_Latitude \
    --value-col ColumnAmountO3
```

---

### July 2025 nugget 3x3 single-node (sbatch)

```bash
cd ./jobscript/tco/gp_exercise
nano fit_july2025_nugget3x3_050926.sh
sbatch fit_july2025_nugget3x3_050926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=nug25_3x3
#SBATCH --output=/home/jl2815/tco/exercise_output/nug25_3x3_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/nug25_3x3_%j.err
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
DATA_PATH="/home/jl2815/tco/data/pickle_2025/tco_grid_25_07.pkl"
OUTDIR="/home/jl2815/tco/exercise_output/eda/2025_july_summary"

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
    --month 2025-07
    --expected-hours 248
    --time-col hour
    --x-col Source_Longitude
    --y-col Source_Latitude
    --value-col ColumnAmountO3
    --coords raw
    --smooth 0.5
    --neighbors 8
    --max-points 0
    --min-tile-points 80
    --tiles 3
    --n-restarts 5
    --device cuda
)

python "${SCRIPT}" --mode all "${common_args[@]}"

echo "Current date and time: $(date)"
```

```bash
sbatch fit_july2025_nugget3x3_050926.sh
```

---

### Expected outputs

```bash
ls /home/jl2815/tco/exercise_output/eda/2025_july_summary
```

Key files:

| file | meaning |
|---|---|
| `july2025_spatial_fit_248.csv` | one row per fitted hour: day/hour/n_raw/n_used/sigmasq/sigma/range/nugget/loss |
| `running_summary.txt` | compact text summary also printed in the `.out` log |
| `tile_nugget_mean_3x3.csv` | 9-row 3x3 regional nugget mean table |
| `tile_nugget_mean_heatmap_3x3.png` | 3x3 regional mean nugget heatmap |
| `tile_nugget_median_heatmap_3x3.png` | 3x3 regional median nugget heatmap |
| `global_params_by_day.csv` | daily mean/median/sd global parameters |
| `global_params_by_hour_slot.csv` | hour-slot mean sigmasq/range/nugget/loss across days |
| `global_nugget_by_hour_slot.png` | hour-slot mean global nugget plot |
| `global_params_timeseries.png` | hourly global sigma/range/nugget time series |
| `global_params_daily_mean.png` | daily mean sigma/range/nugget with sd band |
| `global_params_hour_slot_mean.png` | hour-slot mean sigma/range/nugget with error bars |
| `global_sigma_day_hour_heatmap.png` | sigma day x hour-slot heatmap |
| `global_range_day_hour_heatmap.png` | range day x hour-slot heatmap |
| `global_nugget_day_hour_heatmap.png` | nugget day x hour-slot heatmap |
