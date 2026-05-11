# July 2022-2025 spatial nugget 3x3 EDA, nu=0.3

### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/real_data/eda"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/eda/fit_july_spatial_nugget_tiles.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/real_data/eda/
```

### Transfer data (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/data/pickle_2022 /home/jl2815/tco/data/pickle_2023 /home/jl2815/tco/data/pickle_2024 /home/jl2815/tco/data/pickle_2025"

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_22_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_23_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_25_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
```


### Transfer results (Amarel -> mac)
```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda"

SMOOTH_TAG=0p3
for YEAR in 2022 2023 2024 2025; do
    scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/${YEAR}_july_summary_nu${SMOOTH_TAG} \
        "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/"
done
```

---

### July 2022-2025 nugget 3x3 GPU, nu=0.3 (sbatch)

```bash
cd ./jobscript/tco/gp_exercise
nano fit_july2022_2025_nugget3x3_nu0p3_050926.sh
sbatch fit_july2022_2025_nugget3x3_nu0p3_050926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=nug3x3_nu0p3
#SBATCH --output=/home/jl2815/tco/exercise_output/nug3x3_nu0p3_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/nug3x3_nu0p3_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi
echo "Current date and time: $(date)"

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/real_data/eda/fit_july_spatial_nugget_tiles.py"
BASE_OUT="/home/jl2815/tco/exercise_output/eda"
SMOOTH=0.3
SMOOTH_TAG="${SMOOTH/./p}"

for YEAR in 2022 2023 2024 2025; do
    YY="${YEAR:2:2}"
    DATA_PATH="/home/jl2815/tco/data/pickle_${YEAR}/tco_grid_${YY}_07.pkl"
    OUTDIR="${BASE_OUT}/${YEAR}_july_summary_nu${SMOOTH_TAG}"
    case "${YEAR}" in
        2022) EXPECTED_HOURS=240 ;;
        2023) EXPECTED_HOURS=248 ;;
        2024) EXPECTED_HOURS=248 ;;
        2025) EXPECTED_HOURS=247 ;;
    esac
    mkdir -p "${OUTDIR}"
    export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
    mkdir -p "${MPLCONFIGDIR}"

    echo ""
    echo "================================================================================"
    echo "Running July ${YEAR} -> ${OUTDIR}"
    echo "================================================================================"

    python "${SCRIPT}" --mode all \
        --input "${DATA_PATH}" \
        --output-dir "${OUTDIR}" \
        --month "${YEAR}-07" \
        --expected-hours "${EXPECTED_HOURS}" \
        --time-col hour \
        --x-col Source_Longitude \
        --y-col Source_Latitude \
        --value-col ColumnAmountO3 \
        --coords raw \
        --smooth "${SMOOTH}" \
        --neighbors 8 \
        --max-points 0 \
        --min-tile-points 80 \
        --tiles 3 \
        --n-restarts 5 \
        --device cuda \
        --summary-every 8
done

echo "Current date and time: $(date)"
```

### Output folders

Each year is written to a separate smooth-labelled folder:

```bash
/home/jl2815/tco/exercise_output/eda/2022_july_summary_nu0p3
/home/jl2815/tco/exercise_output/eda/2023_july_summary_nu0p3
/home/jl2815/tco/exercise_output/eda/2024_july_summary_nu0p3
/home/jl2815/tco/exercise_output/eda/2025_july_summary_nu0p3
```

Key files in each folder:

| file | meaning |
|---|---|
| `julyYYYY_spatial_fit_<observed_hours>.csv` | one row per fitted hour: day/hour/sigmasq/sigma/range/nugget/loss |
| `running_summary.txt` | compact text summary also printed in `.out` |
| `tile_nugget_mean_3x3.csv` | 9-row 3x3 regional nugget table |
| `tile_nugget_mean_heatmap_3x3.png` | 3x3 regional mean nugget heatmap |
| `tile_nugget_median_heatmap_3x3.png` | 3x3 regional median nugget heatmap |
| `global_params_by_day.csv` | daily mean/median/sd global parameters |
| `global_params_by_hour_slot.csv` | hour-slot global parameter summary |
| `global_params_timeseries.png` | hourly global sigma/range/nugget time series |
| `global_params_daily_mean.png` | daily mean sigma/range/nugget with sd band |
| `global_params_hour_slot_mean.png` | hour-slot mean sigma/range/nugget with error bars |
| `global_sigma_day_hour_heatmap.png` | sigma day x hour-slot heatmap |
| `global_range_day_hour_heatmap.png` | range day x hour-slot heatmap |
| `global_nugget_day_hour_heatmap.png` | nugget day x hour-slot heatmap |
