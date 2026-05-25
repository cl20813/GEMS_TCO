# Reusable July ST circulant simulation assets, 2022-2025

Files live under:

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data
```

This job generates simulated data once, then the same pickles can be reused by
pure-space, space-time, gridded, and real-location Vecchia tests.

Truth:

```text
sigmasq    = 10
range_lat = 0.2
range_lon = 0.3
range_time = 2.0
advec_lat = 0.08
advec_lon = -0.2
nugget    = 1
```

High-resolution grid and griddification:

```text
dlat_hr = 0.044 / 100
dlon_hr = 0.063 / 10

griddification accepts a source -> regular-grid assignment only if:
abs(source_lat - grid_lat) <= 0.044 / 2
abs(source_lon - grid_lon) <= 0.063 / 2
```

The generator writes one folder per year.  Current local July pickle counts are
2022=240 hours, 2023=248, 2024=248, 2025=247, so the script uses available
hours up to 248 rather than inventing missing keys.

At high resolution, `lat x100` and `lon x10` make a full 248-hour 3D FFT too
large for routine testing.  The generator therefore uses independent daily
8-hour 3D circulant-embedding blocks.  With `range_time=2.0`, overnight
cross-day correlation is negligible for the current daily GEMS ST/pure-space
tests.

### Transfer generator (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/simulate_data"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/simulate_data/
```

### Transfer real-location templates (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/data/pickle_2022 /home/jl2815/tco/data/pickle_2023 /home/jl2815/tco/data/pickle_2024 /home/jl2815/tco/data/pickle_2025"

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_22_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_23_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_25_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
```

### Transfer generated assets (Amarel -> mac)
```bash
LOCAL_SIM_ROOT="/Users/joonwonlee/Documents/GEMS_DATA/simulation"
AMAREL_SIM_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"

mkdir -p "${LOCAL_SIM_ROOT}"

# Preferred: resumable copy from Amarel to local mac.
rsync -avh --progress \
    jl2815@amarel.rutgers.edu:${AMAREL_SIM_ROOT} \
    "${LOCAL_SIM_ROOT}/"

# Fallback if rsync is unavailable.
scp -r jl2815@amarel.rutgers.edu:${AMAREL_SIM_ROOT} \
    "${LOCAL_SIM_ROOT}/"

# Quick local check.
find "${LOCAL_SIM_ROOT}/july_st_circulant_realpattern" -maxdepth 2 -type f \
    \( -name "*truth.json" -o -name "*real_locations.pkl" -o -name "*gridded.pkl" \) | sort
```

---

### Generate all years (sbatch)

```bash
cd ./jobscript/tco/gp_exercise
nano generate_july_st_circulant_real_locations_2022_2025_051026.sh
sbatch generate_july_st_circulant_real_locations_2022_2025_051026.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_july_st
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_july_st_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_july_st_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=main

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py"
OUTDIR="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"

mkdir -p "${OUTDIR}"

echo "Running on: $(hostname)"
echo "Current date and time: $(date)"

srun python "${SCRIPT}" \
    --years "2022,2023,2024,2025" \
    --input-root /home/jl2815/tco/data \
    --output-dir "${OUTDIR}" \
    --max-hours 248 \
    --hours-per-day 8 \
    --seed 20240701 \
    --smooth 0.5 \
    --sigmasq 10 \
    --range-lat 0.2 \
    --range-lon 0.3 \
    --range-time 2.0 \
    --advec-lat 0.08 \
    --advec-lon=-0.2 \
    --nugget 1 \
    --mean-intercept 260 \
    --mean-lat-slope 1 \
    --mean-lat-center=-0.5 \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --lat-factor-hr 100 \
    --lon-factor-hr 10 \
    --hr-pad 0.1

echo "Current date and time: $(date)"
```

### Expected outputs

Each year folder contains:

```bash
sim_julyYYYY_st_circulant_real_locations.pkl
sim_julyYYYY_st_circulant_gridded.pkl
sim_julyYYYY_st_circulant_manifest.csv
sim_julyYYYY_st_circulant_griddification_diag.csv
sim_julyYYYY_st_circulant_embedding_diag.csv
sim_julyYYYY_st_circulant_truth.json
```

Folder layout:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2022_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2023_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2024_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2025_july_st_circulant
```
