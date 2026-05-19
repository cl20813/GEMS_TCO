# Reusable July ST circulant simulation assets, 2022-2025, smooth=0.3 DGP

Files live under:

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data
```

This job generates a second set of July space-time simulation pickles for a
smoothness-misspecification experiment.  The only intended DGP change from the
baseline generator is:

```text
Matérn smoothness nu = 0.3
Matérn correlation evaluated by natural cubic spline
```

The output root is separate from the baseline smooth=0.5 assets:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3
```

Truth:

```text
smooth     = 0.3
sigmasq    = 10
range_lat  = 0.2
range_lon  = 0.3
range_time = 2.0
advec_lat  = 0.08
advec_lon  = -0.2
nugget     = 1
```

Spline settings:

```text
spline_n_points = 4000
spline_r_max    = 30.0
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

### Transfer generator and Slurm script (mac -> Amarel)
```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/simulate_data"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/simulate_data/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.sh" \
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

### Generate all years (sbatch)

```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/simulate_data
nano generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.sh
sbatch generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.sh
```

The submitted Slurm script should be:

```bash
#!/bin/bash
#SBATCH --job-name=sim_july_st_s03_051926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_july_st_s03_051926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_july_st_s03_051926_%j.err
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=main

set -euo pipefail

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926.py"
OUTDIR="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3"

echo "Running on: $(hostname)"
echo "Python: ${PYTHON}"
echo "Script: ${SCRIPT}"
echo "Output directory: ${OUTDIR}"
echo "Current date and time: $(date)"

srun "${PYTHON}" - <<'PY'
import sys
import numpy
import pandas
import scipy
import torch

print("preflight python:", sys.executable, flush=True)
print("preflight numpy:", numpy.__version__, flush=True)
print("preflight pandas:", pandas.__version__, flush=True)
print("preflight scipy:", scipy.__version__, flush=True)
print("preflight torch:", torch.__version__, flush=True)
PY

srun "${PYTHON}" "${SCRIPT}" \
    --years "2022,2023,2024,2025" \
    --input-root "/home/jl2815/tco/data" \
    --output-dir "${OUTDIR}" \
    --max-hours 248 \
    --hours-per-day 8 \
    --seed 20240701 \
    --smooth 0.3 \
    --spline-n-points 4000 \
    --spline-r-max 30.0 \
    --sigmasq 10.0 \
    --range-lat 0.2 \
    --range-lon 0.3 \
    --range-time 2.0 \
    --advec-lat 0.08 \
    --advec-lon -0.2 \
    --nugget 1.0 \
    --mean-intercept 260.0 \
    --mean-lat-slope 1.0 \
    --mean-lat-center -0.5 \
    --lat-range="-3,2" \
    --lon-range="121,131" \
    --lat-factor-hr 100 \
    --lon-factor-hr 10 \
    --hr-pad 0.1

echo "Current date and time: $(date)"
```

### Monitor

```bash
squeue -u jl2815

tail -f /home/jl2815/tco/exercise_output/sim_july_st_s03_051926_<JOBID>.out
tail -f /home/jl2815/tco/exercise_output/sim_july_st_s03_051926_<JOBID>.err
```

### Transfer generated assets (Amarel -> mac)
```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_DATA/simulation"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3 \
    "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
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
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2022_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2023_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2024_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3/2025_july_st_circulant
```

The truth JSON in each year folder should include:

```json
{
  "smooth": 0.3,
  "smooth_generation_method": "natural cubic spline Matérn correlation",
  "spline_n_points": 4000,
  "spline_r_max": 30.0
}
```
