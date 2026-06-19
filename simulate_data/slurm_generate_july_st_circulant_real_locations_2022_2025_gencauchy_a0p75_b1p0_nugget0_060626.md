# Reusable July ST circulant simulation assets, 2022-2025, generalized Cauchy a=0.75, b=1.0, nugget=0

Files live under:

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data
```

This job generates July space-time simulation pickles where the
data-generating process has:

```text
correlation model = generalized Cauchy
C(r)              = (1 + r^a)^(-b/a)
a                 = 0.75
b                 = 1.0
nugget            = 0
```

Here `r` is the same anisotropic advective space-time distance used by the
existing circulant generator:

```text
r = sqrt(
  ((lat - advec_lat * time) / range_lat)^2
  + ((lon - advec_lon * time) / range_lon)^2
  + (time / range_time)^2
)
```

The output root is separate from the earlier Matern smooth=0.3, nugget=0
assets, so it will not overwrite those files:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0
```

Truth:

```text
cauchy_a   = 0.75
cauchy_b   = 1.0
sigmasq    = 10
range_lat  = 0.2
range_lon  = 0.3
range_time = 2.0
advec_lat  = 0.08
advec_lon  = -0.2
nugget     = 0
```

High-resolution grid and griddification:

```text
dlat_hr = 0.044 / 100
dlon_hr = 0.063 / 10
hr_pad  = 0.1

griddification accepts a source -> regular-grid assignment only if:
abs(source_lat - grid_lat) <= 0.044 / 2
abs(source_lon - grid_lon) <= 0.063 / 2
```

The generator writes one folder per year. Current local July pickle counts are
2022=240 hours, 2023=248, 2024=248, 2025=247, so the script uses available
hours up to 248 rather than inventing missing keys.

At high resolution, `lat x100` and `lon x10` make a full 248-hour 3D FFT too
large for routine testing. The generator therefore uses independent daily
8-hour 3D circulant-embedding blocks.

### Transfer generator (mac -> Amarel)

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/simulate_data /home/jl2815/tco/exercise_output/logs"

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/generate_july_st_circulant_real_locations_2022_2025_gencauchy_a0p75_b1p0_nugget0_060626.py" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/simulate_data/"
```

### Transfer real-location templates (mac -> Amarel)

Run this only if the July template pickles are not already on Amarel.

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
nano generate_july_st_circulant_real_locations_2022_2025_gencauchy_a0p75_b1p0_nugget0_060626.sh
sbatch generate_july_st_circulant_real_locations_2022_2025_gencauchy_a0p75_b1p0_nugget0_060626.sh
```

The submitted Slurm script should be:

```bash
#!/bin/bash
#SBATCH --job-name=sim_july_st_gc075b10_n0_060626
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/sim_july_st_gc075b10_n0_060626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/sim_july_st_gc075b10_n0_060626_%j.err
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="/home/jl2815/tco/simulate_data/generate_july_st_circulant_real_locations_2022_2025_gencauchy_a0p75_b1p0_nugget0_060626.py"
OUTDIR="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0"

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
print("preflight cuda available:", torch.cuda.is_available(), flush=True)
PY

srun "${PYTHON}" "${SCRIPT}" \
  --years "2022,2023,2024,2025" \
  --input-root "/home/jl2815/tco/data" \
  --output-dir "${OUTDIR}" \
  --max-hours 248 \
  --hours-per-day 8 \
  --seed 20240701 \
  --cauchy-a 0.75 \
  --cauchy-b 1.0 \
  --sigmasq 10.0 \
  --range-lat 0.2 \
  --range-lon 0.3 \
  --range-time 2.0 \
  --advec-lat 0.08 \
  --advec-lon -0.2 \
  --nugget 0.0 \
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

tail -f /home/jl2815/tco/exercise_output/logs/sim_july_st_gc075b10_n0_060626_<JOBID>.out
tail -f /home/jl2815/tco/exercise_output/logs/sim_july_st_gc075b10_n0_060626_<JOBID>.err
```

### Transfer generated assets (Amarel -> mac)

Use this when the generalized-Cauchy assets already exist on Amarel and you
only want to copy them back to the local computer.

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_DATA/simulation"

rsync -avh --progress \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0" \
  "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
```

Equivalent `scp` fallback:

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_DATA/simulation"

scp -r "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0" \
  "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
```

After transfer, the local root should be:

```bash
/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0
```

Quick local check:

```bash
find "/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0" \
  -maxdepth 2 -type f \( -name "*truth.json" -o -name "*real_locations.pkl" -o -name "*gridded.pkl" \) \
  | sort
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
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0/2022_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0/2023_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0/2024_july_st_circulant
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_gencauchy_a0p75_b1p0_nugget0/2025_july_st_circulant
```

The truth JSON in each year folder should include:

```json
{
  "correlation_model": "generalized_cauchy",
  "correlation_formula": "(1 + r^a)^(-b/a), where r is the anisotropic advective space-time distance",
  "cauchy_a": 0.75,
  "cauchy_b": 1.0,
  "sigmasq": 10.0,
  "range_lat": 0.2,
  "range_lon": 0.3,
  "range_time": 2.0,
  "advec_lat": 0.08,
  "advec_lon": -0.2,
  "nugget": 0.0
}
```
