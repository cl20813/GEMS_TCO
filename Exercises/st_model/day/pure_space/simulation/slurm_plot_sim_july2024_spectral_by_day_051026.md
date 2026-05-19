# Simulation Spectral Plots: July 2024, Daily, Smooth 0.3 / 0.5

This runbook uses the reusable 3D circulant-embedding simulation asset created by
`generate_july_st_circulant_real_locations_2022_2025.py`.

For spectrum plots, use the **gridded** simulation pickle, not the real-location
pickle. The radial FFT/Nyquist interpretation assumes a regular grid.

Default output is PNG plots only. The PNG includes the two full-fit rows and
the two partial-profile rows:

1. full: nugget fixed 0
2. full: nugget free
3. partial: sigma only / nugget0
4. partial: range only / nugget0

```text
/home/jl2815/tco/exercise_output/eda/simulation/spectral_2024_july/
  spectral_plot_line_meanings.txt
  nu0p3/
    sim_spectral_20240701_nu0p3.png
    ...
  nu0p5/
    sim_spectral_20240701_nu0p5.png
    ...
```

The fit uses per-hour pure-space isotropic Hybrid Vecchia with neighbor 8 and
GLS mean `base`, which is effectively intercept + centered latitude for these
one-hour tensors. The optimizer uses the same microergodic-style
reparameterization as the expanded-grid nugget runs:

```text
phi2 = 1 / range
phi1 = sigmasq * phi2
sigmasq = phi1 / phi2
range = 1 / phi2
```

For `smooth=0.3`, the Matérn correlation uses spline evaluation. For
`smooth=0.5`, it uses the closed-form exponential correlation.

### How to read the red and black lines

Each PNG panel is one day, one smooth value, one fitting variant, and one grid
thinning stride.

| line | meaning |
|---|---|
| thin gray lines | one radial residual periodogram per observed hour in that day |
| thick black line | daily mean of those hourly empirical residual spectra |
| dashed red line | fitted Matérn theoretical spectrum for the full-fit row, averaged across hours and vertically rescaled to the empirical spectrum |
| dashed blue/green lines | optional partial-profile theory curves: blue for sigma-only, green for range-only |
| dotted gray vertical line | approximate maximum radial frequency supported by the thinned data grid |

Important: the dashed theory line is **vertically median-rescaled** to the
empirical black spectrum before plotting. So the plot is mainly checking
whether the fitted Matérn model has the right frequency-shape/slope, not whether
the absolute spectral level matches exactly.

---

### Update packages (mac -> Amarel)

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer plot script (mac -> Amarel)

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation'

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/pure_space/simulation/plot_sim_july_spectral_by_day.py" \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation/
```

### Simulation asset location

This run expects the 2024 gridded simulation asset here:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/2024_july_st_circulant/sim_july2024_st_circulant_gridded.pkl
```

If it is missing, first run:

```text
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/simulate_data/slurm_generate_july_st_circulant_real_locations_2022_2025.md
```

---

### Connect & setup

```bash
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

---

### Daily spectral plots (single GPU job)

This uses one GPU job and runs all selected days/smooth values inside the
Python script. It is simpler than an array job and gives one Slurm output file.

```bash
cd ./jobscript/tco/gp_exercise
nano plot_sim24_spectral_by_day_051026.sh
sbatch plot_sim24_spectral_by_day_051026.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim24_spec
#SBATCH --output=/home/jl2815/tco/exercise_output/sim24_spec_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim24_spec_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu018


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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count()); print('CUDA_VISIBLE_DEVICES from python:', __import__('os').environ.get('CUDA_VISIBLE_DEVICES'))"

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEAR=2024
MONTH=7
DAY_RANGE="1,31"
SMOOTHS="0.3,0.5"

SCRIPT="/home/jl2815/tco/exercise_25/st_model/day/pure_space/simulation/plot_sim_july_spectral_by_day.py"
DATA_PATH="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern/${YEAR}_july_st_circulant/sim_july${YEAR}_st_circulant_gridded.pkl"
OUTROOT="/home/jl2815/tco/exercise_output/eda/simulation/spectral_${YEAR}_july"

mkdir -p "${OUTROOT}"
export MPLCONFIGDIR="${OUTROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "Running smooths=${SMOOTHS}, days=${DAY_RANGE}"

srun python "${SCRIPT}" \
    --input "${DATA_PATH}" \
    --output-root "${OUTROOT}" \
    --year "${YEAR}" \
    --month "${MONTH}" \
    --days "${DAY_RANGE}" \
    --smooths "${SMOOTHS}" \
    --strides "8,4,2,1" \
    --neighbors 8 \
    --mean-design base \
    --x-col Longitude \
    --y-col Latitude \
    --value-col ColumnAmountO3 \
    --device cuda \
    --target-chunk-size 1024 \
    --lbfgs-steps 8 \
    --lbfgs-eval 20 \
    --include-partials \
    --skip-existing

echo "Current date and time: $(date)"
```

To also run smooth 1.5, change this line:

```bash
SMOOTHS="0.3,0.5,1.5"
```

To save fit parameter CSVs, append:

```bash
    --save-fit-csv
```

---

### Transfer plots (Amarel -> mac)

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/spectral_2024_july"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/simulation/spectral_2024_july \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/simulation/"
```
