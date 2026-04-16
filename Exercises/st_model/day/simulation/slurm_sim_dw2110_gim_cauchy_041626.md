### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_gim_cauchy_041626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design
- **Models**: DW, Vecchia-Irr, Cauchy-Vecchia (β=1.0)
- **J method**: Observed information (no parametric bootstrap)
  - DW: per-frequency score outer products — `jac.T @ jac / n_freq²`
  - VC / CY: per-unit score outer products — `jac.T @ jac / N_units²`
- **Input estimates**:
  - DW  : `real_dw_july_22_23_24_25.csv`
  - VC  : `real_vecc_july_22_23_24_25_mm20.csv`
  - CY  : `real_cauchy_b10_july_22_23_24_25_mm20.csv`
- **Skip policy**: a day is skipped if **any** of the three models has no estimate for that day

---

### Connect & setup

```
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

---

### Real data GIM — all days 2022–2025 (sbatch)
Loops over all July days (1–28) for years 2022, 2023, 2024, 2025.
Outputs:
  - `GIM_cauchy_all_july_22_23_24_25_obsJ.csv`   — one row per day
  - `GIM_cauchy_summary_july_22_23_24_25_obsJ.csv` — mean GIM SE per param

~112 day-jobs × ~20 min/day ≈ 37 h; set time=72h for safety.

```
cd ./jobscript/tco/gp_exercise
nano sim_dw2110_gim_cauchy_041626.sh
sbatch sim_dw2110_gim_cauchy_041626.sh

```

```bash

#!/bin/bash
#SBATCH --job-name=gim_cauchy_032826
#SBATCH --output=/home/jl2815/tco/exercise_output/gim_cauchy_032826_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/gim_cauchy_032826_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu035

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_gim_cauchy_041626.py \
    --years "2022,2023,2024,2025" \
    --days "1,28" \
    --month 7 \
    --v 0.5 \
    --gc-beta 1.0 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2

echo "Current date and time: $(date)"

```

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/GIM/GIM_cauchy_all_july_22_23_24_25_obsJ.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/GIM/GIM_cauchy_summary_july_22_23_24_25_obsJ.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```
