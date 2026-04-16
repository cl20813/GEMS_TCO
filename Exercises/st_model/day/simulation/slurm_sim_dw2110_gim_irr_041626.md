### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_gim_irr_041626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model




scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_gim_irr_041626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Observed J (no bootstrap)

J is computed from the observed data directly (Varin, Reid & Firth 2011):
- DW:     per-frequency score outer products — `jac.T @ jac / n_freq²`
- Vecchia: per-unit score outer products     — `jac.T @ jac / N_units²`

This replaces parametric bootstrap. No `--num-sims`, `--lat-factor`, `--lon-factor` args.

### Transfer results (Amarel → mac)  [see also updated section below]
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/GIM/*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```

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
  - `GIM_all_july_22_23_24_25_obsJ.csv`  — one row per day
  - `GIM_summary_july_22_23_24_25_obsJ.csv` — RMSRE/MdARE/P90-P10 + mean GIM SE per param

~112 day-jobs × ~15 min/day ≈ 28 h; set time=72h for safety.

```
cd ./jobscript/tco/gp_exercise
nano sim_GIM_real_031926.sh
sbatch sim_GIM_real_031926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_GIM_real_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_GIM_real_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_GIM_real_031926_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu037

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_gim_irr_041626.py \
    --years "2022,2023, 2024,2025" \
    --days "1,28" \
    --month 7 \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 1000 \
    --limit-a 16 \
    --limit-b 16 \
    --limit-c 16 \
    --daily-stride 2

echo "Current date and time: $(date)"


```

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/GIM/GIM_all_july_22_23_24_25_obsJ.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/GIM/GIM_summary_july_22_23_24_25_obsJ.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```

---

### Heads comparison GIM (sbatch)
Simulated data → fit Vecchia with varying nheads → GIM SE per heads config
(DW computed once as reference; orthodox GIM: H from sim obs data, J from FFT bootstrap)

```
cd ./jobscript/tco/gp_exercise
nano sim_heads_GIM_031926.sh
sbatch sim_heads_GIM_031926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_heads_GIM_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_heads_GIM_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_heads_GIM_031926_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu037

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_gim_irr_041626.py \
    --v 0.5 \
    --lr 1.0 \
    --mm-cond-number 100 \
    --epochs 20 \
    --head-configs "300,500,1000" \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
    --daily-stride 8 \
    --num-sims 100

echo "Current date and time: $(date)"
```
