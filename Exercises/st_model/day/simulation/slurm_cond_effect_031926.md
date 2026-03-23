### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_cond_effect_031926.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel → mac)
```
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_cond_effect_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_cond_summary_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
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

### Conditioning number effect — Vecchia-Irregular (sbatch)
nheads=0 fixed; COND_LIST = [6, 8, 12, 20] (limit_A=limit_B=limit_C=n_cond)
MM_COND_MAX = 100 (consistent with heads_effect and real data fitting)
Goal: show that increasing conditioning neighbors alone does NOT improve temporal parameter estimation
→ per-parameter RMSRE + mean estimates + wall-clock time per n_cond config

```
cd ./jobscript/tco/gp_exercise
nano sim_cond_effect_031926.sh
sbatch sim_cond_effect_031926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_cond_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_cond_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_cond_031926_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu039

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_cond_effect_031926.py \
    --v 0.5 \
    --cond-list "6,8,12,20" \
    --nheads 0 \
    --mm-cond-max 100 \
    --daily-stride 2 \
    --num-iters 200 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```
