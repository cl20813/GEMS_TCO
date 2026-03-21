### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_heads_effect_031926.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel → mac)
```
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_heads_effect_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_heads_summary_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
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

### Heads effect simulation — Vecchia-Irregular (sbatch)
nheads = [0, 100, 200, 400, 800] × same field/obs pattern per iteration
→ per-parameter RMSRE + mean estimates + wall-clock time per heads config

```
cd ./jobscript/tco/gp_exercise
nano sim_heads_effect_031926.sh

```

```bash
#!/bin/bash
#SBATCH --job-name=sim_heads_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_heads_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_heads_031926_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu034

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_effect_031926.py \
    --v 0.5 \
    --mm-cond-number 16 \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
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
