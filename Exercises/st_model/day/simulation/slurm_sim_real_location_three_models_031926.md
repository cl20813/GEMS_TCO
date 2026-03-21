### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_three_model_comparison_031926.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel → mac)
```
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_three_model_comparison_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_three_model_summary_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
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

### Three-model simulation — real obs locations (sbatch)
FFT field → irregular (actual GEMS obs) + regular (step3 1:1) → Vecchia-Irr / Vecchia-Reg / DW fit → RMSRE in original param space, repeated num-iters times

```
cd ./jobscript/tco/gp_exercise
nano sim_three_models_031926.sh
sbatch sim_three_models_031926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_three_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_three_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_three_031926_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_three_model_comparison_031926.py \
    --v 0.5 \
    --mm-cond-number 52 \
    --nheads 800 \
    --limit-a 16 \
    --limit-b 16 \
    --limit-c 16 \
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

## 참고로 init noise 0.7dms exp(0.7)  exp(1) = 2.7배
0.7 정도면 13 -> 6.5에서 26 사이  advec lat 0.022 +- 0.044
