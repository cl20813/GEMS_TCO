### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac -> Amarel)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_irregular_lag_ratio_043026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_dw2110_irregular_lag_ratio_043026.sh" jl2815@amarel.rutgers.edu:/home/jl2815/tco/jobscript/tco/gp_exercise/
```

### Transfer results (Amarel -> mac)
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_*_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_candidates_*_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_*_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_candidates_*_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_*_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_irregular_lag_ratio_candidates_*_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"
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

### Simulation: irregular Vecchia lag-budget candidates

Real-data-like simulation:

high-resolution FFT field -> real GEMS source observation locations -> irregular Vecchia fit.

Models fit on the same simulated field and same initial values at every iteration:

- `Irr_Full_A20_B20_C20`: original `20/20/20`.
- `Irr_Ratio_A20_B16_C10`: same current budget, `t-1 ~= 0.8x`, `t-2 = 0.5x`.
- `Irr_Cand_A20_B18_C15`: `t-1 = 0.9x`, `t-2 = 0.75x`.
- `Irr_Cand_A20_B18_C12`: `t-1 = 0.9x`, `t-2 = 0.6x`.
- `Irr_Cand_A20_B20_C10`: full `t-1`, `t-2 = 0.5x`.
- `Irr_Cand_A20_B16_C20`: `t-1 = 0.8x`, full `t-2`.

The total conditioning sizes are intentionally not matched.  The goal is to see whether advection recovery mainly needs full `t-1`, full `t-2`, or only a less aggressive reduction.

```bash
cd ./jobscript/tco/gp_exercise
nano sim_dw2110_irregular_lag_ratio_043026.sh
sbatch sim_dw2110_irregular_lag_ratio_043026.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_irr_ratio_043026
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_irr_ratio_043026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_irr_ratio_043026_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_irregular_lag_ratio_043026.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --base-limit-a 20 \
    --base-limit-b 20 \
    --base-limit-c 20 \
    --ratio-limit-a 20 \
    --ratio-limit-b 16 \
    --ratio-limit-c 10 \
    --daily-stride 2 \
    --num-iters 300 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --seed 42 \
    --out-prefix sim_dw2110_irregular_lag_ratio_candidates

echo "Current date and time: $(date)"
```
