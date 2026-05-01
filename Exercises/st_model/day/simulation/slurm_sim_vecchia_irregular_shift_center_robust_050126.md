### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files (mac -> Amarel)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_shift_center_robust_050126.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_shift_center_robust_050126.sh" jl2815@amarel.rutgers.edu:/home/jl2815/tco/jobscript/tco/gp_exercise/
```

### Transfer results (Amarel -> mac)
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_shift_center_robust_true*_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_shift_center_robust_true*_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_shift_center_robust_true*_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"
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

### Simulation: Vecchia irregular shifted-center robustness

Real-data-like Vecchia-only simulation:

high-resolution FFT field -> real GEMS source observation locations -> irregular Vecchia fit.

This experiment tests whether lagged conditioning sets should reuse the current-time neighbor locations or instead choose fresh nearest neighbors around predicted upstream centers.

Standard controls:

- `Irr_Full_A20_B20_C20`: original full lag budget.
- `Irr_Cand_A20_B18_C15`: current best local-only reduced baseline.
- `Irr_Ratio_A20_B16_C10`: aggressive local-only reduction control.

Shift-center candidates:

- `A=20`, `B/C in {16/10, 14/8, 18/15}`.
- predicted lag-1 longitude offsets in `{0.10, 0.126, 0.16, 0.20, 0.25}`.
- predicted lag-2 offset is `2 * lag1 offset`.
- same-location temporal anchor is retained; the remaining lagged NN list is freshly selected around the shifted center.

SLURM array changes the true simulated longitudinal advection:

- task 0: `true_advec_lon = -0.10`
- task 1: `true_advec_lon = -0.16`
- task 2: `true_advec_lon = -0.25`

This separates the oracle case from robustness under misspecified advection offsets.

```bash
cd ./jobscript/tco/gp_exercise
nano sim_vecchia_irregular_shift_center_robust_050126.sh
sbatch sim_vecchia_irregular_shift_center_robust_050126.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=vec_irr_shiftctr_050126
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_irr_shiftctr_050126_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_irr_shiftctr_050126_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

TRUE_ADVECS=(-0.10 -0.16 -0.25)
TRUE_ADVEC=${TRUE_ADVECS[$SLURM_ARRAY_TASK_ID]}
TRUE_LABEL=${TRUE_ADVEC/-/m}
TRUE_LABEL=${TRUE_LABEL/./p}

echo "Running on: $(hostname)"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID}"
echo "True advec_lon: ${TRUE_ADVEC}"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_shift_center_robust_050126.py \
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
    --true-advec-lat 0.25 \
    --true-advec-lon "${TRUE_ADVEC}" \
    --shift-offsets "0.10,0.126,0.16,0.20,0.25" \
    --shift-budgets "16:10,14:8,18:15" \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --seed 42 \
    --out-prefix "sim_vecchia_irregular_shift_center_robust_true${TRUE_LABEL}"

echo "Current date and time: $(date)"
```
