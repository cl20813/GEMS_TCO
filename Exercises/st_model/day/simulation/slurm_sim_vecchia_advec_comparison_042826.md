### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac -> Amarel)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_advec_comparison_042826.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel -> mac)
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_advec_comparison_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_advec_summary_*.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
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

### Vecchia standard vs advection-aware comparison

Direct target-grid simulation:

0.044 x 0.063 regular grid -> simulated GP field -> Vecc_Std / Vecc_Advec fit -> RMSRE + Vecchia likelihood loss comparison.

No DW and no high-resolution observation remapping.

Fairness control:

Vecc_Std uses same-location-centered temporal neighborhoods with one extra standard temporal neighbor per lag (`limit_B + 1`, `limit_C + 1`).

Vecc_Advec keeps the same lag-wise conditioning-set size, but at `t-1` and `t-2` it uses an upstream-centered temporal neighborhood: same location + advection-shifted center + local neighbors around that shifted center.  So both models keep comparable total and lag-wise budgets; the temporal-neighborhood center differs.

```bash
cd ./jobscript/tco/gp_exercise
nano sim_vecchia_advec_comparison_042826.sh
sbatch sim_vecchia_advec_comparison_042826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_vadvec_042826
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_vadvec_042826_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_vadvec_042826_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_advec_comparison_042826.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
    --daily-stride 2 \
    --num-iters 100 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --advec-lon-offset 0.126 \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --seed 42

echo "Current date and time: $(date)"
```

`--advec-lon-offset 0.126` is `0.063 * 2`, so Vecc_Advec centers the `t-1` temporal neighborhood near `lon + 0.126` and the `t-2` temporal neighborhood near `lon + 0.252` when `--daily-stride 2`.
