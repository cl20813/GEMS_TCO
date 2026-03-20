### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_param_recovery_veccDW_regular_031926.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_reg_dw_031926_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_reg_vecc_031926_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"
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

### Parameter recovery simulation — regular grid (sbatch)
FFT data generation → DW + Vecchia fit → RMSRE, repeated num-iters times

```
cd ./jobscript/tco/gp_exercise
nano sim_param_recovery_031926.sh
sbatch sim_param_recovery_031926.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_recovery_031926
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_recovery_031926_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_recovery_031926_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_param_recovery_veccDW_regular_031926.py \
    --v 0.5 \
    --mm-cond-number 16 \
    --nheads 1000 \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
    --daily-stride 8 \
    --num-iters 100

echo "Current date and time: $(date)"
```
