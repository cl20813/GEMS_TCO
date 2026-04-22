### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw2110_no_threshold_042026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
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

### No-threshold missing sim (sbatch)

Single-condition test: no distance threshold.
Every grid cell matched to nearest obs (obs→cell, 1:1 dedup).
Missing only from deduplication — expected to be small fraction.

Three models: Vecc_Irr | Vecc_Reg | DW (debiased_whittle_2110)
True params: July 2024 fit values.

```
cd ./jobscript/tco/gp_exercise
nano sim_dw2110_no_threshold_042026.sh
sbatch sim_dw2110_no_threshold_042026.sh
```

#### Option A — mem partition (recommended)
```bash
#!/bin/bash
#SBATCH --job-name=dw_no_thr_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_no_thr_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_no_thr_042026_%j.err
#SBATCH --time=6:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --partition=mem

#### Load Modules
module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

#### Initialize conda
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_no_threshold_042026.py \
    --num-iters 300 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```


cd ./jobscript/tco/gp_exercise
nano sim_dw2110_no_threshold_042026.sh
sbatch sim_dw2110_no_threshold_042026.sh

#### Option B — gpu-redhat partition
```bash
#!/bin/bash
#SBATCH --job-name=dw_no_thr_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_no_thr_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_no_thr_042026_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw2110_no_threshold_042026.py \
    --num-iters 60 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```

---

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw2110_no_thr_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```
