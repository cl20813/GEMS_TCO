### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_advec_direction_042026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
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

### Advection-direction filter comparison (sbatch)

Tests whether differencing direction interacts with advection direction.
  Scenario A: advec_lat=0,   advec_lon=-0.2  (pure lon advection)
  Scenario B: advec_lat=0.2, advec_lon=0     (pure lat advection)

Four filters: Raw | Lat-1-1 | Lon-1-1 | 2-1-1-0
60 iters per scenario (120 total).

New DW modules required (must be in src/GEMS_TCO/):
  debiased_whittle_lat1.py   — Z=X(i+1,j)-X(i,j),   excludes w1=0 row
  debiased_whittle_lon1.py   — Z=X(i,j+1)-X(i,j),   excludes w2=0 col

Note: DEVICE_DW=cpu — DW likelihood runs on CPU only.
  Option A (mem): more CPUs, no GPU wait → recommended if GPU queue is long
  Option B (gpu-redhat): use if GPU helps field generation speed

```
cd ./jobscript/tco/gp_exercise
nano sim_dw_advec_direction_042026.sh
sbatch sim_dw_advec_direction_042026.sh
```

#### Option A — mem partition (CPU only, no GPU wait)
```bash
#!/bin/bash
#SBATCH --job-name=dw_advec_dir_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_advec_dir_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_advec_dir_042026_%j.err
#SBATCH --time=6:00:00
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_advec_direction_042026.py \
    --num-iters 60 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --dw-steps 5 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```


```
cd ./jobscript/tco/gp_exercise
nano sim_dw_advec_direction_042026.sh
sbatch sim_dw_advec_direction_042026.sh
```

#### Option B — gpu-redhat partition
```bash
#!/bin/bash
#SBATCH --job-name=dw_advec_dir_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_advec_dir_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_advec_dir_042026_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu033

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_advec_direction_042026.py \
    --num-iters 60 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --dw-steps 5 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```

---

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_advec_dir_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```
