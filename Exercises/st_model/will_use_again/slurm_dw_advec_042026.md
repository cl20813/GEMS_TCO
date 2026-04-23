### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_advec_lat_042026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_advec_lon_042026.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

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
  Scenario A: advec_lat=0.2, advec_lon=0.0  (pure lat advection)
  Scenario B: advec_lat=0.0, advec_lon=-0.2 (pure lon advection)

Four filters: Raw | 2-1-1-0 | Lat-1 | Lon-1

DW modules required (must be in src/GEMS_TCO/):
  debiased_whittle_lat1.py   — Z=X(i+1,j)−X(i,j), excludes w1=0 row
  debiased_whittle_lon1.py   — Z=X(i,j+1)−X(i,j), excludes w2=0 col


cd ./jobscript/tco/gp_exercise
nano sim_advec_lat.sh
sbatch sim_advec_lat.sh


#### Scenario A — advec_lat=0.2 (mem partition)
```bash

#!/bin/bash
#SBATCH --job-name=dw_advec_lat_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_advec_lat_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_advec_lat_042026_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu033

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_advec_lat_042026.py \
    --num-iters 60 \
    --years "2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --dw-steps 5 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```
cd ./jobscript/tco/gp_exercise
nano sim_advec_lon.sh
sbatch sim_advec_lon.sh


#### Scenario B — advec_lon=-0.2 (mem partition)
```bash
#!/bin/bash
#SBATCH --job-name=dw_advec_lon_042026
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_advec_lon_042026_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_advec_lon_042026_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu037

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_advec_lon_042026.py \
    --num-iters 60 \
    --years "2024,2025" \
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
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_advec_lat_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_advec_lon_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```
