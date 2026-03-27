### Update packages (mac → Amarel)
```
# mac
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco

# window
scp -r "C:\Users\joonw\TCO\GEMS_TCO-1\GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer data (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_22_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_23_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_25_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_D_whittle_day_v05_dynamic_grid_031826.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer estimate results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/*.json "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
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

### Debiased Whittle L-BFGS dynamic grid (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano fit_dw_031826.sh
sbatch fit_dw_031826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=fit_dw_031826
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_dw_031826.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_dw_031826.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_D_whittle_day_v05_dynamic_grid_031826.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --no-keep-exact-loc

echo "Current date and time: $(date)"
```
