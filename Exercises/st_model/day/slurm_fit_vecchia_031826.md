### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer data (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_22_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024

scp "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_25_07.pkl" jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/fit_gpu_vecc_day_v05_031826.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
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

### GPU 인터랙티브 테스트
```
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash
python /home/jl2815/tco/exercise_25/st_model/torch_gpu_test.py
conda list | grep torch
```

---

### Vecchia GPU (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano fit_vecc_gpu_031826.sh
sbatch fit_vecc_gpu_031826.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=fit_vecc_h300m30
#SBATCH --output=/home/jl2815/tco/exercise_output/fit_vecc_h300m30.out
#SBATCH --error=/home/jl2815/tco/exercise_output/fit_vecc_h300m30.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_vecc_day_v05_031826.py \
    --v 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2024,2025" \
    --mm-cond-number 52 \
    --nheads 1000 \
    --limit-a 16 \
    --limit-b 16 \
    --limit-c 16 \
    --daily-stride 2 \
    --keep-exact-loc

echo "Current date and time: $(date)"

```

---

### Heads testing (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano heads_testing_031826.sh
sbatch heads_testing_031826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=heads_testing_031826
#SBATCH --output=/home/jl2815/tco/exercise_output/heads_testing_031826_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/heads_testing_031826_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:2
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_regular_vecc_testing_day_v05_010126.py \
    --v 0.5 \
    --mm-cond-number 30
```
