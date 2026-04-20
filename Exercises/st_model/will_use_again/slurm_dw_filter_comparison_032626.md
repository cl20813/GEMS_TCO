### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_filter_comparison_032626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
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

### DW filter comparison (sbatch)

Compares DW_old (filter [[-2,1],[1,0]], DC-only exclusion)
vs  DW_new (filter [[-1,1],[1,-1]], full-axis exclusion)
on simulated FFT fields with true params matched to July 2024 fit.

```
cd ./jobscript/tco/gp_exercise
nano sim_dw_filter_comparison_032626.sh
sbatch sim_dw_filter_comparison_032626.sh
```



#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --partition=mem


```bash

#!/bin/bash
#SBATCH --job-name=dw_filter_cmp_032626
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_filter_cmp_032626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_filter_cmp_032626_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu032

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_filter_comparison_032626.py \
    --num-iters 300 \
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
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```




```
cd ./jobscript/tco/gp_exercise
nano sim_dw_filter_comparison_032626.sh
sbatch sim_dw_filter_comparison_032626.sh
```


```bash
#!/bin/bash
#SBATCH --job-name=dw_filter_cmp_032626
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_filter_cmp_032626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_filter_cmp_032626_%j.err
#SBATCH --time=24:00:00
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
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_filter_comparison_032626.py \
    --num-iters 300 \
    --years "2022,2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --dw-steps 5 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"

```