### Update package on Amarel
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_column3_compare_050626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_highres_column3_compare_050626.sh" jl2815@amarel.rutgers.edu:/home/jl2815/tco/jobscript/tco/gp_exercise/
```

### Run
```bash
ssh jl2815@amarel.rutgers.edu
cd /home/jl2815/tco/jobscript/tco/gp_exercise

nano sim_vecchia_highres_column3_compare_050626.sh
sbatch sim_vecchia_highres_column3_compare_050626.sh
```

### SLURM script
```bash
#!/bin/bash
#SBATCH --job-name=vec_col3_050626
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_col3_050626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_col3_050626_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi
echo "Current date and time: $(date)"

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_highres_column3_compare_050626.py \
    --num-iters 20 \
    --seed 123 \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --lat-factor 1 \
    --lon-factor 1 \
    --mm-cond-number 100 \
    --daily-stride 2 \
    --init-noise 0.25 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --column-chunk-size 4096 \
    --out-prefix "sim_vecchia_highres_column3_compare_050626"

echo "Current date and time: $(date)"
```

### Retrieve results
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_highres_column3_compare_050626_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_highres_column3_compare_050626_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_highres_column3_compare_050626_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"
```
