### Gradient / 2-1-1-0 / Lat-1 DW comparison on Amarel

This run keeps the existing high-resolution simulation, real-location matching,
and forced regular-grid setup from `sim_dw_filter_comparison_032626.py`.

Models compared:

- `Gradient`: vector finite increments `(D_lat X, D_lon X)`, p = 16 for 8 time blocks
- `2-1-1-0`: scalar `-2X + X_down + X_right`, p = 8
- `Lat-1`: scalar `X_down - X`, p = 8

The Python script uses `torch.optim.LBFGS(..., line_search_fn="strong_wolfe")`
for all three models.

### Update package code on Amarel

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer the run script

```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_dw_grad_2110_lat1_042826.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Connect and set up

```bash
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

### Submit

```bash
cd ./jobscript/tco/gp_exercise
nano sim_dw_grad_2110_lat1_042826.sh
sbatch sim_dw_grad_2110_lat1_042826.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=dw_grad_2110_lat1
#SBATCH --output=/home/jl2815/tco/exercise_output/dw_grad_2110_lat1_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/dw_grad_2110_lat1_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu032

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_dw_grad_2110_lat1_042826.py \
    --num-iters 300 \
    --years "2024,2025" \
    --month 7 \
    --lat-factor 100 \
    --lon-factor 10 \
    --dw-steps 5 \
    --init-noise 0.7 \
    --seed 42

echo "Current date and time: $(date)"
```

### Transfer results back

```bash
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_dw_grad_2110_lat1*.csv \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/"
```
