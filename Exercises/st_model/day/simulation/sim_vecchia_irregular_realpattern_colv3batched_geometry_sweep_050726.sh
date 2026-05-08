#!/bin/bash
#SBATCH --job-name=vec_geom_050726
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_geom_050726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_geom_050726_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_realpattern_colv3batched_geometry_sweep_050726.py \
    --num-iters 5 \
    --seed 123 \
    --years 2024 \
    --month 7 \
    --day-idxs 2 \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --lat-factor-hr 100 \
    --lon-factor-hr 10 \
    --mm-cond-number 100 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --init-noise 0.25 \
    --column-chunk-size 512 \
    --column-head-right-cols 0 \
    --out-prefix "sim_vecchia_irregular_realpattern_colv3batched_geometry_sweep_050726"

echo "Current date and time: $(date)"
