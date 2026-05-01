

#!/bin/bash
#SBATCH --job-name=vec_irr_advmix_050126
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_irr_advmix_050126_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_irr_advmix_050126_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_advec_mix_050126.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --base-limit-a 20 \
    --base-limit-b 20 \
    --base-limit-c 20 \
    --ratio-limit-a 20 \
    --ratio-limit-b 16 \
    --ratio-limit-c 10 \
    --daily-stride 2 \
    --num-iters 300 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --lat-factor 100 \
    --lon-factor 10 \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --seed 42 \
    --out-prefix sim_vecchia_irregular_advec_mix

echo "Current date and time: $(date)"
