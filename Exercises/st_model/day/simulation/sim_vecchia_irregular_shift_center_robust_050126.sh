


#!/bin/bash
#SBATCH --job-name=vec_irr_shiftctr_050126
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_irr_shiftctr_050126_%A_%a.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_irr_shiftctr_050126_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

TRUE_ADVECS=(-0.10 -0.16 -0.25)
TRUE_ADVEC=${TRUE_ADVECS[$SLURM_ARRAY_TASK_ID]}
TRUE_LABEL=${TRUE_ADVEC/-/m}
TRUE_LABEL=${TRUE_LABEL/./p}

echo "Running on: $(hostname)"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID}"
echo "True advec_lon: ${TRUE_ADVEC}"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_shift_center_robust_050126.py \
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
    --true-advec-lat 0.25 \
    --true-advec-lon "${TRUE_ADVEC}" \
    --shift-offsets "0.10,0.126,0.16,0.20,0.25" \
    --shift-budgets "16:10,14:8,18:15" \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --seed 42 \
    --out-prefix "sim_vecchia_irregular_shift_center_robust_true${TRUE_LABEL}"

echo "Current date and time: $(date)"
