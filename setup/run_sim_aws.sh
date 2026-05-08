#!/bin/bash
# run_sim_aws.sh
# Runs the hybrid compare simulation on AWS EC2.
# Called by deploy_to_aws.sh via tmux, or run manually.

set -e
AWS_HOME="/home/ec2-user/gems_tco"

eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

python ${AWS_HOME}/exercise_25/st_model/sim_vecchia_irregular_hybrid_compare_050226.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 0 \
    --daily-stride 2 \
    --num-iters 300 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --lat-range "-3,2" \
    --lon-range "121,131" \
    --lat-factor 100 \
    --lon-factor 10 \
    --true-advec-lat 0.08 \
    --true-advec-lon "-0.10,-0.16,-0.25" \
    --init-noise 0.7 \
    --lbfgs-steps 5 \
    --lbfgs-eval 20 \
    --lbfgs-hist 10 \
    --compute-godambe \
    --seed 42 \
    --out-prefix "sim_vecchia_irregular_hybrid_compare_050226"

echo "Simulation complete: $(date)"
