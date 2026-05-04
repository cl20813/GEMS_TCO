### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run files (mac -> Amarel)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_hybrid_compare_050226.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_hybrid_compare_050226.sh" jl2815@amarel.rutgers.edu:/home/jl2815/tco/jobscript/tco/gp_exercise/
```

### Transfer results (Amarel -> mac)
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_hybrid_compare_050226_*_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_hybrid_compare_050226_*_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_hybrid_compare_050226_*_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_hybrid_compare_050226_*_gim_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"
```

---

### Connect & setup
```bash
ssh jl2815@amarel.rutgers.edu
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
conda activate faiss_env
```

---

### Simulation: Vecchia irregular hybrid conditioning comparison

Real-data-like simulation comparing baseline local-only Vecchia against hybrid and pure-fresh conditioning sets.

High-resolution FFT field -> real GEMS source observation locations -> irregular Vecchia fit.

This experiment compares four conditioning families across three true longitudinal advection strengths:

Baseline (local-only):
- `Irr_Cand_A20_B18_C15`: standard local-only Vecchia (55 conditioning points).

Hybrid NearLocal (A=20, B_local=16, B_fresh=2, C_local=12, C_fresh=2, total=54):
- `Hybrid_NearLocal_L16F02_C12F02_Op0p063`: small offset (0.063).
- `Hybrid_NearLocal_L16F02_C12F02_Op0p126`: medium offset (0.126).

Hybrid Lean (A=20, B_local=8, B_fresh=4, C_local=4, C_fresh=3, total=41):
- `Hybrid_Lean_L08F04_C4F03_Op0p060`: small offset (0.060).
- `Hybrid_Lean_L08F04_C4F03_Op0p126`: medium offset (0.126).

Pure FreshShift (A=20, B_fresh=18, C_fresh=12, total=54):
- `FreshShift_A20_B18_C12_Op0p063`: small offset (0.063).
- `FreshShift_A20_B18_C12_Op0p126`: medium offset (0.126).

Single job cycles through `true_advec_lon ∈ {-0.10, -0.16, -0.25}` per iteration.
300 iterations → ~100 per advec_lon value. Running summary groups by `(model, true_advec_lon)`.

```bash
cd ./jobscript/tco/gp_exercise
nano sim_vecchia_irregular_hybrid_compare_050226.sh
sbatch sim_vecchia_irregular_hybrid_compare_050226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=vec_irr_hyb_050226
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_irr_hyb_050226_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_irr_hyb_050226_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_hybrid_compare_050226.py \
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

echo "Current date and time: $(date)"
```
