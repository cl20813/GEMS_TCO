### Transfer run files (mac -> Amarel)
```bash
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_advec_mix_050126.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model


```

### Transfer results (Amarel -> mac)
```bash
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_advec_mix_*_raw.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_advec_mix_*_model_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"

scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_advec_mix_*_param_summary.csv" "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/"
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

### Simulation: irregular Vecchia advection-mixed candidates

Real-data-like simulation:

high-resolution FFT field -> real GEMS source observation locations -> irregular Vecchia fit.

Models fit on the same simulated field and same initial values at every iteration. The target is to beat or match `20/18/15` while using a smaller conditioning budget.

- `Irr_Full_A20_B20_C20`: original `20/20/20`.
- `Irr_Cand_A20_B18_C15`: best standard reduced candidate from the previous run; total 55.
- `Irr_Ratio_A20_B16_C10`: aggressive standard reduced candidate that was unstable for advection.
- `Irr_Std_A20_B14_C08`: smaller standard control; total 44.
- `Irr_Std_A20_B12_C06`: very small standard control; total 40.

Same-total rescue candidates relative to `20/16/10`:

- `Irr_AdvMix_A20_B14p2_C08p2`: total 48; replace 2 lag-1 and 2 lag-2 local slots with upstream points.
- `Irr_AdvMix_A20_B14p2_C06p4`: total 48; keep lag-1 narrow and give t-2 a wider upstream band.
- `Irr_AdvMix_A20_B12p4_C06p4`: total 48; wider upstream bands, fewer local lag neighbors.

More aggressive, smaller-than-`20/16/10` candidates:

- `Irr_AdvMix_A20_B12p2_C08p2`: total 46.
- `Irr_AdvMix_A20_B12p2_C06p4`: total 46.
- `Irr_AdvMix_A20_B10p4_C06p4`: total 46; advec-heavy compact budget.
- `Irr_AdvMix_A20_B10p2_C06p4`: total 44; most aggressive advec-rescue candidate.

Near-`20/18/15` but still smaller candidates:

- `Irr_AdvMix_A20_B16p2_C10p2`: total 52.
- `Irr_AdvMix_A20_B16p2_C08p4`: total 52.
- `Irr_AdvMix_A20_B14p4_C08p4`: total 52.
- `Irr_AdvMix_A20_B16p4_C08p4`: total 54; largest advec-mixed budget still below `20/18/15`.

Upstream longitude offsets use the regular grid cell size `0.063`:

- lag 1 narrow band: `+2,+3` cells, roughly `+0.126,+0.189` degrees east.
- lag 1 wide band: `+1,+2,+3,+4` cells, roughly `+0.063` to `+0.252` degrees east.
- lag 2 narrow band: `+4,+5` cells, roughly `+0.252,+0.315` degrees east.
- lag 2 wide band: `+3,+4,+5,+6` cells, roughly `+0.189` to `+0.378` degrees east.

This tests whether directed upstream information can rescue the aggressive `20/16/10` budget without returning to `20/18/15`.

```bash
cd ./jobscript/tco/gp_exercise
nano sim_vecchia_irregular_advec_mix_050126.sh
sbatch sim_vecchia_irregular_advec_mix_050126.sh
```

```bash
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
```
