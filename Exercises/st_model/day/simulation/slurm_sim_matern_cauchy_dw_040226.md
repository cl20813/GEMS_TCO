### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_matern_cauchy_dw_comparison_040226.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

### Transfer results (Amarel → mac)
```
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_matern_cauchy_dw/ "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/sim_matern_cauchy_dw/"
```

---

### Study design
Three-way comparison: DGP = Matérn ν=0.5

| Model | Kernel | Data |
|---|---|---|
| Vecc_Matern | Matérn ν=0.5 | Irregular source locations |
| Vecc_Cauchy | Generalized Cauchy β | Same irregular source locations |
| DW | Matérn ν=0.5 (spectral) | Regular re-gridded (step3) |

Output files (in `estimates/day/sim_matern_cauchy_dw/`):
- `sim_matern_cauchy_dw_b10_<date>.csv`      — raw per-iteration records
- `sim_matern_cauchy_dw_b10_<date>_summary.csv` — per-parameter RMSRE / P90-P10 table

Columns: `iter, obs_year, obs_day, model, gc_beta, rmsre, time_s, sigmasq_est, range_lat_est, range_lon_est, range_t_est, advec_lat_est, advec_lon_est, nugget_est, init_sigmasq, init_range_lon`

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

### Cauchy β=1.0  (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano sim_mcd_b10_040226.sh
sbatch sim_mcd_b10_040226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_mcd_b10_040226
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_mcd_b10_040226_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_mcd_b10_040226_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu041

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_matern_cauchy_dw_comparison_040226.py \
    --gc-beta 1.0 \
    --num-iters 100 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --lat-factor 100 \
    --lon-factor 10

echo "Current date and time: $(date)"


```

---

### Cauchy β=0.5  (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano sim_mcd_b05_040226.sh
sbatch sim_mcd_b05_040226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_mcd_b05_040226
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_mcd_b05_040226_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_mcd_b05_040226_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu033

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_matern_cauchy_dw_comparison_040226.py \
    --gc-beta 0.5 \
    --num-iters 100 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --lat-factor 100 \
    --lon-factor 10

echo "Current date and time: $(date)"
```

---

### Cauchy β=2.0  (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano sim_mcd_b20_040226.sh
sbatch sim_mcd_b20_040226.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=sim_mcd_b20_040226
#SBATCH --output=/home/jl2815/tco/exercise_output/sim_mcd_b20_040226_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/sim_mcd_b20_040226_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu031

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0
eval "$(conda shell.bash hook)"
conda activate faiss_env

echo "Running on: $(hostname)"
nvidia-smi

srun python /home/jl2815/tco/exercise_25/st_model/sim_matern_cauchy_dw_comparison_040226.py \
    --gc-beta 2.0 \
    --num-iters 100 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --lat-factor 100 \
    --lon-factor 10

echo "Current date and time: $(date)"
```
