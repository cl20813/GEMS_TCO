### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_cauchy_beta_comparison_032626.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design
- **Data DGP**: Matérn ν=0.5 (exponential decay) — generated via FFT circulant embedding
- **Models fitted**: Cauchy β=0.5, β=1.0, β=2.0 — all misspecified relative to DGP
- **Purpose**: misspecification robustness — which β recovers Matérn parameters most faithfully?
- **Dataset**: irregular source locations (GEMS obs pattern), same irr_map for all three models per iteration

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

### Cauchy β comparison (sbatch)

~100 iters × 3 models × ~8 min/model ≈ 40 h; set time=72h for safety.

```
cd ./jobscript/tco/gp_exercise
nano sim_cauchy_beta_comparison_032626.sh
sbatch sim_cauchy_beta_comparison_032626.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=cauchy_beta_032626
#SBATCH --output=/home/jl2815/tco/exercise_output/cauchy_beta_032626_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/cauchy_beta_032626_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu031

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

srun python /home/jl2815/tco/exercise_25/st_model/sim_cauchy_beta_comparison_032626.py \
    --v 0.5 \
    --mm-cond-number 100 \
    --nheads 300 \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
    --daily-stride 8 \
    --num-iters 500 \
    --lat-factor 100 \
    --lon-factor 10

echo "Current date and time: $(date)"
```

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_cauchy_beta_comparison_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_cauchy_beta_summary_*.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"
```

---

### Quick local test
```bash
conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation

python sim_cauchy_beta_comparison_032626.py \
    --num-iters 1 \
    --lat-factor 10 \
    --lon-factor 4 \
    --nheads 100
```
