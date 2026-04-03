### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again/sim_heads_vs_limit_032726.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design
- **DGP**: Generalized Cauchy β=1.0 via FFT circulant embedding (correctly specified)
- **Grid**: nheads ∈ {0,100,200,400,800} × limit ∈ {4,6,8,12,16,20,24} = **35 combos**
- **mm_cond_number = 30** (fixed for all combos; NNS map stores 30 neighbors)
- **Per iteration**: all 35 combos fitted on the same generated dataset → fair comparison
- **10 parallel jobs × 10 iters each = 100 total iterations** (merge CSVs afterward)

Key iso-compute comparison (within the grid):
- `(heads=0,   limit=24)` vs `(heads=800, limit=16)` — similar cost, heads-only vs limit-only
- `(heads=0,   limit=16)` vs `(heads=400, limit=8)`  — another iso-compute pair

Output files (in `estimates/day/`):
- `sim_heads_vs_limit_{date}_j{0..9}.csv`        — raw records (35 combos × iters)
- `sim_heads_vs_limit_summary_{date}_j{0..9}.csv` — mean/sd per combo per job

Merge after all jobs finish:
```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob("sim_heads_vs_limit_*_j*.csv")])
df.to_csv("sim_heads_vs_limit_merged.csv", index=False)
```

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

### Single job (10 iters, 35 combos each)

~35 combos × ~12 min × 10 iters ≈ 70 h; set time=72h.

```
cd ./jobscript/tco/gp_exercise
nano sim_heads_vs_limit_032726.sh
sbatch sim_heads_vs_limit_032726.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=hvl_032726
#SBATCH --output=/home/jl2815/tco/exercise_output/hvl_032726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/hvl_032726_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
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

srun python /home/jl2815/tco/exercise_25/st_model/sim_heads_vs_limit_032726.py \
    --num-iters 10 \
    --job-id 0 \
    --lat-factor 100 \
    --lon-factor 10 \
    --years "2022,2023,2024,2025" \
    --month 7 \
    --daily-stride 2 \
    --init-noise 0.5 \
    --seed 42

echo "Current date and time: $(date)"
```

---

### Transfer results (Amarel → mac)
```
scp "jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_heads_vs_limit_*.csv" \
    "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/heads_vs_limit/"
```

---

### Quick local test
```bash
conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer

python sim_heads_vs_limit_local_032726.py \
    --num-iters 1 \
    --lat-factor 10 \
    --lon-factor 4
```

---

### How to interpret results
After merging CSVs, the summary table has columns:
`nheads, limit, mean_loss, mean_time_s, mean_rmsre`

Key questions to answer from the data:
1. **Does heads help at all?**  Compare rows with same `limit`, varying `nheads`.
   If RMSRE decreases as `nheads` increases → heads add value.
2. **Is heads better than limit at same cost?**
   Compare `(h=0, L=24)` vs `(h=800, L=16)` for RMSRE.
3. **Efficiency frontier**: scatter plot `(mean_time_s, mean_rmsre)` for all 35 combos.
   Pareto-optimal combos are those where no other combo is both faster and more accurate.
4. **NLL caveat**: NLL comparisons are most valid within fixed-`nheads` groups
   (varying limit only), since different heads values change the likelihood approximation itself.
