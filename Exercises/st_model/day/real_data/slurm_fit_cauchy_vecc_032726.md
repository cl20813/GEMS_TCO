### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_gpu_cauchy_vecc_day_032726.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design
Fits Generalized Cauchy Vecchia to all July days 2022–2025 (112 days × 3 β values).
Saves `vecchia_nll` (minimized NLL at MLE) for direct comparison with Matérn Vecchia NLL.

Output files (in `estimates/day/july_22_23_24_25/`):
- `real_cauchy_b05_july_22_23_24_25_h1000_mm16.csv`  — β=0.5
- `real_cauchy_b10_july_22_23_24_25_h1000_mm16.csv`  — β=1.0
- `real_cauchy_b20_july_22_23_24_25_h1000_mm16.csv`  — β=2.0

Columns: `day, sigma, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget, gc_beta, vecchia_nll, elapsed, cov_name`

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

### β=0.5  (sbatch)

~112 days × ~15 min/day ≈ 28 h; set time=72h for safety.

```
cd ./jobscript/tco/gp_exercise
nano fit_cauchy_b05_032726.sh
sbatch fit_cauchy_b05_032726.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=cauchy_b05_032726
#SBATCH --output=/home/jl2815/tco/exercise_output/cauchy_b05_032726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/cauchy_b05_032726_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_cauchy_vecc_day_032726.py \
    --gc-beta 0.5 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --mm-cond-number 100 \
    --nheads 300 \
    --limit-a 8 \
    --limit-b 8 \
    --limit-c 8 \
    --daily-stride 2 \
    --keep-exact-loc

echo "Current date and time: $(date)"

```

---


### β=1.0  (sbatch)

```
nano fit_cauchy_b10_032726.sh
sbatch fit_cauchy_b10_032726.sh
```

```bash

#!/bin/bash
#SBATCH --job-name=cauchy_b10_032726
#SBATCH --output=/home/jl2815/tco/exercise_output/cauchy_b10_032726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/cauchy_b10_032726_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_cauchy_vecc_day_032726.py \
    --gc-beta 1.0 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --mm-cond-number 100 \
    --nheads 1000 \
    --limit-a 16 \
    --limit-b 16 \
    --limit-c 16 \
    --daily-stride 2 \
    --keep-exact-loc

echo "Current date and time: $(date)"

```

---

### β=2.0  (sbatch)

```
nano fit_cauchy_b20_032726.sh
sbatch fit_cauchy_b20_032726.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=cauchy_b20_032726
#SBATCH --output=/home/jl2815/tco/exercise_output/cauchy_b20_032726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/cauchy_b20_032726_%j.err
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

srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_cauchy_vecc_day_032726.py \
    --gc-beta 2.0 \
    --space "1,1" \
    --days "0,28" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --mm-cond-number 16 \
    --nheads 1000 \
    --limit-a 16 \
    --limit-b 16 \
    --limit-c 16 \
    --daily-stride 2 \
    --keep-exact-loc

echo "Current date and time: $(date)"
```

---

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25/real_cauchy_b05_july_22_23_24_25_h1000_mm16.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25/real_cauchy_b10_july_22_23_24_25_h1000_mm16.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"

scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25/real_cauchy_b20_july_22_23_24_25_h1000_mm16.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```

---

### NLL comparison with Matérn (after transfer)
The `vecchia_nll` column can be compared directly against the Matérn NLL.
Add `vecchia_nll` saving to `fit_gpu_vecc_day_v05_031826.py` if not already present,
then compare per-day: `cauchy_nll < matern_nll` → Cauchy fits better on that day.
