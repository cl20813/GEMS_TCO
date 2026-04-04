### Update packages (mac → Amarel)
```
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```

### Transfer run file (mac → Amarel)
```
scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/real_data/fit_gpu_vecc_2day_v05_040326.py" jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model
```

---

### Study design

- **Model**: Vecchia-Irr, Matérn ν=0.5
- **Temporal window**: 2 days × 8 hours = **p = 16** time steps per fit
- **Windows**: 14 non-overlapping windows per year (days 1+2, 3+4, ..., 27+28)
- **Years**: 2022, 2023, 2024, 2025
- **Total fits**: 14 × 4 = **56**
- **Output**: `real_vecc_2day_july_22_23_24_25_mm100.csv`
  - `day` column: `"YYYY-MM-DD_DD"` (e.g. `"2022-07-01_02"`)

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

### Vecchia 2-day GPU (sbatch)

```
cd ./jobscript/tco/gp_exercise
nano fit_vecc_2day_040326.sh
sbatch fit_vecc_2day_040326.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=vecc_2day_040326
#SBATCH --output=/home/jl2815/tco/exercise_output/vecc_2day_040326_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vecc_2day_040326_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu043

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

srun python /home/jl2815/tco/exercise_25/st_model/fit_gpu_vecc_2day_v05_040326.py \
    --v 0.5 \
    --space "1,1" \
    --windows "0,14" \
    --month 7 \
    --years "2022,2023,2024,2025" \
    --mm-cond-number 100 \
    --nheads 0 \
    --limit-a 20 \
    --limit-b 20 \
    --limit-c 20 \
    --daily-stride 2 \
    --keep-exact-loc

echo "Current date and time: $(date)"
```

---

### Transfer results (Amarel → mac)
```
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/july_22_23_24_25/real_vecc_2day_july_22_23_24_25_mm100.csv "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_23_24_25/"
```
