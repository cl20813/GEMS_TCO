# Real-Pattern Irregular Column V3 Fill vs Hybrid

Purpose: Amarel simulation with known truth and real GEMS source-location/missing pattern.

Models in one run:

- Hybrid Lean exact-location, nominal m = 41
- Column V3 Fill, `head_right_cols=3`, nominal tail m = 42
- Column V3 Fill, `head_right_cols=0`, nominal tail m = 42

Column details:

- Reverse-L scan/order uses the regular grid.
- Covariance offsets use real `Source_Latitude` / `Source_Longitude`.
- Missing neighbors are skipped, then later reverse-L candidates are scanned until the per-lag cap is filled when possible.
- Template count is allowed to explode because this run is for RMSRE/conditioning geometry, not speed.

## Local Files

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch_fill.py  # DEPRECATED/REMOVED
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.py
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.sh
```

## Transfer From Mac

Run from the local Mac terminal, not inside Amarel:

```bash
scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/
# DEPRECATED/REMOVED: scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch_fill.py  # DEPRECATED/REMOVED jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/
scp /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.py jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/
scp /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.sh jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/
```

## SLURM Script Content

The third `scp` above transfers this file. If you want to recreate it directly on Amarel, make
`/home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.sh`
with this exact content:

```bash
#!/bin/bash
#SBATCH --job-name=vec_irrl_050726
#SBATCH --output=/home/jl2815/tco/exercise_output/vec_irrl_050726_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec_irrl_050726_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
echo "Current date and time: $(date)"

srun python /home/jl2815/tco/exercise_25/st_model/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.py \
    --num-iters 5 \
    --seed 123 \
    --years 2024 \
    --month 7 \
    --day-idxs 2 \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --lat-factor-hr 100 \
    --lon-factor-hr 10 \
    --mm-cond-number 100 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --init-noise 0.25 \
    --column-chunk-size 512 \
    --out-prefix "sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726"

echo "Current date and time: $(date)"
```

## Submit On Amarel

```bash
cd /home/jl2815/tco/exercise_25/st_model
nano sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.sh
sbatch sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.sh
```

## Monitor

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/vec_irrl_050726_<JOBID>.out
tail -f /home/jl2815/tco/exercise_output/vec_irrl_050726_<JOBID>.err
```

## Output CSVs

```bash
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726_raw.csv
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726_model_summary.csv
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726_param_summary.csv
```

## Retrieve To Mac

```bash
scp jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726_*.csv /Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day/
```

## Resource Notes

The high-resolution matching grid uses `lat_factor_hr=100` and `lon_factor_hr=10`.
The SLURM request is intentionally larger than the earlier perfect-grid run:

```bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
```

If queue time is too long, first reduce `--num-iters 5` to `--num-iters 1` for a smoke run.
