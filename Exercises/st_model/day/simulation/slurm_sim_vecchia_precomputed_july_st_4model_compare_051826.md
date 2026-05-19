# Precomputed July ST Four-Model Vecchia Compare

Purpose: compare four Vecchia variants on the reusable July ST circulant
simulation assets.  This job does not generate simulation data; it only reads
the existing pickles under:

```bash
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern
```

Models in one run:

- `GEMS_TCO.kernels_vecchia_hybrid.HybridVecchiaFit`
- `GEMS_TCO.kernels_vecchia_cluster_hybrid.ClusterHybridVecchiaFit`
- `GEMS_TCO.kernel_vecchia_col_batch.ReverseLColumnVecchiaFitBatch`
- `GEMS_TCO.kernels_vecchia_cluster_column_batch.ClusterColumnVecchiaFitBatch`

The default run uses 100 pre-generated daily 8-hour assets from 2022-2025,
GPU, `gridded` simulation pickle layout, regular-grid ordering/block geometry,
and real `Source_Latitude` / `Source_Longitude` covariance coordinates.

## Local Files

```bash
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_hybrid.py
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_cluster_hybrid.py
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py
/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_cluster_column_batch.py
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_precomputed_july_st_4model_compare_051826.py
/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_precomputed_july_st_4model_compare_051826.sh
```

## Transfer From Mac

Run from the local Mac terminal:

### Update packages (mac -> Amarel)
```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" jl2815@amarel.rutgers.edu:/home/jl2815/tco
```


```bash
scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_hybrid.py \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_cluster_hybrid.py \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernel_vecchia_col_batch.py \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO/kernels_vecchia_cluster_column_batch.py \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/GEMS_TCO/

scp /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation/sim_vecchia_precomputed_july_st_4model_compare_051826.py \
    jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/


```

## Submit On Amarel

```bash
cd /home/jl2815/tco/exercise_25/st_model
nano sim_vecchia_precomputed_july_st_4model_compare_051826.sh

sbatch sim_vecchia_precomputed_july_st_4model_compare_051826.sh
```

## SLURM Script Content

```bash
#!/bin/bash
#SBATCH --job-name=vec4_pre_051826
#SBATCH --output=/home/jl2815/tco/exercise_output/vec4_pre_051826_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/vec4_pre_051826_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu043

set -euo pipefail

export PYTHONPATH="/home/jl2815/tco:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/jl2815/.conda/envs/faiss_env/bin/python"
SCRIPT="/home/jl2815/tco/exercise_25/st_model/sim_vecchia_precomputed_july_st_4model_compare_051826.py"
SIM_ROOT="/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"
OUTDIR="/home/jl2815/tco/exercise_output/estimates/day"

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "Python: ${PYTHON}"
echo "Current date and time: $(date)"
echo "Using pre-generated simulation assets from: ${SIM_ROOT}"

srun "${PYTHON}" - <<'PY'
import os
import sys
import torch

print("preflight python:", sys.executable, flush=True)
print("preflight torch:", torch.__version__, flush=True)
print("preflight CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("preflight cuda_available:", torch.cuda.is_available(), flush=True)
print("preflight cuda_device_count:", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed before model run.")
print("preflight device0:", torch.cuda.get_device_name(0), flush=True)
PY

srun "${PYTHON}" "${SCRIPT}" \
    --require-cuda \
    --num-iters 100 \
    --seed 123 \
    --years "2022,2023,2024,2025" \
    --day-idxs all \
    --max-asset-days 100 \
    --sim-data-root "${SIM_ROOT}" \
    --data-kind gridded \
    --lat-range=-3,2 \
    --lon-range=121,131 \
    --mm-cond-number 100 \
    --lbfgs-steps 5 \
    --lbfgs-eval 15 \
    --lbfgs-hist 10 \
    --init-noise 0.25 \
    --column-above-count 3 \
    --column-right-col-count 3 \
    --column-per-lag-count 14 \
    --column-lag-count 2 \
    --column-head-right-cols 0 \
    --column-chunk-size 512 \
    --block-shape 3x3 \
    --cluster-hybrid-chunk-size 128 \
    --cluster-column-chunk-size 64 \
    --out-dir "${OUTDIR}" \
    --out-prefix "sim_vecchia_precomputed_july_st_4model_compare_051826"

echo "Current date and time: $(date)"
```

## Output CSVs

```bash
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_precomputed_july_st_4model_compare_051826_raw.csv
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_precomputed_july_st_4model_compare_051826_model_summary.csv
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_precomputed_july_st_4model_compare_051826_param_summary.csv
/home/jl2815/tco/exercise_output/estimates/day/sim_vecchia_precomputed_july_st_4model_compare_051826_errors.csv
```

Running summaries printed to stdout include mean, median, and p90-p10 for model
metrics, plus parameter RMSRE, median relative error, and p90-p10.

## Monitor

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/vec4_pre_051826_<JOBID>.out
tail -f /home/jl2815/tco/exercise_output/vec4_pre_051826_<JOBID>.err
```

## Smoke Test

For a quick queue-friendly run, temporarily change:

```bash
--num-iters 1
```
