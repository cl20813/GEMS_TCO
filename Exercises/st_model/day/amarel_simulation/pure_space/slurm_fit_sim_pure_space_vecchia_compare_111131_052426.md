# Pure-Space Cluster B2 Vecchia on Amarel

Created 2026-05-24.

This run uses the default 4x4 cluster isotropic max-min Vecchia:

- `GEMS_TCO.kernels_space_iso_cluster_052426.ClusterSpaceIsoTrendVecchiaFit`
- block shape `4x4`
- neighbor block count `B=2`

The sbatch below is one visible Slurm job, one node, no array.

## 1. Transfer Files

From local:

```bash
scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_sim_july_pure_space_vecchia_compare_111131_052426.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

## 2. Create sbatch on Amarel

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano slurm_fit_sim_pure_space_cluster_b2_111131_052426.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=ps_cluster_b2_111131
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/ps_cluster_b2_111131_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/ps_cluster_b2_111131_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module use /projects/community/modulefiles
module load anaconda/2024.06-ts840
module load cuda/12.1.0

eval "$(conda shell.bash hook)"
conda activate faiss_env

export PYTHONPATH=/home/jl2815/tco:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p /home/jl2815/tco/exercise_output/logs
mkdir -p /home/jl2815/tco/exercise_output/estimates/day/pure_space_cluster_b2_111131_052426

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_sim_july_pure_space_vecchia_compare_111131_052426.py

echo "Host: $(hostname)"
echo "Started: $(date)"
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

srun python "${SCRIPT}" \
  --num-sims 200 \
  --years 2022,2023,2024,2025 \
  --day-idxs all \
  --asset-sampling cycle \
  --data-kind real_locations \
  --lat-range=-3,7 \
  --lon-range=111,131 \
  --cluster-neighbor-blocks 2 \
  --cluster-block-shape 4x4 \
  --mean-design latlon \
  --lbfgs-steps 5 \
  --lbfgs-eval 15 \
  --cluster-target-chunk-size 96 \
  --summary-every 5 \
  --require-cuda \
  --resume \
  --out-dir /home/jl2815/tco/exercise_output/estimates/day/pure_space_cluster_b2_111131_052426

echo "Finished: $(date)"
```

Submit:

```bash
sbatch slurm_fit_sim_pure_space_cluster_b2_111131_052426.sh
```

## 3. Monitor

```bash
tail -f /home/jl2815/tco/exercise_output/logs/ps_cluster_b2_111131_<JOBID>.out

cat /home/jl2815/tco/exercise_output/estimates/day/pure_space_cluster_b2_111131_052426/running_summary.txt
```

## 4. Copy Results Back

```bash
scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/estimates/day/pure_space_cluster_b2_111131_052426 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/"
```
