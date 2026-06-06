# Pure-Space Group Vecchia Global And Tile Nugget Fits

This is a separate 2024-only expanded-area diagnostic.  The hourly global
spatial fit uses pure-space anisotropic cluster/group Vecchia with
`cluster_block_shape=4x4` and `cluster_neighbor_blocks=2`.  Each hour first
runs a Vecchia QC pass, removes observations with `|whitened residual| > 10`,
and then refits on the cleaned observations.  After the post-QC global fit,
each tile is fitted with the same torch/autograd group Vecchia likelihood.

Key settings:

- data domain: expanded region, latitude `[-3, 7]`, longitude `[111, 131]`
- source pkl file:
  `/home/jl2815/tco/data/pickle_2024/tco_grid_lat-3to7_lon111to131_24_07.pkl`
- year/month: `2024-07`
- fitted subset: first 10 observed days, i.e. the first 80 July hour slots
- smoothness values: `nu=0.2` and `nu=0.5`
- execution style: one Slurm job, one GPU node, sequential loops, no job array
- global fit: anisotropic Matern group Vecchia, `beta0 + beta1 * centered latitude`
- tile fit: torch/autograd anisotropic Matern group Vecchia, estimating
  `sigma^2`, `range_lat`, `range_lon`, and nugget per tile
- tile plots: longitude/latitude axes plus GEMS nadir marker at `(lon=128, lat=0)`

The Python script keeps the legacy option name `--tile-full-max-points` for
compatibility, but it now means deterministic thinning before tile Vecchia.
Set `TILE_FULL_MAX_POINTS=0` if you want all tile points.

## 1. Transfer Files

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_full_lik_tile_111131_052626.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

The expanded July pkl file should already be on Amarel if this was run locally:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step5_transfer_to_amarel_032626.py --years 2024 --months 7 --only-extra-bounds
```

## 2. Single Sequential Slurm Job

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano fit_sim_july_group_vecchia_global_full_lik_tile_111131_2024_seq_052626.sh
sbatch fit_sim_july_group_vecchia_global_full_lik_tile_111131_2024_seq_052626.sh
```

#SBATCH --nodelist=gpu021

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=real_gv_fulltile24
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/real_gv_fulltile24_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/real_gv_fulltile24_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEAR=2024
EXPECTED_HOURS=248
FIT_HOURS=80
SMOOTHS=(0.2 0.5)
QC_WHITENED_THRESHOLD=10

# Legacy name: now controls deterministic thinning before tile Vecchia.
TILE_FULL_MAX_POINTS=1200

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_full_lik_tile_111131_052626.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/eda/real_data/july_expanded_bounds_group_vecchia_global_tile_vecchia_111131_first10days_060526

YY=$(printf "%02d" $((YEAR % 100)))
DATA_PATH=${DATA_ROOT}/pickle_${YEAR}/tco_grid_lat-3to7_lon111to131_${YY}_07.pkl
YEAR_OUT=${BASE_ROOT}/${YEAR}_july_expanded_bounds
MANIFEST=${YEAR_OUT}/manifest_hours.csv

mkdir -p /home/jl2815/tco/exercise_output/logs "${YEAR_OUT}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Data=${DATA_PATH}"
echo "Output=${YEAR_OUT}"
echo "Fitting first ${FIT_HOURS} manifest rows only"
echo "QC whitened threshold=${QC_WHITENED_THRESHOLD}"
nvidia-smi
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

echo "Building manifest for ${YEAR}..."
python "${SCRIPT}" \
  --mode manifest \
  --input "${DATA_PATH}" \
  --output-dir "${YEAR_OUT}" \
  --manifest "${MANIFEST}" \
  --month "${YEAR}-07" \
  --expected-hours "${EXPECTED_HOURS}" \
  --time-col hour \
  --x-col Longitude \
  --y-col Latitude \
  --value-col ColumnAmountO3 \
  --coords raw \
  --lat-range=-3,7 \
  --lon-range=111,131 \
  --require-region-cover

echo "Manifest row count:"
wc -l "${MANIFEST}"

for SMOOTH in "${SMOOTHS[@]}"; do
  SMOOTH_TAG=${SMOOTH/./p}
  OUTDIR=${YEAR_OUT}/nu${SMOOTH_TAG}
  mkdir -p "${OUTDIR}"
  export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
  mkdir -p "${MPLCONFIGDIR}"

  echo "============================================================"
  echo "Sequential fit start: year=${YEAR}, smooth=${SMOOTH}, output=${OUTDIR}"
  echo "Fitting only first ${FIT_HOURS} hour slots from the July manifest"
  echo "Tile Vecchia deterministic max points=${TILE_FULL_MAX_POINTS}"
  echo "Vecchia QC threshold=${QC_WHITENED_THRESHOLD}"
  echo "Time: $(date)"
  echo "============================================================"

  for HOUR_IDX in $(seq 0 $((FIT_HOURS - 1))); do
    echo "---- year=${YEAR} smooth=${SMOOTH} hour_idx=${HOUR_IDX}/${FIT_HOURS} first-10-days $(date) ----"
    srun python "${SCRIPT}" \
      --mode fit \
      --input "${DATA_PATH}" \
      --output-dir "${OUTDIR}" \
      --manifest "${MANIFEST}" \
      --month "${YEAR}-07" \
      --expected-hours "${EXPECTED_HOURS}" \
      --array-index "${HOUR_IDX}" \
      --time-col hour \
      --x-col Longitude \
      --y-col Latitude \
      --value-col ColumnAmountO3 \
      --coords raw \
      --lat-range=-3,7 \
      --lon-range=111,131 \
      --require-region-cover \
      --smooth "${SMOOTH}" \
      --tile-y 4 \
      --tile-x 8 \
      --cluster-block-shape 4x4 \
      --cluster-neighbor-blocks 2 \
      --target-chunk-size 96 \
      --min-target-points 1 \
      --qc-whitened-threshold "${QC_WHITENED_THRESHOLD}" \
      --min-tile-points 80 \
      --tile-full-max-points "${TILE_FULL_MAX_POINTS}" \
      --mean-design lat \
      --sigmasq-init 10 \
      --range-init 0.2 \
      --range-lat-init 0.2 \
      --range-lon-init 0.2 \
      --nugget-init 1.0 \
      --device cuda
  done

  echo "Summarizing first ${FIT_HOURS} fitted hours for year=${YEAR}, smooth=${SMOOTH}: $(date)"
  python "${SCRIPT}" \
    --mode summarize \
    --input "${DATA_PATH}" \
    --output-dir "${OUTDIR}" \
    --manifest "${MANIFEST}" \
    --month "${YEAR}-07" \
    --expected-hours "${EXPECTED_HOURS}" \
    --time-col hour \
    --x-col Longitude \
    --y-col Latitude \
    --value-col ColumnAmountO3 \
    --coords raw \
    --lat-range=-3,7 \
    --lon-range=111,131 \
    --require-region-cover \
    --smooth "${SMOOTH}" \
    --tile-y 4 \
    --tile-x 8 \
    --cluster-block-shape 4x4 \
    --cluster-neighbor-blocks 2 \
    --target-chunk-size 96 \
    --min-target-points 1 \
    --min-tile-points 80 \
    --tile-full-max-points "${TILE_FULL_MAX_POINTS}" \
    --qc-whitened-threshold "${QC_WHITENED_THRESHOLD}" \
    --mean-design lat

  echo "Completed year=${YEAR}, smooth=${SMOOTH}: $(date)"
done

echo "Finished tile Vecchia run: $(date)"
```

Submit:

```bash
sbatch fit_sim_july_group_vecchia_global_full_lik_tile_111131_2024_seq_052626.sh
```

## 3. Monitor

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/logs/real_gv_fulltile24_<JOBID>.out

ls -R /home/jl2815/tco/exercise_output/eda/real_data/july_expanded_bounds_group_vecchia_global_tile_vecchia_111131_first10days_060526
```

Expected output folders:

```text
july_expanded_bounds_group_vecchia_global_tile_vecchia_111131_first10days_060526/
  2024_july_expanded_bounds/
    manifest_hours.csv
    nu0p2/
      hourly/
      summary/
    nu0p5/
      hourly/
      summary/
```

## 4. Copy Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/real_data/july_expanded_bounds_group_vecchia_global_tile_vecchia_111131_first10days_060526 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data/"
```
