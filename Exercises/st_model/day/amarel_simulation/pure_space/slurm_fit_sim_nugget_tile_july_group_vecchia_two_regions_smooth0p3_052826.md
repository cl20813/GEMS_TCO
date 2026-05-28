# Two Open-Water July Group Vecchia Tile Nugget Fit, Smooth 0.3

This run fits the two newly selected July open-water regions in one sequential
Slurm job.  It uses pure-space isotropic cluster/group Vecchia and fixes Matern
smoothness at `nu=0.3`.

Regions:

- `west_pacific`: latitude `[15, 20]`, longitude `[123, 145]`
- `micronesia_open_water`: latitude `[1, 6]`, longitude `[129, 139]`

Key settings:

- years: `2022`, `2023`, `2024`, `2025`
- smoothness: `nu=0.3` only
- execution style: one Slurm job, one GPU node, sequential region/year/hour loops
- output separation: region-specific folders under one shared run root
- tile grid: `4 x 8`
- mean design: `lat`
- global fit: isotropic Matern with `log(phi1), log(phi2), log(nugget)`
- tile fit: hold global `sigma^2` and range fixed, profile only tile nugget

The West Pacific requested longitude range is `123..145`, but the finite GEMS
retrieval swath reaches about `139.5E`.  The regular grid still carries the
requested longitude domain with missing values, so the run can keep the
requested tag while the finite-value coverage warning is expected.

The `micronesia_open_water` region keeps the same July time-step pattern as the
West Pacific region: `240/248/248/247` for 2022-2025.  It replaces the earlier
equatorial Indian Ocean candidate because that candidate did not have full
observed-hour coverage.

## 1. Transfer Files

From the local machine:

```bash
ssh jl2815@amarel.rutgers.edu "mkdir -p /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space"

scp -r "/Users/joonwonlee/Documents/GEMS_TCO-1/src/GEMS_TCO" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/

scp "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_two_regions_smooth0p3_052826.py" \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/
```

If the job already finished all hourly fits and failed only during
`--mode summarize` with `IndexError: index 8 is out of bounds for axis 1 with
size 8`, just re-copy the patched Python file above and rerun the summarize
commands for the affected completed output folders.  You do not need to refit
the hourly models.

Example for one finished output folder:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space

python fit_sim_july_spatial_nugget_tiles_group_vecchia_two_regions_smooth0p3_052826.py \
  --mode summarize \
  --input /home/jl2815/tco/data/pickle_2022/tco_grid_lat1to6_lon129to139_22_07.pkl \
  --output-dir /home/jl2815/tco/exercise_output/eda/real_data/july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826/micronesia_open_water/2022_july_lat1to6_lon129to139/nu0p3 \
  --manifest /home/jl2815/tco/exercise_output/eda/real_data/july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826/micronesia_open_water/2022_july_lat1to6_lon129to139/manifest_hours.csv \
  --month 2022-07 \
  --expected-hours 240 \
  --time-col hour \
  --x-col Longitude \
  --y-col Latitude \
  --value-col ColumnAmountO3 \
  --coords raw \
  --lat-range=1,6 \
  --lon-range=129,139 \
  --require-region-cover \
  --smooth 0.3 \
  --tile-y 4 \
  --tile-x 8 \
  --cluster-block-shape 4x4 \
  --cluster-neighbor-blocks 2 \
  --target-chunk-size 96 \
  --min-target-points 1 \
  --min-tile-points 80 \
  --mean-design lat \
  --nadir-lat 3.5 \
  --nadir-lon 128.2
```

The two region data bundles should already be on Amarel after running:

```text
/Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st/west_pacific_step5_scp_commands.md
/Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st/micronesia_step5_scp_commands.md
```

## 2. Single Sequential Slurm Job

On Amarel:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space
nano fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh
sbatch fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh
```

If the script was already created on Amarel before the `--lat-range` fix, patch
that existing `.sh` file before resubmitting:

```bash
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space

sed -i 's/--lat-range "\${LAT_RANGE}"/--lat-range="\${LAT_RANGE}"/g' fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh
sed -i 's/--lon-range "\${LON_RANGE}"/--lon-range="\${LON_RANGE}"/g' fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh

grep -n -- '--lat-range\\|--lon-range' fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh
```

The grep output must show the equals-sign form:

```text
--lat-range="${LAT_RANGE}"
--lon-range="${LON_RANGE}"
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=ow2_gv_nu03
#SBATCH --output=/home/jl2815/tco/exercise_output/logs/ow2_gv_nu03_%j.out
#SBATCH --error=/home/jl2815/tco/exercise_output/logs/ow2_gv_nu03_%j.err
#SBATCH --time=10:00:00
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

YEARS=(2022 2023 2024 2025)
SMOOTH=0.3
SMOOTH_TAG=0p3

SCRIPT=/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/pure_space/fit_sim_july_spatial_nugget_tiles_group_vecchia_two_regions_smooth0p3_052826.py
DATA_ROOT=/home/jl2815/tco/data
BASE_ROOT=/home/jl2815/tco/exercise_output/eda/real_data/july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826

mkdir -p /home/jl2815/tco/exercise_output/logs "${BASE_ROOT}"

echo "Host: $(hostname)"
echo "Started: $(date)"
nvidia-smi
which python
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

expected_hours_for_region_year() {
  local region="$1"
  local year="$2"
  case "${region}:${year}" in
    west_pacific:2022) echo 240 ;;
    west_pacific:2023) echo 248 ;;
    west_pacific:2024) echo 248 ;;
    west_pacific:2025) echo 247 ;;
    micronesia_open_water:2022) echo 240 ;;
    micronesia_open_water:2023) echo 248 ;;
    micronesia_open_water:2024) echo 248 ;;
    micronesia_open_water:2025) echo 247 ;;
    *) echo "Unsupported region/year: ${region}/${year}" >&2; return 1 ;;
  esac
}

run_region() {
  local REGION="$1"
  local TAG="$2"
  local LAT_RANGE="$3"
  local LON_RANGE="$4"
  local NADIR_LAT="$5"
  local NADIR_LON="$6"

  local REGION_ROOT="${BASE_ROOT}/${REGION}"
  mkdir -p "${REGION_ROOT}"

  for YEAR in "${YEARS[@]}"; do
    local EXPECTED_HOURS
    EXPECTED_HOURS=$(expected_hours_for_region_year "${REGION}" "${YEAR}")
    local YY
    YY=$(printf "%02d" $((YEAR % 100)))
    local DATA_PATH="${DATA_ROOT}/pickle_${YEAR}/tco_grid_${TAG}_${YY}_07.pkl"
    local YEAR_OUT="${REGION_ROOT}/${YEAR}_july_${TAG}"
    local OUTDIR="${YEAR_OUT}/nu${SMOOTH_TAG}"
    local MANIFEST="${YEAR_OUT}/manifest_hours.csv"

    mkdir -p "${YEAR_OUT}" "${OUTDIR}"
    export MPLCONFIGDIR="${OUTDIR}/.mplconfig"
    mkdir -p "${MPLCONFIGDIR}"

    echo "============================================================"
    echo "Region=${REGION}, year=${YEAR}, smooth=${SMOOTH}, expected_hours=${EXPECTED_HOURS}"
    echo "Data=${DATA_PATH}"
    echo "Output=${OUTDIR}"
    echo "Time=$(date)"
    echo "============================================================"

    echo "Building manifest..."
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
      --lat-range="${LAT_RANGE}" \
      --lon-range="${LON_RANGE}" \
      --require-region-cover

    echo "Manifest row count:"
    wc -l "${MANIFEST}"

    for HOUR_IDX in $(seq 0 $((EXPECTED_HOURS - 1))); do
      echo "---- region=${REGION} year=${YEAR} smooth=${SMOOTH} hour_idx=${HOUR_IDX}/${EXPECTED_HOURS} $(date) ----"
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
        --lat-range="${LAT_RANGE}" \
        --lon-range="${LON_RANGE}" \
        --require-region-cover \
        --smooth "${SMOOTH}" \
        --tile-y 4 \
        --tile-x 8 \
        --cluster-block-shape 4x4 \
        --cluster-neighbor-blocks 2 \
        --target-chunk-size 96 \
        --min-target-points 1 \
        --min-tile-points 80 \
        --mean-design lat \
        --sigmasq-init 10 \
        --range-init 0.2 \
        --nugget-init 1.0 \
        --nadir-lat "${NADIR_LAT}" \
        --nadir-lon "${NADIR_LON}" \
        --device cuda
    done

    echo "Summarizing region=${REGION}, year=${YEAR}, smooth=${SMOOTH}: $(date)"
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
      --lat-range="${LAT_RANGE}" \
      --lon-range="${LON_RANGE}" \
      --require-region-cover \
      --smooth "${SMOOTH}" \
      --tile-y 4 \
      --tile-x 8 \
      --cluster-block-shape 4x4 \
      --cluster-neighbor-blocks 2 \
      --target-chunk-size 96 \
      --min-target-points 1 \
      --min-tile-points 80 \
      --mean-design lat \
      --nadir-lat "${NADIR_LAT}" \
      --nadir-lon "${NADIR_LON}"

    local QUICKLOOK_DIR="${REGION_ROOT}/tile_median_quicklook"
    mkdir -p "${QUICKLOOK_DIR}"
    if [[ -f "${OUTDIR}/summary/tile_nugget_summary_4x8.csv" ]]; then
      cp "${OUTDIR}/summary/tile_nugget_summary_4x8.csv" \
        "${QUICKLOOK_DIR}/${REGION}_${YEAR}_nu${SMOOTH_TAG}_tile_nugget_summary_4x8.csv"
    fi
    if [[ -f "${OUTDIR}/summary/tile_nugget_median_heatmap_4x8.png" ]]; then
      cp "${OUTDIR}/summary/tile_nugget_median_heatmap_4x8.png" \
        "${QUICKLOOK_DIR}/${REGION}_${YEAR}_nu${SMOOTH_TAG}_tile_nugget_median_heatmap_4x8.png"
    fi
    if [[ -f "${OUTDIR}/summary/tile_nugget_vs_nadir_distance_4x8.csv" ]]; then
      cp "${OUTDIR}/summary/tile_nugget_vs_nadir_distance_4x8.csv" \
        "${QUICKLOOK_DIR}/${REGION}_${YEAR}_nu${SMOOTH_TAG}_tile_nugget_vs_nadir_distance_4x8.csv"
    fi

    echo "Completed region=${REGION}, year=${YEAR}: $(date)"
  done
}

run_region "west_pacific" "lat15to20_lon123to145" "15,20" "123,145" "17.5" "128.2"
run_region "micronesia_open_water" "lat1to6_lon129to139" "1,6" "129,139" "3.5" "128.2"

echo "Finished both regions at $(date)"
```



Submit:

```bash
sbatch fit_sim_july_group_vecchia_tiles_two_regions_smooth0p3_052826.sh
```

## 3. Monitor

```bash
squeue -u jl2815
tail -f /home/jl2815/tco/exercise_output/logs/ow2_gv_nu03_<JOBID>.out

ls -R /home/jl2815/tco/exercise_output/eda/real_data/july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826
```

Expected output folders:

```text
july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826/
  west_pacific/
    tile_median_quicklook/
      west_pacific_2022_nu0p3_tile_nugget_summary_4x8.csv
      west_pacific_2022_nu0p3_tile_nugget_median_heatmap_4x8.png
      west_pacific_2022_nu0p3_tile_nugget_vs_nadir_distance_4x8.csv
      ...
    2022_july_lat15to20_lon123to145/
      manifest_hours.csv
      nu0p3/
        hourly/
        summary/
    ...
  micronesia_open_water/
    tile_median_quicklook/
      micronesia_open_water_2022_nu0p3_tile_nugget_summary_4x8.csv
      micronesia_open_water_2022_nu0p3_tile_nugget_median_heatmap_4x8.png
      micronesia_open_water_2022_nu0p3_tile_nugget_vs_nadir_distance_4x8.csv
      ...
    2022_july_lat1to6_lon129to139/
      manifest_hours.csv
      nu0p3/
        hourly/
        summary/
    ...
```

## 4. Copy Results Back

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data"

scp -r jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/eda/real_data/july_two_open_water_regions_group_vecchia_tiles_smooth0p3_052826 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/eda/real_data/"
```
