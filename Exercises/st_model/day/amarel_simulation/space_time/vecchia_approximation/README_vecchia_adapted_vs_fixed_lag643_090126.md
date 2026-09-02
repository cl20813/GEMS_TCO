# Four-geometry lag643 Vecchia comparison

This experiment compares four fixed conditioning graphs on five real July
days and five smooth=0.5 synthetic July days. Every graph uses the same M3
masked-FFT plus safeguarded Q3 advection initializer and the same optimizer
start/budget.

- `adapted`: 6/4/3 blocks; the M3+Q3-calibrated signed 2-D corridor spans
  `target-[0.5,1.5]*v_hat` at t-1 and `target-[0,2]*v_hat` at t-2.
- `shifted`: 6/4/3 blocks; four nearest blocks around `target-v_hat` at t-1
  and three nearest blocks around `target-2*v_hat` at t-2.
- `fixed`: 6/4/3 blocks; both past centers remain at the target.
- `union`: exact deduplicated union of the preceding three sets, up to
  6/12/9 blocks.

The sign is intentional. The covariance is parameterized by `h-v*tau`, so the
past conditioning location for a current target is displaced by `-v*tau`.

The original `vecchia_realdata_corridor_width_4x4_lag643.py` uses a positive
longitude-only fixed `delta=0.126`.  The new
`vecchia_realdata_adapted_corridor_width_4x4_lag643.py` preserves its literal
width geometry (`[0.5v,1.5v]` and `[0,2v]`) while replacing that scalar with the
M3+Q3-calibrated signed two-dimensional vector.

`vecchia_realdata_calibrated_shifted_center_4x4_lag643.py` implements the
calibrated one-point centers at `v` and `2v`. In both modules, the past
conditioning offsets are `-v` and `-2v` because the covariance uses `h-v*tau`.

## Reproducible selections

The exact selections and seeds are stored in
`vecchia_adapted_vs_fixed_selection_090126.json`. The driver copies the
selection and source hourly keys into every task output; aggregation also writes
`selected_dates.csv` and includes the dates in `REPORT.md`.

## Upload to Amarel

Run on the local Mac:

```bash
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_SRC="/home/jl2815/tco/GEMS_TCO"

ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu \
  "mkdir -p ${REMOTE_DIR} ${REMOTE_SRC} /home/jl2815/tco/exercise_output/summer/logs"

scp -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  "${LOCAL_ROOT}/src/GEMS_TCO/vecchia_cluster.py" \
  "${LOCAL_ROOT}/src/GEMS_TCO/vecchia_realdata_corridor_width_4x4_lag643.py" \
  "${LOCAL_ROOT}/src/GEMS_TCO/vecchia_realdata_adapted_corridor_width_4x4_lag643.py" \
  "${LOCAL_ROOT}/src/GEMS_TCO/vecchia_realdata_calibrated_shifted_center_4x4_lag643.py" \
  "jl2815@amarel.rutgers.edu:${REMOTE_SRC}/"

scp -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  "${LOCAL_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py" \
  "${LOCAL_DIR}/vecchia_adapted_vs_fixed_selection_090126.json" \
  "${LOCAL_DIR}/slurm_vecchia_adapted_vs_fixed_lag643_090126.sh" \
  "${LOCAL_DIR}/README_vecchia_adapted_vs_fixed_lag643_090126.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Verify remote inputs

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel-new.hpc.rutgers.edu '
set -e
test -f /home/jl2815/tco/data/pickle_2023/tco_grid_23_07.pkl
test -f /home/jl2815/tco/data/pickle_2024/tco_grid_24_07.pkl
test -f /home/jl2815/tco/data/pickle_2025/tco_grid_25_07.pkl
for y in 2023 2024 2025; do
  d=/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5/${y}_july_st_circulant
  test -f ${d}/sim_july${y}_st_circulant_gridded.pkl
  test -f ${d}/sim_july${y}_st_circulant_truth.json
done
echo inputs-ok
'
```

## Submit fits and dependent aggregation

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu '
set -e
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation
fit_job=$(sbatch --parsable slurm_vecchia_adapted_vs_fixed_lag643_090126.sh)
aggregate_job=$(sbatch --parsable \
  --dependency=afterok:${fit_job} \
  --array=0-0 \
  --export=ALL,RUN_MODE=aggregate \
  slurm_vecchia_adapted_vs_fixed_lag643_090126.sh)
echo fit_job=${fit_job}
echo aggregate_job=${aggregate_job}
'
```

Monitor:

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu \
  "squeue -u jl2815 -o '%.18i %.24j %.8T %.10M %.6D %R'"
```

## Main outputs

- `selected_dates.csv`, `REPORT.md`
- `all_initializers.csv`
- `all_fits.csv`
- `all_cross_likelihoods.csv`
- `union_reference_likelihood_gaps.csv` and `.png`
- `all_eigen_summaries.csv`
- `conditional_eigen_mean_four_geometries.png`, with all four colors and mean
  native/union-evaluated likelihoods in the legend
- `synthetic_parameter_errors.png`
- per-dataset task folders with exact source keys and a four-curve eigen plot;
  each legend records that fit's native and union-evaluated likelihoods

Native NLL values from different graphs are retained, but the primary
likelihood comparison uses the common union graph: evaluate the adapted,
shifted, and fixed fitted parameter vectors under the union conditioning set
and compare all three with the union fit.

## Pull results

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"
scp -r -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_090126 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
