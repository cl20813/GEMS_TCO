# Real 2024-07-30 + 30 synthetic days — four-geometry Vecchia run03

This run uses one fixed real-data day and 30 reproducibly sampled synthetic
days.  The synthetic sample is stratified as 10 complete eight-slot July days
per year.  The random seed is `20260902`.

- real: `2024-07-30`
- synthetic 2023: `07-03, 07-05, 07-11, 07-13, 07-16, 07-22, 07-26, 07-28, 07-29, 07-31`
- synthetic 2024: `07-03, 07-06, 07-10, 07-13, 07-16, 07-20, 07-22, 07-24, 07-25, 07-26`
- synthetic 2025: `07-06, 07-07, 07-09, 07-12, 07-14, 07-17, 07-19, 07-21, 07-22, 07-30`

The synthetic asset is required to have `smooth=0.5` and `nugget=0` in its
truth JSON.  The optimizer starts a zero-truth nugget at the finite positive
value `0.2470`, so the seventh parameter can be estimated rather than being
stuck at `log(0)`.  Truth likelihood evaluation still uses an effectively zero
nugget.  The exact Amarel root shown in the reference screenshot is used:

```text
/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926
```

Every data set fits `union`, `adapted`, `shifted`, and `fixed`, in that order.
The unique result root is:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_real20240730_synth30_run03
```

The array has 31 tasks (`0` is real and `1–30` are synthetic).  Each task uses
one node and one GPU; `%6` limits the run to at most six concurrent nodes.
Giving multiple nodes to one task would not accelerate the implementation.

## Upload

Run locally:

```bash
LOCAL_ROOT="/Users/joonwonlee/Documents/GEMS_TCO-1"
LOCAL_DIR="${LOCAL_ROOT}/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation"
REMOTE_DIR="/home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation"

ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu \
  "mkdir -p ${REMOTE_DIR} /home/jl2815/tco/exercise_output/summer/logs"

scp -r -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  "${LOCAL_ROOT}/src/GEMS_TCO" \
  "jl2815@amarel.rutgers.edu:/home/jl2815/tco/"

scp -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  "${LOCAL_DIR}/vecchia_adapted_vs_fixed_lag643_090126.py" \
  "${LOCAL_DIR}/vecchia_fulltest_real20240730_synth30_run03_selection.json" \
  "${LOCAL_DIR}/slurm_vecchia_fulltest_real20240730_synth30_run03.sh" \
  "${LOCAL_DIR}/README_vecchia_fulltest_real20240730_synth30_run03.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit

`afterany` lets the aggregate task write a partial report and failure-stage
table even if one fit task fails.

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu '
set -e
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation
fit_job=$(sbatch --parsable slurm_vecchia_fulltest_real20240730_synth30_run03.sh)
aggregate_job=$(sbatch --parsable \
  --dependency=afterany:${fit_job} \
  --array=0-0 \
  --export=ALL,RUN_MODE=aggregate \
  slurm_vecchia_fulltest_real20240730_synth30_run03.sh)
echo fit_job=${fit_job}
echo aggregate_job=${aggregate_job}
'
```

## Results

The clean output tables and plots are below `comparison_report/`:

- `vecchia_comparison_summary.csv`: one row per date and method, identifiers
  first, then two initialized advection values, likelihoods, all seven fitted
  parameters, fit/precompute/diagnostic seconds, and eigen/truth diagnostics;
- `vecchia_comparison_report.md`: compact human-readable tables;
- `plots/daily/`: one four-color conditional-eigen PNG per date;
- `plots/mean/`: separate real and synthetic mean four-color PNGs.

The exact real and synthetic dates are also written to `selected_dates.csv` at
the result root.

Human-facing numeric output is rounded to four decimal places.  Raw per-task
files retain full precision.  A task failure is recorded as
`FAILED_task_XX.json`, and every task writes `PROGRESS.json` while running.

## Pull run03

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"
scp -r -C -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_real20240730_synth30_run03 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
