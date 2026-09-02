# 2024-07-30 four-geometry Vecchia full test — run02

This targeted run validates both input branches on the same date:

- task 0: real GEMS TCO, 2024-07-30
- task 1: smooth=0.5 synthetic data, 2024-07-30

Each task fits `union`, `adapted`, `shifted`, and `fixed`, in that order.  The
union fit runs first with smaller GPU chunks so a union-specific failure cannot
be hidden behind three earlier fits or cumulative allocator fragmentation.

The unique remote result root is:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_20240730_run02
```

The driver refuses to write into a nonempty task directory.  Use `run03` in
both the Slurm file and result path for another fresh attempt instead of
overwriting run02.

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
  "${LOCAL_DIR}/vecchia_fulltest_20240730_run02_selection.json" \
  "${LOCAL_DIR}/slurm_vecchia_fulltest_20240730_run02.sh" \
  "${LOCAL_DIR}/README_vecchia_fulltest_20240730_run02.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Confirm the two possible smooth=0.5 synthetic roots

The original failed run stopped all synthetic tasks before `asset_manifest`.
The updated loader first checks the explicit `_smooth0p5` root and then the
generator's original `july_st_circulant_realpattern` root.  It accepts a
fallback only when the truth JSON says `smooth=0.5`.

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu '
for root in \
  /home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5 \
  /home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern
do
  echo "ROOT=${root}"
  ls -lh \
    "${root}/2024_july_st_circulant/sim_july2024_st_circulant_gridded.pkl" \
    "${root}/2024_july_st_circulant/sim_july2024_st_circulant_truth.json" \
    2>&1 || true
done
'
```

## Submit

`afterany` is deliberate: aggregation writes a clearly labelled partial
report and failure-stage table even if one fit task fails.

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu '
set -e
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation
fit_job=$(sbatch --parsable slurm_vecchia_fulltest_20240730_run02.sh)
aggregate_job=$(sbatch --parsable \
  --dependency=afterany:${fit_job} \
  --array=0-0 \
  --export=ALL,RUN_MODE=aggregate \
  slurm_vecchia_fulltest_20240730_run02.sh)
echo fit_job=${fit_job}
echo aggregate_job=${aggregate_job}
'
```

The array contains two tasks, so it may use two one-GPU nodes concurrently.
The implementation does not use multiple GPUs within one fit; requesting more
nodes or GPUs for a single task would not accelerate or repair union.

## Output contract

`comparison_report/vecchia_comparison_summary.csv` has one row per data set
and method.  Its leading identifiers are followed by:

- the two M3+Q3 initial advection components;
- native and common-union likelihoods;
- all seven fitted parameters;
- precompute, optimizer fit, eigen diagnostic, and total method seconds;
- conditional-eigen diagnostics and synthetic truth errors.

All human-facing summary CSV and Markdown numeric values use four decimal
places.  Raw per-task numerical files retain full precision for reproducible
analysis.  PNG outputs are written below:

```text
comparison_report/plots/daily/
comparison_report/plots/mean/
task_*/conditional_eigen_four_geometries.png
```

Every task also writes `PROGRESS.json`.  On failure the output root contains
`FAILED_task_XX.json`, including the precise Python error, traceback, requested
device, GPU name, and allocated/reserved CUDA memory.

## Pull run02

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"
scp -r -C -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_20240730_run02 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
