# Run05: one Slurm job on gpu032, no array

This version has no Slurm array and no separate aggregate job.  One allocation
on `gpu032` runs task indices 0 through 30 sequentially and then produces the
aggregate CSV, Markdown, and PNG results before releasing the node.

- node allocations: one
- Slurm jobs: one
- visible GPUs requested: one
- statistical nugget: fixed at zero and excluded from optimization
- time limit: three days, the `gpu` partition limit shown by `sinfo`
- output: `/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_real20240730_synth30_run05_single_gpu032`

If one data task fails, the script records its exit status, continues with the
remaining dates, writes a partial aggregate report at the end, and finally
marks the single Slurm job failed.  Thus one bad date does not erase the other
completed fits.

## Upload

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
  "${LOCAL_DIR}/slurm_vecchia_fulltest_real20240730_synth30_run05_single_gpu032.sh" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit one job

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu '
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation
sbatch slurm_vecchia_fulltest_real20240730_synth30_run05_single_gpu032.sh
'
```

## Monitor

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu 'squeue -u jl2815'
```

The only log pair is:

```text
/home/jl2815/tco/exercise_output/summer/logs/vecc_s30_r05_JOBID.out
/home/jl2815/tco/exercise_output/summer/logs/vecc_s30_r05_JOBID.err
```
