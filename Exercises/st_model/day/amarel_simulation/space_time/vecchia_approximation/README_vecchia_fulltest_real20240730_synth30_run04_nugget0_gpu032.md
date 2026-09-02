# Run04: fixed nugget 0 on gpu032 only

This run reuses the deterministic selection of one real day (`2024-07-30`)
and 30 smooth=0.5, nugget=0 synthetic days selected with seed `20260902`.
All four geometries optimize six parameters and keep the statistical nugget
fixed at exactly zero.  The reported seven-parameter vector therefore always
has `nugget_hat=0.0000`.  Small diagonal numerical jitter remains solely for
Cholesky stability and is not an estimated nugget.

The Slurm array is `0-30%1` and is restricted to `gpu032`.  Thus the 31 data
sets run one at a time on that single node, each requesting one GPU.  If
`gpu032` becomes busy or unavailable, the jobs wait rather than moving to
another GPU node.

The fresh result root is:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_real20240730_synth30_run04_nugget0_gpu032
```

## Upload from the local Mac

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
  "${LOCAL_DIR}/slurm_vecchia_fulltest_real20240730_synth30_run04_nugget0_gpu032.sh" \
  "${LOCAL_DIR}/submit_vecchia_fulltest_real20240730_synth30_run04_nugget0_gpu032.sh" \
  "${LOCAL_DIR}/README_vecchia_fulltest_real20240730_synth30_run04_nugget0_gpu032.md" \
  "jl2815@amarel.rutgers.edu:${REMOTE_DIR}/"
```

## Submit

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu \
  'bash /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation/submit_vecchia_fulltest_real20240730_synth30_run04_nugget0_gpu032.sh'
```

## Pull results

```bash
mkdir -p "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26"
scp -r -C -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_four_geometry_lag643_real20240730_synth30_run04_nugget0_gpu032 \
  "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/"
```
