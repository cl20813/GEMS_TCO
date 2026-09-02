# Real 30 + synthetic 30: three-geometry lag643 run07

This run compares `adapted`, `fixed`, and their exact `union` using 4x4 target
blocks and 6/4/3 conditioning-block budgets. `shifted` is intentionally absent.

- Real data: July 1-30, 2024 (30 complete eight-slot days).
- Synthetic data: 30 complete days already present on Amarel, sampled as 10
  days per year for 2023-2025 with seed `20260902`.
- Synthetic truth: Matérn smoothness 0.5 and nugget 0.
- Nugget fitting: disabled. The optimizer receives the truth value 0 and the
  covariance classes keep it fixed at 0; diagonal jitter is numerical only.
- Initialization: the same daily M3 masked-FFT plus safeguarded Q3 advection
  seed is shared by all three methods.
- Output: daily plots for every dataset and separate 30-day real/synthetic mean
  plots under `comparison_report/plots/`.

## Why this avoids the earlier errors

The Slurm script calls `/home/jl2815/.conda/envs/faiss_env/bin/python`
directly and does not call `module`, so the Lmod Lua `posix` failure is bypassed.
The 60 array tasks are throttled to one GPU task at a time. Each task is a fresh
Python process, the union now has only two components, and union fit/diagnostic
chunks are 4/8, substantially reducing peak GPU memory relative to the failed
three-way union.

## Upload from the Mac

The helper uploads code only; it does not submit anything:

```bash
bash /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation/scp_vecchia_real30_synth30_three_geometry_lag643_run07.sh
```

## Verify the Amarel inputs

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu jl2815@amarel.rutgers.edu '
test -f /home/jl2815/tco/data/pickle_2024/tco_grid_24_07.pkl
for y in 2023 2024 2025; do
  d=/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_oneday_070926/${y}_july_st_circulant
  test -f ${d}/sim_july${y}_st_circulant_gridded.pkl
  test -f ${d}/sim_july${y}_st_circulant_truth.json
done
/home/jl2815/.conda/envs/faiss_env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo inputs-ok
'
```

## Submit

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu jl2815@amarel.rutgers.edu \
  'bash /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation/submit_vecchia_real30_synth30_three_geometry_lag643_run07.sh'
```

The fit job is a `0-59%1` array. The dependent aggregate job runs only after all
60 tasks succeed, preventing an incomplete mean plot from being mistaken for a
final result.

## Monitor

```bash
ssh -o HostKeyAlias=amarel-new.hpc.rutgers.edu jl2815@amarel.rutgers.edu \
  "squeue -u jl2815 -o '%.18i %.24j %.8T %.10M %.6D %R'"
```

## Pull results

```bash
mkdir -p /Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26
scp -r -C -o HostKeyAlias=amarel-new.hpc.rutgers.edu \
  jl2815@amarel.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_three_geometry_lag643_real30_synth30_run07_nugget0_serial \
  /Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/
```

Principal outputs are `all_fits.csv`, `all_cross_likelihoods.csv`,
`comparison_report/vecchia_comparison_summary.csv`, 60 daily conditional-eigen
plots, and `comparison_report/plots/mean/{real,synthetic}_mean_conditional_eigen.png`.
