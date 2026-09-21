# Adapted vs fixed: full eigen + likelihood + SLQ/Ritz-512

## What this run compares

The nominal 60-day window is July 1--30 in 2024 and 2025. The incomplete
2025-07-24 record has seven rather than eight hourly slots, so 59 dates are
actually analyzed. Each adapted/fixed lag-6/4/3 fit is compared in three ways:

1. dense full eigendecomposition on 400 max-min points per hour (3,200 total),
2. native Vecchia negative log likelihood per observation,
3. full-data sparse-precision SLQ and 512 selected implicit Ritz modes.

SLQ estimates the empirical mode-count CDF and places the band boundaries at
its 1/3 and 2/3 quantiles. Thus the low, middle, and high spectral bands each
contain about one third of all modes before 170, 170, and 172 representatives
are selected. Small-to-large precision eigenvalue is used as a low-to-high
frequency proxy; it is not an exact Fourier frequency on this irregular
space-time Vecchia graph.

The 512 modes are random-start Rayleigh--Ritz approximations. The code does not
silently treat all of them as converged eigenvectors: absolute and relative
tail Ritz-residual estimates are saved for every selected mode, and the plots
report the fraction below the configured quality tolerance.

## Upload and submit

From the local `vecchia_approximation` directory:

```bash
bash scp_vecchia_real59_adapted_fixed_threeway_slq512_lag643.sh
ssh jl2815@amarel-new.hpc.rutgers.edu \
  'cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation && sbatch slurm_vecchia_real59_adapted_fixed_threeway_slq512_lag643.sh'
```

The SLURM allocation is eight hours, 12 CPU cores, 128 GB RAM, and one GPU.
The same job can be submitted again after interruption: completed date/method
diagnostics and fitted parameters are checkpointed and reused.

The job excludes `gpu017`, which returned a CUDA initialization error on
2026-09-08. It runs a CUDA preflight before analysis and launches the single
Python task directly inside the batch allocation to avoid a second `srun`
device remapping.

## Main outputs

- `daily_subplots/YYYY-MM-DD_threeway.png`: one three-panel plot per date.
- `2024_07_monthly_average_threeway.png` and
  `2025_07_monthly_average_threeway.png`: monthly averages in the output root.
- `daily_selected_512_ritz_modes.csv`: all selected modes and quality metrics.
- `daily_slq_three_band_boundaries.csv`: data-driven band limits and count SEs.
- `daily_lanczos512_metrics.csv`: per-date/method band energy and Ritz quality.
- `daily_full_eigen_metrics.csv`, `daily_native_nll.csv`, and fit parameters.

The Amarel output directory is:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_real59_adapted_fixed_threeway_slq512_lag643
```

Download it after the job finishes:

```bash
scp -r jl2815@amarel-new.hpc.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_real59_adapted_fixed_threeway_slq512_lag643 \
  /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation/
```
