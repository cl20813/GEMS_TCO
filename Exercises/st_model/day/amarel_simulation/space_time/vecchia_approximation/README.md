# Adapted vs fixed Vecchia likelihood comparison

This directory now keeps one production analysis: adapted versus fixed
lag-6/4/3 Vecchia likelihoods for the 59 usable July dates in 2024 and 2025.
The incomplete 2025-07-24 record is excluded.

`native Vecchia NLL` means the negative log likelihood used to fit each method
on that method's own conditioning graph.  It is computed from all valid target
observations.  It is not a dense full-Gaussian-process likelihood.  Lower NLL
is better when comparing the two methods on the same date.

`daily_native_nll.csv` reports `fixed_minus_adapted_total_nll`; a positive
value means the adapted fit has the lower native Vecchia NLL on that date.

No eigen, SLQ, Lanczos, or Ritz diagnostic is run.

## Files

- `vecchia_adapted_fixed_lag643_core.py`: fitting backend.
- `vecchia_real59_adapted_fixed_nll_lag643.py`: likelihood-only production run.
- `slurm_vecchia_real59_adapted_fixed_nll_lag643.sh`: Amarel batch job.
- `run_vecchia_real59_gpu_on_amarel.sh`: upload, submit, status, and download helper.
- `verify_vecchia_real59_nll_results.py`: completion and row-count validator.
- `AMAREL_VECCHIA_GPU_OPTIMIZATION_MEMO_090326.txt`: fitting configuration notes.

## Fresh GPU rerun

The rerun requests one A100-class GPU, 12 CPU cores, 128 GB of RAM, and a
6-hour wall time.  It writes to a new output directory and does not import the
older full-eigen checkpoint.  If this new job is interrupted, submitting the
same batch file again resumes from its own per-fit checkpoint.

Both fitted graphs use the same 4x4 target blocks and the same lag-6/4/3
conditioning-block budget.  The adapted graph uses the FFT initializer's
reference advection vector; the fixed graph uses a zero reference vector for
neighbor selection.  The covariance advection parameters are optimized in
both fits.

From the local Mac:

```bash
bash run_vecchia_real59_gpu_on_amarel.sh submit
bash run_vecchia_real59_gpu_on_amarel.sh status
```

Each command opens one non-multiplexed Amarel connection and should request
the password once.  The submit connection remains open while it uploads,
installs, verifies, and calls `sbatch`; do not run Amarel `/home/...` paths
with a local `cd` command.

If the current source is already installed and uploaded on Amarel, submit
directly there with:

```bash
mkdir -p /home/jl2815/tco/exercise_output/fall_26/logs
cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation
sbatch slurm_vecchia_real59_adapted_fixed_nll_lag643.sh
```

After `RUN_COMPLETE.json` appears, download and validate the results locally:

```bash
bash run_vecchia_real59_gpu_on_amarel.sh pull
```

The new remote output directory is:

```text
/home/jl2815/tco/exercise_output/fall_26/vecchia_real59_adapted_fixed_nll_lag643_rerun_20260927
```

Main outputs:

- `fit_checkpoint_native_nll.json`
- `daily_fit_results.csv`
- `daily_native_nll.csv`
- `native_nll_summary.csv`
- `daily_winner_summary.json`
- `paper_daily_comparison.csv`
- `paper_nll_summary.csv`
- `paper_nll_summary.tex`
- `daily_native_nll.png`
- `run_config.json`
- `RUN_COMPLETE.json`
