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
- `scp_vecchia_real59_adapted_fixed_nll_lag643.sh`: upload helper.
- `AMAREL_VECCHIA_GPU_OPTIMIZATION_MEMO_090326.txt`: fitting configuration notes.

## Run

```bash
bash scp_vecchia_real59_adapted_fixed_nll_lag643.sh
ssh jl2815@amarel-new.hpc.rutgers.edu \
  'cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation && sbatch slurm_vecchia_real59_adapted_fixed_nll_lag643.sh'
```

If the earlier full-eigen run's fit checkpoint exists, the batch script imports
its completed fits into the new likelihood-only output directory.  Otherwise,
missing date/method fits are computed and checkpointed after every fit.

Main outputs:

- `fit_checkpoint_native_nll.json`
- `daily_fit_results.csv`
- `daily_native_nll.csv`
- `native_nll_summary.csv`
- `daily_native_nll.png`
- `run_config.json`
