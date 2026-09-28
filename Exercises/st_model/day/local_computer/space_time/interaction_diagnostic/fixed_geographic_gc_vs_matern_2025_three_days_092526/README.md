# July 2025 local GC versus Matérn audit

This local CPU audit applies the already frozen fixed-geographic diagnostic to
`2025-07-07`, `2025-07-15`, and `2025-07-23`. July 7 and 15 were specified by
the user; July 23 was selected with seed `20250925` from complete late-July
manifest dates. No contrast, lag, coefficient, or covariance parameter is
selected from these outcomes.

The computation uses corridor 4/3/2, 4x4 target blocks, chunk 64, float64,
nugget fixed at zero, exact native generalized Cauchy, and exact exponential
Matérn with smoothness 0.5. Only GC and Matérn are fitted.

Run from the repository root:

```bash
/opt/anaconda3/envs/faiss_env/bin/python \
  Exercises/st_model/day/local_computer/space_time/interaction_diagnostic/fixed_geographic_gc_vs_matern_2025_three_days_092526/run_local_gc_vs_matern_2025.py
```

The primary comparison is the frozen bivariate Gaussian contrast score; lower
is better. The output also reports empirical and fitted `C_AB`, `Var(L)`, and
the raw Frobenius discrepancy between fitted and empirical 2x2 second-moment
matrices. These are descriptive three-day results, not calibrated inferential
evidence.
