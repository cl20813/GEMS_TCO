# Fitted-model two-contrast variance diagnostic

Let `L = d1*Q_A + d2*Q_B`. The primary comparison is

```text
empirical Var(L) - fitted-model Var(L)
```

and that difference is attributed exactly as

```text
[d1^2 empirical Var(Q_A) - d1^2 fitted Var(Q_A)]
+ [d2^2 empirical Var(Q_B) - d2^2 fitted Var(Q_B)]
+ [2*d1*d2 empirical Cov(Q_A,Q_B) - 2*d1*d2 fitted Cov(Q_A,Q_B)].
```

Thus the output determines whether a total variance discrepancy is specifically driven by the cross term. This cross term is the covariance contribution inside `Var(L)`, not `Var(Q_A*Q_B)`. Because all three are zero-sum contrasts, known-zero second moments are primary; centered empirical versions are retained as sensitivity columns. Simulation truth is only an evaluation reference.

## Frozen fitting rule

- Matérn simulations: fitted joint Matérn `nu=0.5`.
- Generalized-Cauchy simulations: fitted joint GC `a=1, b=5`.
- Nugget fixed at zero.
- Data-only M3/Q3 advection initializer.
- Direction-adapted 4x4 corridor Vecchia with lag `6/4/3`.
- Matérn chunk 256; GC chunk 128.
- Every simulated day is fitted independently. The study therefore contains 180 fits.
- Eta 1 is the correctly specified joint endpoint. Eta 0 and 0.5 intentionally test interaction misspecification after nuisance fitting.
- One non-array A100 job processes all work sequentially and writes a day JSON plus refreshed scenario/master CSV after every completed fit.
- Resubmission verifies source hashes and reuses compatible completed fits/days.

## Local validation

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh validate-local
```

## Recommended Amarel sequence

Run a six-scenario one-day pilot first:

```bash
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh upload-submit-pilot
```

After inspecting convergence and GPU memory, submit/restart the full 180-fit job:

```bash
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh submit-full
```

The default full allocation is one A100, 12 CPUs, 128 GiB system RAM, and 12 hours. The job is restartable, so expiration does not discard completed fits; rerun `submit-full` until `FINAL_COMPLETE.json` is present.

Monitor and download:

```bash
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh progress
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh logs
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/sim_two_contrast_fitted_cross_643_092826/amarel_fitted_cross_643.sh download
```

## Final outputs

- `all_daily_fitted_cross_centers.csv`
- `all_daily_fitted_cross_strata.csv`
- `scenario_fitted_cross_summary.csv`
- `paired_eta_fitted_residual_effects.csv`
- `fitted_vs_empirical_variance_decomposition.png`
- `RESULTS.md`
- `FINAL_COMPLETE.json`
