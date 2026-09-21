# Ten-day full-data NLL + cumulative Ritz-512x4 test

This fresh test uses July 3, 5, 7, 12, and 15 in both 2024 and 2025. It compares
adapted and fixed lag-6/4/3 Vecchia fits using only the native full-data NLL and
a full-data sparse-precision SLQ/Lanczos diagnostic. The old 400 x 8 dense
eigendecomposition is not computed.

## Why the resolution remains 512

The earlier 512-mode run already resolved each spectral third with roughly 170
modes, while only about 89--93% of selected Ritz pairs passed the configured
5% tail-residual tolerance. Raising the selected count to 1024 would reduce
idealized per-day sampling noise, but it would also require a substantially
larger Krylov candidate space and would emphasize less-converged interior Ritz
pairs. The new test therefore keeps 512 selected modes per start, uses four
independent starts, and reports their mean curve and standard error. Adapted
and fixed receive the same SLQ probes and the same four starting vectors, so
their difference is paired rather than contaminated by different randomness.

## Outputs

- `daily_subplots/`: native NLL plus the four-start mean cumulative curve.
- `daily_band_subplots/`: four-start mean low/middle/high reset curves.
- `2024_07_selected5_average_nll_ritz512x4_cumulative.png`
- `2025_07_selected5_average_nll_ritz512x4_cumulative.png`
- `2024_07_selected5_average_ritz512x4_bands.png`
- `2025_07_selected5_average_ritz512x4_bands.png`
- `daily_ritz512x4_cumulative_curves.csv`: all four replicate curves.
- `daily_nll_ritz512x4_metrics.csv`: replicate-mean metrics and replicate SD.
- `daily_slq_spectrum_curves.csv`
- `daily_slq_three_band_boundaries.csv`
- `daily_fit_results.csv` and native NLL summary outputs.

Each spectral third has weight 1/3. The endpoint is not normalized back to one.
The gray daily envelope is scaled for the mean of four independent starts. It
is a pointwise Gaussian/chi-square heuristic for approximate fitted Ritz modes,
not a formal simultaneous confidence band. Colored ribbons are the standard
error across the four random starts.

## Standardized residual energy

For a covariance eigenpair `Sigma u_j = lambda_j u_j`, the whitened score and
energy are

```text
z_j = (u_j' r) / sqrt(lambda_j)
e_j = z_j^2 = (u_j' r)^2 / lambda_j.
```

The implementation applies the precision operator. With
`omega_j = 1/lambda_j`, the identical energy is evaluated as

```text
e_j = omega_j * (u_j' r)^2.
```

Dividing the projection by `lambda_j` before squaring would incorrectly give
`(u_j' r)^2/lambda_j^2`; that is not the diagnostic used here.

## Upload and submit

From the local `vecchia_approximation` directory:

```bash
bash scp_vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643.sh
```

Then run the submit command printed by that script. The new output root is:

```text
/home/jl2815/tco/exercise_output/summer/vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643
```

This is separate from the earlier three-way output and therefore starts the
requested experiment from scratch. Re-submitting after a time limit or node
failure safely reuses completed date/method caches in this new output root.
