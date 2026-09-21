# Exact covariance versus subset-specific Vecchia: four-way validation

## Construction

- Observations: `3,200` = 400 max-min locations x 8 times
- Parameters: stored full-data adapted lag-4/3/2, batch-64 fit; no refit
- Residual: identical full-data GLS residual in E1, E2, V1, and V2
- Vecchia subset: rebuilt from selected observations and nonempty native-grid blocks; no principal precision submatrix
- Lanczos: precision operator, `m=512`, `32` shared Rademacher probes

## Numerical error (Lanczos only)

- E2 versus E1 cumulative energy/n RMSE: `0.000831021`
- E2 versus E1 spectral-CDF RMSE: `0.00151112`
- V2 versus V1 cumulative energy/n RMSE: `0.000822141`
- V2 versus V1 spectral-CDF RMSE: `0.00128537`
- E2 versus E1 hard 20-band energy/mode RMSE: `0.0420837`
- V2 versus V1 hard 20-band energy/mode RMSE: `0.0276508`

## Vecchia approximation error (full eigen, no Lanczos)

- Real-residual cumulative curve/n RMSE: `0.0116505`
- Real-residual 20-band energy/mode RMSE: `0.128955`
- Covariance relative Frobenius error: `0.164002`
- KL exact-to-Vecchia per observation: `0.00494304`
- Mean expected Vecchia energy if exact K is true: `1.00058`
- Largest absolute expected 20-band deviation from one: `0.113284`

## Exact-K simulation robustness

Across `128` independent residuals generated from exact K:

- Median / 95% curve RMSE per n: `0.0105188` / `0.0141657`
- Median / 95% band-energy RMSE: `0.11972` / `0.154669`

This experiment separates Lanczos truncation/SLQ error from Vecchia spectral
approximation error.  Hard-band error remains larger than the approximately
`sqrt(2/7018)=0.0169` known-parameter fluctuation of a 5% band at full-data
size, so the current full-data hard bands remain exploratory.  Smooth filters
and probe-count convergence are required before calibration.  A final
goodness-of-fit test also requires simulation with beta and covariance
refitting.
