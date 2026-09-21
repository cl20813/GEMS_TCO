# Exact covariance versus subset-specific Vecchia: four-way validation

## Construction

- Observations: `3,200` = 400 locations x 8 times
- Spatial selection: `central fully common-valid native-grid rectangle`
- Parameters: stored full-data adapted lag-4/3/2, batch-64 fit; no refit
- Residual: identical full-data GLS residual in E1, E2, V1, and V2
- Vecchia subset: rebuilt from selected observations and nonempty native-grid blocks; no principal precision submatrix
- Lanczos: precision operator, `m=512`, `32` shared Rademacher probes

## Numerical error (Lanczos only)

- E2 versus E1 cumulative energy/n RMSE: `0.000588772`
- E2 versus E1 spectral-CDF RMSE: `0.00235967`
- V2 versus V1 cumulative energy/n RMSE: `0.000604996`
- V2 versus V1 spectral-CDF RMSE: `0.00216948`
- E2 versus E1 hard 20-band energy/mode RMSE: `0.021451`
- V2 versus V1 hard 20-band energy/mode RMSE: `0.0213199`

## Vecchia approximation error (full eigen, no Lanczos)

- Real-residual cumulative curve/n RMSE: `0.00156352`
- Real-residual 20-band energy/mode RMSE: `0.0429808`
- Covariance relative Frobenius error: `0.08971`
- KL exact-to-Vecchia per observation: `0.00138867`
- Mean expected Vecchia energy if exact K is true: `1.0014`
- Largest absolute expected 20-band deviation from one: `0.0313462`

## Exact-K simulation robustness

Across `128` independent residuals generated from exact K:

- Median / 95% curve RMSE per n: `0.00311178` / `0.00459914`
- Median / 95% band-energy RMSE: `0.0668547` / `0.091128`

This experiment separates Lanczos truncation/SLQ error from Vecchia spectral
approximation error.  Hard-band error remains larger than the approximately
`sqrt(2/7018)=0.0169` known-parameter fluctuation of a 5% band at full-data
size, so the current full-data hard bands remain exploratory.  Smooth filters
and probe-count convergence are required before calibration.  A final
goodness-of-fit test also requires simulation with beta and covariance
refitting.
