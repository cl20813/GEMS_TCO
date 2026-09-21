# Full real-data matrix-free Vecchia eigen diagnostic

## Construction check

The fitted adapted lag-4/3/2 block conditionals were stacked into a sparse
whitening matrix `B`, with precision applied only as `B.T @ (B @ v)`.

- Valid observations: `140,352`
- B nonzeros: `18,387,361`
- Mean nonzeros per row: `131.01`
- CSR storage: `210.96 MiB`
- Native-vs-sparse quadratic relative error: `2.227e-11`

## 8 x 400 sparse-operator validation

Full eigendecomposition of the 3,200-variable principal precision was compared
with Lanczos/SLQ using the sparse `B` operator.  At
`m=512`:

- cumulative energy/n RMSE: `0.0241622`
- energy/n maximum error: `0.0801529`
- spectral-CDF RMSE: `0.00598456`
- spectral-CDF maximum error: `0.0197382`
- endpoint energy relative error: `1.821e-14`

This validates Lanczos on the actual fitted sparse Vecchia precision, not only
on a dense exact covariance callback.  The principal precision is a
conditional-subset object; it is not the marginal precision of the 3,200-point
subset and is used only as a numerical reference.

## Full-data diagnostic

The full curve used `m=512` and
`12` Rademacher SLQ probes.

- Mean standardized energy: `1`
- Scale statistic: `0.00087879`
- Shape D: `6.32281`
- Highest-energy 5% band: band `9`, energy/mode
  `1.13198`
- First two large-covariance bands: `0.811277`, `0.864604`
- Final three small-covariance bands: `0.870941, 0.865857, 0.891393`
- Relative to m=512, m=256 has cumulative energy/n RMSE `0.00109538`
  and spectral-CDF RMSE `0.000723235`

The fitted scale makes the endpoint mean close to one by construction; the
non-uniform distribution across eigenvalue bands is the informative feature.
The broad low-high-low pattern is visible at both m=256 and m=512, but the
hard-projector maximum `D` is more sensitive to Lanczos order.  These values
remain exploratory: beta and covariance parameters were fitted on the same
day, so neither the reference line nor `D` is calibrated for estimation
leverage.  The next statistical step is a parametric bootstrap that repeats
fitting and the same matrix-free diagnostic.
