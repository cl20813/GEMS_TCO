# Full eigen versus Lanczos/SLQ on a real 8 x 400 subset

## Scope

This run uses 400 max-min ordered spatial locations that are valid in all
eight hours of 2024-07-03 (`n=3,200`).  The covariance parameters are the
stored adapted lag-4/3/2, batch-64 fit.  No covariance refit was performed.
GLS beta was recomputed at the stored parameters, and both numerical methods
use exactly the same raw residual and exact fitted covariance.

The full eigendecomposition is the reference.  Lanczos approximates the
residual-weighted inverse-covariance spectral measure, while SLQ estimates the
spectral mode count.  The largest run uses `512`
Lanczos steps and `32` fixed-seed Rademacher probes.

## Largest-run agreement

- Spectral-CDF mode-fraction RMSE: `0.00352067`
- Spectral-CDF maximum absolute error: `0.00796359`
- Cumulative energy/n RMSE: `0.00130667`
- Cumulative energy/n maximum absolute error: `0.00548101`
- Endpoint energy relative error: `0`
- Exact versus matrix-free shape D: `3.59099` versus
  `3.73397`

## Interpretation boundary

The Lanczos code accesses the covariance only through a matvec callback.  For
this validation run that callback uses the already materialized dense matrix,
because the dense matrix is required for the full-eigen reference.  A separate
streamed-kernel callback, which never stores the full covariance, agreed with
the dense callback to relative error
`2.319e-16`.

Thus this run validates the numerical Lanczos/SLQ formulation on the exact
fitted covariance.  It is not yet the scalable 144,000-point implementation:
the next operator should be the sparse adapted-Vecchia precision
`v -> A.T @ D^-1 @ (A @ v)`.  That next comparison will additionally contain
Vecchia approximation error, whereas this one intentionally does not.

Hard spectral thresholds are discontinuous, so convergence is slowest near
eigenvalue clusters and at very small-rank tails.  Smooth overlapping filters
should be added after this exact-reference test is accepted.
