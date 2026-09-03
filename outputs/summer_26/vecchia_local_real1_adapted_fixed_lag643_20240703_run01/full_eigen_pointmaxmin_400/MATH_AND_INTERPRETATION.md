# Point-maxmin full-covariance eigen-whitening diagnostic

For each of eight hours, all valid points are max-min ordered at point level and
the first 400 are retained. The combined diagnostic has
3200 observations, shared by both fitted methods.

The fitted mean design is the same one used by the lag-643 fits: intercept,
centered latitude, and seven hourly dummy variables. For every method the
diagnostic directly decomposes the full 3,200 by 3,200 fitted covariance:

`Sigma_hat = S Lambda S'`


`y_star = Lambda^(-1/2) S' y`,

`X_star = Lambda^(-1/2) S' X`,

`beta_hat = (X_star' X_star)^(-1) X_star' y_star`, and

`e = y_star - X_star beta_hat`.

Thus the denominator is the square root of the eigenvalue, not the eigenvalue.
All 3,200 covariance eigenpairs are retained and ordered from largest fitted
eigenvalue to smallest. The plotted observed curve is cumulative `e_j^2`; its
mean-estimation-adjusted expected curve is cumulative `1-h_j`, where `h_j` is
the GLS leverage of eigen-coordinate `j`. Both are divided by residual df, so a
well-calibrated fitted covariance should follow the identity line.

The fitted parameters were estimated from the same day, so the plotted 95%
bands are descriptive diagnostics rather than exact hypothesis tests. Both
curves use the same selected points; differences arise only from their fitted
covariance parameters. The statistical nugget is taken from each fitted row
(estimated separately for each real-data fit), with numerical covariance jitter
`1.00e-08` added to the diagonal.
