# Point-maxmin full-covariance eigen-whitening diagnostic

For each of eight hours, all valid points are max-min ordered at point level and
the first 400 are retained. The combined diagnostic has
3200 observations, shared by all four fitted methods.

The fitted mean design is the same column space used by the lag-432 fits:
intercept, centered latitude, and seven hourly dummy variables. If `Q` spans
this design, `R = I - QQ'`. For each method the diagnostic decomposes

`R Sigma_hat R = S Lambda S'`

and computes

`Y = Lambda^(-1/2) S' R z`.

Thus the denominator is the square root of the eigenvalue, not the eigenvalue.
Under a correctly specified Gaussian covariance, the retained `Y_j` are
approximately iid N(0,1), so cumulative `Y_j^2` should follow the identity line.
Eigenpairs are ordered from the largest fitted eigenvalue to the smallest.

The fitted parameters were estimated from the same day, so the plotted 95%
bands are descriptive diagnostics rather than exact hypothesis tests. All four
curves use the same selected points; differences arise only from their fitted
covariance parameters. The statistical nugget is taken from each fitted row
(zero in this experiment), with numerical covariance jitter
`1.00e-08` added to the diagonal.
