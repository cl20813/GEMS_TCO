# Space-time contrast diagnostic prototype

The coefficients of every implemented linear contrast sum to zero.  In
intrinsic-random-function terminology these are authorized linear combinations
of order 0 (ALC-0).  The even central second differences and the four-corner
mixed rectangle additionally annihilate affine trends (ALC-1).

## What each family targets

- Oriented cross-variogram: broad, intuitive mixed-lag check.  Its h versus -h
  difference targets directional time asymmetry/advection, but the raw level
  still mixes spatial, temporal, and joint dependence.
- Log-correlation interaction: removes fitted/empirical pure-space and
  pure-time correlation margins.  It is zero under multiplicative
  separability, so it targets general nonseparability rather than advection
  alone.  It is nonlinear in empirical covariance estimates.
- Six-direction odd/even basis: D_lat/D_lon crossed with D_time is the signed
  odd-odd sector and is most sensitive to advection.  Q_lat/Q_lon crossed with
  Q_time is the symmetric even-even interaction sector.  Odd-even and
  even-odd sectors are useful leakage/stationarity checks.
- Mixed rectangle: the tensor-product first difference (+1,-1,-1,+1) removes
  any additive space-only plus time-only mean exactly.  Its variance measures
  joint local roughness; comparing +h and -h versions restores directional
  information.

A single six-neighbor 3-D Laplacian is retained only implicitly through the Q
contrasts.  Collapsing Q_lat + Q_lon + Q_time to one number would mix marginal
spatial curvature, marginal temporal curvature, and interaction, so it is not
recommended as the primary interaction diagnostic.

Jackknife standard errors treat 8x8 spatial tiles as clusters.  They are useful
for ranking diagnostic sensitivity in this prototype, but neighboring tiles
remain correlated.  Production inference should use multiple independent days
or a larger moving-block/bootstrap calibration.
