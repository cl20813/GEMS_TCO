# Advection-Ridge Semivariogram Diagnostic

Date: 2026-05-25

This note proposes a diagnostic plot to replace or supplement the current
latitude-only and longitude-only empirical semivariogram slices.

## Short Answer

The advection-ridge diagnostic is probably the best primary diagnostic for the
current problem, but it is not the only diagnostic we should rely on.

It is best as a first diagnostic because it directly targets the feature that
the fitted model claims to estimate:

```text
effective spatial lag = spatial lag - advection * time lag
```

The current semivariogram plots are functions of one spatial lag at a time.
They can show asymmetry, but they cannot clearly separate spatial range,
temporal range, and advection. A 2D lag-surface across multiple temporal lags
can.

## Why The Current Plot Is Limited

The current empirical cross-semivariogram is already close to the right object:

```text
gamma_cross(h, tau=1)
  = 0.5 * mean{ Z(s + h, t) - Z(s, t + 1) }^2
```

In the code this appears as:

```text
diffs = cur_vals[idx_col] - next_vals[idx_row]
```

This contains the advection signal. If a plume or TCO field moves by `a` grid
units per hour, the minimum cross-semivariance should occur near `h = a`.

The problem is that the stored plots use only:

```text
(h_lat, 0) or (0, h_lon)
```

But the fitted model uses a distance like:

```text
d_ST^2 =
  ((h_lat - a_lat * tau) / range_lat)^2
+ ((h_lon - a_lon * tau) / range_lon)^2
+ (tau / range_time)^2
```

Therefore the model is a function of a transported 3D lag, but the diagnostic
is a function of one coordinate slice. This makes it easy to misread advection
as spatial anisotropy, temporal range, or nonstationarity.

## Proposed Diagnostic

Compute the empirical surface

```text
gamma_hat(h_lat, h_lon, tau)
  = 0.5 * mean{ r(s, t) - r(s + h, t + tau) }^2
```

where `r` should usually be residual TCO after the chosen mean/detrending step.

For each `tau` in `1, 2, 3, ...`, plot a heatmap over the 2D spatial lag plane:

```text
x axis: h_lon
y axis: h_lat
color: gamma_hat(h_lat, h_lon, tau)
```

Overlay:

```text
1. empirical minimum or smoothed minimum ridge h_star(tau)
2. fitted advection displacement a_hat * tau
3. optional apparent-advection vector from cross-correlation
4. optional ERA5/GEOS wind vector after unit conversion
```

If the model is right, the low-semivariance basin should be centered near
`a_hat * tau`.

## The Four Recommended Panels

### 1. Lag-Surface Small Multiples

Plot `gamma_hat(h_lat, h_lon, tau)` for several `tau` values.

Expected pattern under advection:

```text
tau = 1: minimum near a
tau = 2: minimum near 2a
tau = 3: minimum near 3a
```

Failure modes:

```text
minimum does not move linearly with tau -> time-varying advection or dynamics
minimum is broad and flat -> advection weakly identifiable
minimum sits at boundary -> lag grid is too small or slice is under-resolved
multiple minima -> mixed regimes, cloud/missingness effects, or nonstationarity
```

### 2. Ridge-Tube Plot

Extract

```text
h_star(tau) = argmin_h gamma_hat(h, tau)
```

after smoothing the lag surface. Plot `h_star(tau)` as a vector path in the
lag plane.

Compare it with the fitted line:

```text
h_model(tau) = a_hat * tau
```

This turns the diagnostic into a direct check of the fitted advection
parameter.

### 3. Metric-Collapse Plot

For each pair `(h_lat, h_lon, tau)`, compute the fitted model metric:

```text
d_fit(h, tau)^2 =
  ((h_lat - a_lat * tau) / range_lat)^2
+ ((h_lon - a_lon * tau) / range_lon)^2
+ (tau / range_time)^2
```

Then scatter or bin:

```text
x axis: d_fit
y axis: gamma_hat
color/facet: tau
```

If the metric is well specified, points from different directions and temporal
lags should approximately collapse onto one curve.

This is the strongest diagnostic for deciding whether the fitted distance is
actually explaining the data.

### 4. Odd/Asymmetry Surface

Compute:

```text
A(h, tau) = gamma_hat(h, tau) - gamma_hat(-h, tau)
```

This directly displays space-time asymmetry. It is useful because ordinary
spatial semivariograms are symmetric by construction, while advection creates
directional lead-lag asymmetry.

## Is This Really The Best?

For this specific question, yes, as the primary diagnostic.

Reason:

```text
The object we need to diagnose is not gamma(h_lat), gamma(h_lon), or gamma(t).
It is the geometry of gamma(h_lat, h_lon, tau) after transport by advection.
```

The advection-ridge plot is best because it checks the most interpretable
consequence of the model:

```text
Does the empirical low-variance region move by the fitted advection vector?
```

However, it is not sufficient alone. It should be paired with the
metric-collapse plot, because a ridge can look correct while the range/time
scaling is still wrong.

So the recommended diagnostic hierarchy is:

```text
1. Advection-ridge surface: checks direction and displacement.
2. Metric-collapse plot: checks full fitted distance geometry.
3. Odd/asymmetry surface: checks whether nonseparability/asymmetry is present.
4. Tile or latitude-band ridge maps: checks nonstationarity.
```

## When It Is Not The Best

This is not the best diagnostic if the goal is one of the following:

```text
1. Pure goodness-of-fit ranking:
   Use likelihood, prediction, simulation-based envelopes, or posterior
   predictive checks.

2. Nugget or microscale error diagnosis:
   Use near-zero spatial lag semivariograms and replicate/noise diagnostics.

3. Mean-function misspecification:
   Use residual maps, hourly mean fields, latitude-band residual means, and
   temporal residual autocorrelation.

4. Strongly local dynamics:
   Use tile-specific ridge vectors rather than one global ridge.
```

So the claim is narrower:

```text
For diagnosing whether the fitted advection-aware space-time distance is
geometrically plausible, the advection-ridge plus metric-collapse pair is the
best first diagnostic.
```

## What Would Be Even More Complete?

The full diagnostic would be local:

```text
gamma_hat(h_lat, h_lon, tau | tile, hour, day)
```

But this is too high-dimensional to view directly. A better compression is to
estimate local ridge vectors:

```text
v_ridge(tile, hour, tau) = h_star(tile, hour, tau) / tau
```

Then plot a vector field over latitude/longitude, with color showing:

```text
minimum gamma, ridge sharpness, or metric-collapse residual
```

This becomes a semivariogram-derived transport diagnostic:

```text
lat/lon/time -> local apparent advection vector
```

That is closer to the user's goal of using latitude, longitude, and time
together, without trying to visualize an impossible 5D object.

## Implementation Sketch

### Inputs

```text
daily_hourly_maps: dict
  key -> array with columns [lat, lon, value]

dlat_grid: array
dlon_grid: array
taus: list[int]
tolerance: float
```

### Pair Cache

For every `(dlat, dlon)` bin:

```text
lat_diff = coords[:, None, 0] - coords[None, :, 0]
lon_diff = coords[:, None, 1] - coords[None, :, 1]

mask =
  abs(lat_diff - dlat) <= tolerance_lat
  and
  abs(lon_diff - dlon) <= tolerance_lon
```

Store `(idx_row, idx_col)` for reuse.

### Surface Computation

For each day, start hour, and `tau`:

```text
cur = residual values at t
nxt = residual values at t + tau

diffs = cur[idx_col] - nxt[idx_row]
gamma[dlat, dlon, tau] = 0.5 * mean(diffs^2)
n_pairs[dlat, dlon, tau] = len(diffs)
```

Use `n_pairs` as a reliability mask. The edge bins can otherwise dominate
because they have few pairs.

### Ridge Extraction

Smooth the surface before argmin:

```text
gamma_smooth = gaussian_filter(gamma, sigma=1)
h_star = argmin(gamma_smooth)
```

Also compute ridge sharpness:

```text
sharpness = gamma_second_best - gamma_min
```

or a local Hessian/eigenvalue ratio around the minimum.

### Model Overlay

For a fitted daily parameter row:

```text
h_model_lat = advec_lat * tau
h_model_lon = advec_lon * tau
```

Plot this as a point or arrow on the heatmap.

### Metric Collapse

For all valid lag bins:

```text
d_fit = sqrt(
    ((dlat - advec_lat * tau) / range_lat)^2
  + ((dlon - advec_lon * tau) / range_lon)^2
  + (tau / range_time)^2
)
```

Then bin `gamma_hat` by `d_fit`.

## Practical Warnings

1. Use residuals, not raw TCO, unless the mean field is intentionally part of
   the diagnostic.
2. Missingness can create artificial ridges. Always plot `n_pairs`.
3. A one-hour ridge can be noisy. Check `tau = 1, 2, 3` jointly.
4. If the minimum is often at the lag-grid boundary, widen the lag grid.
5. If the ridge is broad, report uncertainty rather than a single vector.
6. If local ridges vary strongly by latitude, a stationary global advection
   parameter is probably too simple.

## Literature Anchor

This diagnostic is motivated by three existing lines of work:

```text
1. Empirical space-time variograms:
   use gamma(h, tau), not only gamma(h).

2. Nonseparable space-time covariance:
   space and time interact, so separable marginal plots can hide model failure.

3. Advection/diffusion or Lagrangian models:
   covariance is largest when spatial displacement follows transport.
```

The proposed contribution is the direct empirical visualization of the
advection ridge in the lag plane, then comparing that ridge with the fitted
advection parameter and the fitted space-time metric.

Useful starting points:

```text
Gneiting (2002), Nonseparable stationary covariance functions for space-time
data, JASA. DOI: 10.1198/016214502760047113
https://sites.stat.washington.edu/NRCSE/pdf/trs63_nonsep.pdf

Sigrist, Kuensch, and Stahel (2012), SPDE-based modelling of large space-time
data sets. This explicitly connects advection/diffusion to interpretable
space-time dependence.
https://arxiv.org/abs/1204.6118

Sigrist, Kuensch, and Stahel (2011), Dynamic nonstationary spatio-temporal
precipitation model. The model links advection to external wind vectors.
https://arxiv.org/abs/1102.4210

gstat variogramST documentation. This is the standard empirical
space-time variogram object gamma(h, tau).
https://r-spatial.github.io/gstat/reference/variogramST.html

Testing lack of symmetry in spatial-temporal processes. Useful background for
using asymmetry as evidence against fully symmetric space-time covariance.
https://pmc.ncbi.nlm.nih.gov/articles/PMC2662627/
```
