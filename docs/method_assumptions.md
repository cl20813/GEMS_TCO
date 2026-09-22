# Numerical and scientific conventions

This document records assumptions that affect the statistical meaning of a
fit. They are part of the maintained implementation, not incidental software
details. Analyses that change one of these conventions should report the
change and rerun the relevant sensitivity checks.

## Coordinates, time, and Matérn ranges

All spatial calculations use the numeric latitude and longitude values supplied
by the caller. The package does not convert degrees to great-circle distance.
Consequently, fitted spatial ranges and advection coefficients use the same
coordinate units as the input data. Time is likewise used in the units stored
in column 3 of a model tensor.

Matérn models use one range convention throughout the maintained spatial and
spatio-temporal implementations. For dimensionless distance \(d\),

\[
R_\nu(d)=
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\sqrt{2\nu}\,d\right)^\nu
K_\nu\left(\sqrt{2\nu}\,d\right).
\]

Thus \(R_{1/2}(d)=\exp(-d)\) and
\(R_{3/2}(d)=(1+\sqrt{3}d)\exp(-\sqrt{3}d)\). Reported ranges are the
denominators used to form \(d\); they are not practical-correlation ranges.
The direct Bessel, spline, pure-spatial Vecchia, and spatio-temporal Vecchia
implementations are regression-tested against this convention.

The advected anisotropic distance used by the spatio-temporal covariance is

\[
d^2=
\left\{\frac{\Delta\mathrm{lat}-a_{\mathrm{lat}}\Delta t}{r_{\mathrm{lat}}}\right\}^2
+
\left\{\frac{\Delta\mathrm{lon}-a_{\mathrm{lon}}\Delta t}{r_{\mathrm{lon}}}\right\}^2
+
\left(\frac{\Delta t}{r_t}\right)^2.
\]

This is nonseparable in the fixed coordinate system because space and time are
coupled through advection. It can be interpreted in coordinates moving with a
constant velocity, but the remaining covariance is an isotropic Matérn in the
scaled space-time distance, not a product \(R_S R_T\).

All current Debiased Whittle covariance paths use this advected distance with
Matérn smoothness \(\nu=1/2\), hence exponential correlation \(\exp(-d)\).

## Nugget and numerical stabilization

The statistical nugget and numerical diagonal stabilization are distinct:

- full and Vecchia covariance factorizations use a documented default diagonal
  stabilization of \(10^{-6}\);
- the point self-covariance helper uses \(10^{-8}\);
- these values are not reported as part of the fitted nugget;
- direct full-likelihood fits expose `nugget_mode="free"`, `"fixed"`, and
  `"fixed0"`;
- Torch fits can hold any scalar parameter fixed by excluding it from the
  optimizer and setting `requires_grad=False`.

The maintained likelihoods assume there is at most one observation at an exact
latitude/longitude/time coordinate. Under that assumption, nugget variance is
added only to self-covariance. Duplicate observations require an explicit
observation identifier and are outside the current API.

## Debiased Whittle estimators

The scalar estimators require a complete rectangular spatial grid at every
modeled time. Time slices must be chronological, nonempty, and separated by one
unit because the expected periodogram forms temporal lags from slice indices.
Spatial increments default to 0.044 and 0.063 input-coordinate units and should
be supplied explicitly when another grid is used.

The five public filters are:

| Name | Operation |
| --- | --- |
| `identity` | no convolution; spatially demean each time slice |
| `latitude_difference` | first latitude difference |
| `longitude_difference` | first longitude difference |
| `cross_difference` | stencil `[[-1, 1], [1, -1]]`, equal to \(-D_{lat}D_{lon}\) under the package's forward-difference convention |
| `summed_first_differences` | \(D_{lat}+D_{lon}\) |

Each filter has an explicit retained-frequency mask. The mixed-frequency
objective partitions the *discrete Fourier index grid*; its cutoff is not a
physical wavenumber unless the analyst performs that conversion. The
scalar, mixed-frequency, and vector-gradient likelihoods all apply adaptive
diagonal loading to their spectral matrices before Cholesky factorization
(scale \(10^{-8}\), floor \(10^{-9}\)). That loading is a numerical regularizer
and should be included in sensitivity analysis when conclusions depend on
nearly singular frequencies.

The scalar parameter order is
`(log_phi1, log_phi2, log_phi3, log_phi4, advec_lat, advec_lon, log_nugget)`,
where `signal_variance = phi1 / phi2`,
`range_lon = 1 / phi2`,
`range_lat = 1 / (phi2 * sqrt(phi3))`, and
`range_time = 1 / (phi2 * sqrt(phi4))`.

## Grouped corridor Vecchia approximation

Model tensors use columns
`[latitude, longitude, centered_response, time, seven_hour_indicators]`.
The mean design is an intercept, centered latitude, and the seven hour
indicators. Structurally zero columns are removed before the exact GLS solve;
any remaining rank deficiency is an error rather than an implicit ridge fit.

The input mapping's insertion order is its temporal order. Each mapping value
must contain exactly one finite time value, and those values must be strictly
increasing in insertion order. `second_lag_stride` is the index offset used for
the second temporal conditioning layer, not a duration inferred from
timestamps. Every time slice must share the same regular-grid row order. When
`grid_coords` is provided, it defines fixed block geometry while covariance is
evaluated at the source coordinates in the model tensors.

The conditioning graph is selected once using a reference advection value and
then held fixed while covariance advection parameters are optimized. Corridor
selection uses Euclidean latitude/longitude degree geometry. The 4/3/2 and
6/4/3 labels are the numbers of conditioning blocks at the current, first-lag,
and second-lag layers. The directional 4/3/2 configuration uses a conservative
diagonal block span; the directional 6/4/3 configuration uses a projected
rectangular span. These are distinct approximations and should be named in
reported results.

Optimization never reruns max-min ordering, corridor searches, or
conditioning-block selection. Each objective evaluation reuses the same
precomputed batches and changes only covariance values within those fixed
conditionals. Within one fit, the L-BFGS wrapper retains the loss, validity
flag, and gradients for every evaluated covariance-parameter vector. A stored
evaluation is reused only when the raw parameter values have exactly the same
bit representation. This removes duplicate evaluations requested at the same
state; it does not freeze the covariance parameters or approximate the
likelihood.

Spline Matérn variants evaluate a finite lookup table and set correlation to
zero beyond `spline_r_max` (default 20). Analyses using those variants should
check sensitivity to both table resolution and maximum distance.

## Pure-spatial models

Pure-spatial block Vecchia treats time slots as independent replicates sharing
one spatial covariance. Public mean designs are:

- `lat`: intercept and centered latitude;
- `latlon`: intercept, centered latitude, and centered longitude;
- `base`: intercept, centered latitude, and seven hour indicators;
- `latlon_hour`: intercept, centered latitude, centered longitude, and seven
  hour indicators (the block-model default);
- `hour_spatial`: hour-specific intercept, latitude slope, and longitude slope.

The direct full Matérn likelihood supports `constant`, `lat`, and `latlon`.
The meaning of `latlon` is identical in the direct and block-Vecchia paths.

## Optimization and comparison

Closure-based L-BFGS routines re-evaluate the objective after each step, retain
the best finite valid state, and report convergence separately from the final
loss. An invalid covariance factorization is never evidence of convergence.
Flat numerical penalties are used only to steer an optimizer away from invalid
regions; a run with no finite valid state is a failed fit.

Likelihood values are comparable only when they use the same observations,
conditioning graph, mean design, nugget policy, and normalization. The
spatio-temporal grouped Vecchia objective omits the additive Gaussian constant
\(\tfrac12\log(2\pi)\); this does not change parameter estimates but must be
restored before comparison with a fully normalized Gaussian likelihood.

## Data alignment

Raw netCDF geolocation and data groups are aligned by their shared dimension
indices, not by row position. Processed hourly tensors and their aggregated
forms must have identical coordinate/time order. Missing responses may carry
missing source coordinates, but an observed response must have finite
coordinates. Requests that load no processed files fail explicitly rather than
returning an apparently valid zero-mean dataset.
