# Simulation space-time interaction diagnostic: first prototype

## Run

- Data: matched nugget-zero July 2024 simulation, 2024-07-13, 8 hourly fields.
- Truth: `sigma2=10`, `range_lat=0.2`, `range_lon=0.3`,
  `range_time=2`, `v_lat=0.08`, `v_lon=-0.2`, `nugget=0`.
- Fit: adapted Vecchia corridor, lag 4/3/2, 4x4 blocks, batch 64,
  float64, fixed statistical nugget zero, M3/Q3 initializer, one fit only.
- The alternative truth/no-advection/reversed-advection/separable curves were
  evaluated analytically. They were not refitted.

The one fit took 112.53 s after 1.32 s of graph precomputation. The fitted
parameters were `sigma2=9.7538`, `range_lat=0.1950`, `range_lon=0.2933`,
`range_time=1.9496`, `v_lat=0.08069`, and `v_lon=-0.20479`. The native mean
NLL was 0.8976529. All post-fit diagnostics and figures took about 31.13 s
when the saved fit was reused.

## Main findings

The empirical cross-variogram minima recover the displacement path:

| time lag | empirical minimum in cells | physical spatial lag | truth displacement |
|---:|---:|---:|---:|
| 1 | `(dr,dc)=(2,-3)` | `(0.088,-0.189)` | `(0.080,-0.200)` |
| 2 | `(dr,dc)=(4,-6)` | `(0.176,-0.378)` | `(0.160,-0.400)` |

Thus the minimum is useful for a constant-advection initializer. The full
oriented surface is more informative than the minimum: fitted cross-variogram
RMSE is 0.112, compared with 1.358 for no advection and 1.961 for reversed
advection. The h-versus-minus-h asymmetry RMSE is 0.092 for the fitted model,
1.968 for no advection, and 3.928 for reversed advection.

For the proposed seven-point stencil (center plus six directions), retain the
six contrasts as a matrix rather than summing them into one Laplacian:

- `D_lat`, `D_lon`, `D_time` are normalized central first differences.
- `Q_lat`, `Q_lon`, `Q_time` are normalized central second differences.
- Covariances of spatial `D` with temporal `D` form the signed odd-odd block.
- Covariances of spatial `Q` with temporal `Q` form an even-even block.

Across the tested scales, the fitted model has odd-odd RMS standardized
residual 0.27--1.03. Removing advection raises this to 5.56--21.77, and
reversing advection raises it to 11.64--43.40. This makes the odd-odd block the
cleanest directional/advection diagnostic in this experiment.

The even-even block separates a different failure. The fitted model has RMS
standardized residual 0.23--0.86, whereas the separable model with matched
spatial and temporal margins has 15.64--53.81. It therefore detects symmetric
joint structure that is not summarized by the direction of advection.

The mixed rectangle

`[Z(s+h,t+u) - Z(s+h,t) - Z(s,t+u) + Z(s,t)] / 2`

annihilates additive space-only plus time-only means and all joint affine
means. Its variance strongly rejects no advection (aggregate RMS z=18.83) and
the separable comparator (45.19), while fitting the data well (0.46). However,
its variance is exactly unchanged by `h -> -h` and cannot identify the sign of
constant advection: reversed advection also has RMS z=0.48. Pair it with the
odd-odd block or oriented cross-variogram asymmetry whenever direction matters.

## Recommended diagnostic panel

Use all three views because no single scalar represents space-time
interaction:

1. Oriented cross-variogram plus h-versus-minus-h asymmetry for displacement
   and directional lack of fit.
2. Odd-odd/even-even blocks of the six-direction contrast covariance matrix
   for directional versus symmetric interaction.
3. Mixed-rectangle variance and a separable-model reference for local joint
   roughness after removing additive mean contamination.

The cluster jackknife used here deletes 8x8 spatial tiles. It is adequate for
ranking sensitivities in this single-day prototype, but neighboring tiles are
still dependent. Formal p-values should be calibrated from independent
simulation replicates or a larger moving-block bootstrap.
