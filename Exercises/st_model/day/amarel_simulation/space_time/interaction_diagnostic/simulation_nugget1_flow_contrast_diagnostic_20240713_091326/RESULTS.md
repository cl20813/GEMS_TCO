# Nugget-1 flow-coordinate contrast diagnostic

## Data and fit

- Local asset: smooth=0.5 July 2024 simulation, three independent eight-hour days (July 13, 19, 25).
- Confirmed truth: sigma2=10.0, ranges=(0.2,
  0.3, 2.0), advection=(0.08,
  -0.2), **nugget=1.0**.
- One fit only: 2024-07-13, adapted lag 4/3/2, batch size 64, nugget estimated.
- Fitted advection=(0.081189,
  -0.203804), fitted nugget=0.987956,
  mean NLL=1.124294811, optimization time=150.44 s.
- Empirical contrast moments use the known simulation mean and independent-day
  standard errors across three days.  This isolates covariance/advection
  sensitivity from fitted-mean contamination.

## Fitted-flow grid axes

- Parallel integer offset: (dr,dc)=(2,-3),
  physical=(0.088,-0.189),
  direction error=3.25 degrees.
- Perpendicular integer offset: (dr,dc)=(4,1),
  physical=(0.176,0.063),
  direction error=2.03 degrees.
- Deviation between the two implemented grid axes and 90 degrees:
  5.27 degrees.
- Covariance-metric perpendicular offset: (dr,dc)=(3,2),
  physical=(0.132,0.126),
  direction error=1.79 degrees relative to the covariance-metric perpendicular target.

The metric-perpendicular direction is defined by
`h_lat*v_lat/range_lat^2 + h_lon*v_lon/range_lon^2 = 0`.  This is the natural
orthogonality condition after scaling space by the fitted anisotropic ranges;
ordinary Euclidean perpendicularity does not have this property when the two
ranges differ.

## Contrasts being tested

At a center `(s,t)`, the seven-point stencil contains the center, `+/-` two
spatial directions, and `+/-` time.  It is projected onto six zero-sum,
unit-norm contrasts:

- odd first differences: `D_a=[Z(s+h_a,t)-Z(s-h_a,t)]/sqrt(2)` and
  `D_t=[Z(s,t+tau)-Z(s,t-tau)]/sqrt(2)`;
- even second differences: `Q_a=[Z(s+h_a,t)+Z(s-h_a,t)-2Z(s,t)]/sqrt(6)`
  and the analogous `Q_t`.

The diagnostic compares the empirical and model-implied entries of the local
contrast covariance matrix.  In particular, `E[D_a D_t]` is the odd-odd
space-time component: it changes under flow reversal and is the most direct
local advection diagnostic.  `E[Q_a Q_t]` is even-even and measures symmetric
space-time curvature.  The four-point rectangle
`[Z(s,t)-Z(s+h,t)-Z(s,t+tau)+Z(s+h,t+tau)]/2` measures mixed roughness.

The nugget is not being ignored.  It raises the cross-variogram surface and
the rectangle variance.  It cancels from the odd-odd cross-moment because its
spatial and temporal contrasts use disjoint observations, while the shared
center makes it enter the even-even cross-moment.

## Main comparison

| model | metric-flow parallel-time RMS z | metric-flow perpendicular-time RMS z | oriented asymmetry RMSE | rectangle RMS z |
|---|---:|---:|---:|---:|
| truth | 1.344 | 0.652 | 0.1195 | 2.942 |
| fitted_432 | 0.432 | 0.627 | 0.0885 | 3.110 |
| zero | 38.195 | 1.358 | 1.9644 | 82.120 |
| half_speed | 12.695 | 0.837 | 0.7087 | 41.817 |
| double_speed | 20.817 | 0.782 | 0.8874 | 28.785 |
| reversed | 77.620 | 2.164 | 3.9971 | 2.942 |
| rotated_45 | 14.069 | 24.687 | 1.5877 | 39.150 |
| rotated_90 | 28.951 | 42.518 | 2.5243 | 52.865 |

The flow-parallel odd-odd component targets speed/sign mismatch along the
transport path.  The flow-perpendicular odd-odd component localizes angular
misalignment.  Their separation is the main gain over reporting latitude and
longitude components only.

For the metric-flow frame, half/double-speed errors give parallel RMS z values
of 12.695 and
20.817, while their
perpendicular values remain 0.837
and 0.782.  In
contrast, 45/90-degree rotations increase the perpendicular values to
24.687 and
42.518.  Thus this
coordinate change provides attribution, not merely a larger omnibus score.

## Cross-variogram minima

- tau=1: empirical minimum (dr,dc)=(2,-3), physical lag=(0.088,-0.189); truth displacement=(0.080,-0.200)
- tau=2: empirical minimum (dr,dc)=(4,-6), physical lag=(0.176,-0.378); truth displacement=(0.160,-0.400)

## Interpretation guardrails

- Oriented cross-variogram asymmetry and odd-odd contrasts retain advection
  sign.  Reversing the velocity should be strongly visible.
- Mixed-rectangle variance measures local joint space-time roughness, but its
  theoretical value is invariant under v -> -v.  It can diagnose missing or
  wrong-speed interaction but cannot identify advection sign by itself.
- The independent-day z scores are sensitivity measures under this known-mean
  simulation, not formal p-values: with only three independent days, their
  standard errors have two degrees of freedom.  A real-data goodness-of-fit test still needs a parametric
  bootstrap that refits the mean and covariance parameters.
