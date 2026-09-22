# Nugget-zero advected-separable diagnostic: pilot result

## Design

- Independent design days: 2023-07-04, 2023-07-29, 2024-07-13
- Held-out response days: 2024-07-19, 2025-07-06
- Dense dimension per day: 800
- Spatial design: max-min distributed nearest-neighbor paired flow tubes
- Statistical nugget: exactly 0 in both models
- Direction selection: covariance-only on the design split; no held-out response used

## Truth and strongest fitted null

The true joint Matern parameters are `{'variance': 10.0, 'range_lat': 0.2, 'range_lon': 0.3, 'range_time': 2.0, 'advec_lat': 0.08, 'advec_lon': -0.2, 'nugget': 0.0}`.

The KL-projected advected-separable null is `{'variance': 9.870274421200065, 'range_lat': 0.22927005927233685, 'range_lon': 0.2929044049964727, 'range_time': 5.733264709379305, 'advec_lat': 0.16513308259772244, 'advec_lon': -0.20410190436793887, 'nugget': 0.0}`.  Parameters
at an optimization bound: **none**.  The best normalized expected-NLL
objective is `2.526329759`.

On the averaged design covariance, the generalized spectrum has total KL
`1.9713142` and KL per observation
`0.0024641427`.  The spectral identity
sum g(lambda)=KL differs by
`4.685e-14`.

## Held-out calibration

- Max-projection bootstrap p-value: `0.048039`
- Max-projection oracle power at alpha=0.05: `0.15922`
- Top-subspace LLR bootstrap p-value: `0.0300194`
- Top-subspace LLR oracle power at alpha=0.05: `0.29048`

## Moving-rectangle subspace

The 350 adjacent-time double differences retain `0.39540633` KL, or `20.058%`
of the full-design KL.

These are pilot, alternative-specific results for this fixed spatial design.
They do not establish final power for the full GEMS domain.  A
publication run should repeat the fixed pipeline over subset sizes and seeds,
then use at least 1,000 complete null/alternative simulations.  If null fitting
or direction selection is performed on the same responses being tested, the
entire fit-and-select operation must be repeated inside each bootstrap draw.
