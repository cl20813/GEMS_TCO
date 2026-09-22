# Nugget-zero advected-separable diagnostic: pilot result

## Design

- Independent design days: 2023-07-04, 2023-07-29, 2024-07-13
- Held-out response days: 2024-07-19, 2025-07-06
- Dense dimension per day: 800
- Spatial design: max-min distributed local paired flow tubes for moving rectangles
- Statistical nugget: exactly 0 in both models
- Direction selection: covariance-only on the design split; no held-out response used

## Truth and strongest fitted null

The true joint Matern parameters are `{'variance': 10.0, 'range_lat': 0.2, 'range_lon': 0.3, 'range_time': 2.0, 'advec_lat': 0.08, 'advec_lon': -0.2, 'nugget': 0.0}`.

The KL-projected advected-separable null is `{'variance': 9.837345522604702, 'range_lat': 0.18723572770083327, 'range_lon': 0.28456555695530794, 'range_time': 7.084228305500491, 'advec_lat': 0.12504290639770438, 'advec_lon': -0.2817559706219737, 'nugget': 0.0}`.  Parameters
at an optimization bound: **none**.  The best normalized expected-NLL
objective is `2.809016734`.

On the averaged design covariance, the generalized spectrum has total KL
`2.0800523` and KL per observation
`0.0026000653`.  The spectral identity
sum g(lambda)=KL differs by
`2.576e-14`.

## Held-out calibration

- Max-projection bootstrap p-value: `0.0282794`
- Max-projection oracle power at alpha=0.05: `0.17612`
- Top-subspace LLR bootstrap p-value: `0.0971981`
- Top-subspace LLR oracle power at alpha=0.05: `0.319`

## Moving-rectangle subspace

The 350 adjacent-time double differences retain `0.49165355` KL, or `23.637%`
of the full-design KL.

These are pilot, alternative-specific results for this fixed spatial design.
They do not establish final power for the full GEMS domain.  A
publication run should repeat the fixed pipeline over subset sizes and seeds,
then use at least 1,000 complete null/alternative simulations.  If null fitting
or direction selection is performed on the same responses being tested, the
entire fit-and-select operation must be repeated inside each bootstrap draw.
