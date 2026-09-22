# Nugget-zero advected-separable diagnostic: pilot result

## Design

- Independent design days: 2023-07-04, 2023-07-29, 2024-07-13
- Held-out response days: 2024-07-19, 2025-07-06
- Dense dimension per day: 800
- Statistical nugget: exactly 0 in both models
- Direction selection: covariance-only on the design split; no held-out response used

## Truth and strongest fitted null

The true joint Matern parameters are `{'variance': 10.0, 'range_lat': 0.2, 'range_lon': 0.3, 'range_time': 2.0, 'advec_lat': 0.08, 'advec_lon': -0.2, 'nugget': 0.0}`.

The KL-projected advected-separable null is `{'variance': 9.89882600458311, 'range_lat': 0.19228793934195929, 'range_lon': 0.26979558141675436, 'range_time': 3.931090484290789, 'advec_lat': 0.040909360846262684, 'advec_lon': -0.16102839686799336, 'nugget': 0.0}`.  Parameters
at an optimization bound: **none**.  The best normalized expected-NLL
objective is `2.91314105`.

On the averaged design covariance, the generalized spectrum has total KL
`1.5585045` and KL per observation
`0.0019481306`.  The spectral identity
sum g(lambda)=KL differs by
`4.241e-14`.

## Held-out calibration

- Max-projection bootstrap p-value: `0.956861`
- Max-projection oracle power at alpha=0.05: `0.09672`
- Top-subspace LLR bootstrap p-value: `0.318894`
- Top-subspace LLR oracle power at alpha=0.05: `0.1985`

These are pilot, alternative-specific results for the selected 100-location
flow tube.  They do not establish final power for the full GEMS domain.  A
publication run should repeat the fixed pipeline over subset sizes and seeds,
then use at least 1,000 complete null/alternative simulations.  If null fitting
or direction selection is performed on the same responses being tested, the
entire fit-and-select operation must be repeated inside each bootstrap draw.
