# Latitude/longitude band eigenanalysis

## Design

- Days: July 13, 19, and 25, 2024 (eight hourly fields per day).
- Latitude bands: [-3,-2), [-2,-1), [-1,0), [0,1), [1,2].
- Primary alternating longitude bands: [122,123), [124,125), [126,127), [128,129), [130,131].
- East-west 10-degree profiles are split into two 5-degree windows; both directions are resampled to 80 positions.
- Within each day and band, correlations are pooled by spatial lag to form an 80x80 stationary Toeplitz correlation matrix before eigendecomposition.
- Lag pooling avoids the finite-sample eigenvalue-spreading artifact caused by the unequal numbers of E-W and N-S profiles.
- Lower effective rank and fewer modes for 90% trace indicate stronger concentration in smooth leading modes.

## Simulation truth

- range_lat=0.2, range_lon=0.3; expected E-W smoothness is greater because range_lon/range_lat=1.500.
- This same-time analysis does not test advection sign.

## Daily direction averages

| dataset | day | eff.rank E-W | eff.rank N-S | E-W/N-S | lambda1 E-W | lambda1 N-S | modes90 E-W | modes90 N-S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simulation | 13 | 33.754 | 40.746 | 0.828 | 0.0939 | 0.0786 | 37.60 | 42.40 |
| simulation | 19 | 33.416 | 42.242 | 0.791 | 0.0934 | 0.0717 | 37.20 | 43.60 |
| simulation | 25 | 32.168 | 41.859 | 0.768 | 0.1078 | 0.0793 | 36.20 | 43.60 |
| real | 13 | 21.398 | 33.034 | 0.648 | 0.2506 | 0.1229 | 31.40 | 37.40 |
| real | 19 | 28.336 | 36.923 | 0.767 | 0.1866 | 0.1479 | 41.20 | 45.40 |
| real | 25 | 16.590 | 33.730 | 0.492 | 0.3363 | 0.1939 | 27.80 | 44.40 |

## Overall three-day summary

- simulation: mean E-W/N-S effective-rank ratio=0.796 (daily range 0.768–0.828); mean lambda1 ratio=1.285.
- real: mean E-W/N-S effective-rank ratio=0.636 (daily range 0.492–0.767); mean lambda1 ratio=1.678.

## Longitude-band offset robustness

- simulation: primary=41.616, alternate=41.401, relative difference=-0.52%.
- real: primary=34.562, alternate=38.127, relative difference=10.31%.

## Interpretation guardrails

- Raw eigenvalue magnitudes from the original 10-degree E-W and 5-degree N-S domains are not comparable.
- These spectra are descriptive because neighboring profiles and hours are correlated.
- Axis anisotropy is supported when the direction contrast is stable across days and bands and recovers the known simulation ordering.
- Signed transport asymmetry still requires positive-time-lag h versus -h diagnostics or odd-odd space-time contrasts.
