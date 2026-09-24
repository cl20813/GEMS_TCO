# Frozen diagnostic on three randomly selected held-out days

Selection seed: `20260923`. Eligible dates were July 2-22 and July 31; July 1 was the discovery day and July 23-30 were excluded before selection. Selected dates: 2024-07-03, 2024-07-22, 2024-07-31.

The July-1 A/B geometry, d1/d2 coefficients, lag one, range-unit transfer, and nearest-grid rule were frozen. Each selected day was independently fitted with the joint generalized-Cauchy 4/3/2 corridor-Vecchia model and GLS mean. This is a descriptive screen, not a calibrated test.

| date | n | empirical Hab | joint Hab | separable Hab | cross closer | empirical Var(L) | joint Var(L) | separable Var(L) | Var(L) closer |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 2024-07-03 | 44166 | 0.605098 | 0.168267 | 5.76598 | `joint` | 1.69716 | 1.7053 | 2.47269 | `joint` |
| 2024-07-22 | 49792 | 5.79003 | 0.263811 | 3.85204 | `separable` | 1.78935 | 1.34189 | 1.77336 | `separable` |
| 2024-07-31 | 51512 | 0.279501 | 0.0422882 | 1.04974 | `joint` | 1.05804 | 1.05967 | 1.29229 | `joint` |

Joint GC was closer for the cross covariance on `2/3` days and for Var(L) on `2/3` days.

No held-out result was used to alter the contrast. Overlapping translated contrasts mean these rows are descriptive model checks; bootstrap calibration remains separate.
