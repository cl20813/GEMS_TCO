# Exact-comoving K-delta displacement map

This is a population-covariance view of the saved exact-comoving experiment. It performs no simulation, contrast search, coefficient optimization, covariance refit, bootstrap, or power analysis.

## Inputs and definition

- Oracle directory: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226`
- Coordinates: saved `exact_comoving_points.csv`, in standardized moving-coordinate `(latitude, longitude)` units.
- Parameters: saved `experiment_manifest.json` entries `truth` and `matched_margin`.
- `Sigma_1`: joint Matern-half with variance/ranges/advection/nugget `(10, 0.20000000000000001, 0.29999999999999999, 2, 0.080000000000000002, -0.20000000000000001, 0)`.
- `Sigma_M`: advected-separable with matched parameters `(10, 0.20000000000000001, 0.29999999999999999, 2, 0.080000000000000002, -0.20000000000000001, 0)`.
- Numerical jitter ratio `1e-10` is retained to reproduce the saved matrices; statistical nugget is zero.

For each available signed displacement `h`, every valid base anchor was evaluated independently using the four original covariance entries

`K_delta,m(h) = C_m((s,0),(s+h,0)) - C_m((s,0),(s+h,1)) - C_m((s,1),(s+h,0)) + C_m((s,1),(s+h,1))`.

No contrast covariance was inverted to obtain these values.

## Numerical checks

- Signed displacements evaluated: `81`.
- Maximum spread across valid base anchors at a fixed displacement: `3.553e-15`.
- Maximum `h` versus `-h` covariance-symmetry error: `0.000e+00`.
- `delta K_delta(0,0)=-2.6645352591003757e-15`; this was computed, not set to zero.

## Result

The largest nonzero-lag absolute discrepancy is `2.07583470407` at radius `1`. All four shortest signed lags `(-1,0), (0,-1), (0,1), (1,0)` have the same value `delta K_delta=-2.07583470407`.

Thus `(1,0)` is not directionally unique in the intrinsic truth-minus-matched numerator. It is one of the four shortest nonzero lags. The discrepancy is radially symmetric on the standardized moving grid to numerical precision and its magnitude decreases with radius over the available grid.

In particular, the current map does not support an advection-parallel versus advection-perpendicular distinction after transforming to the exact moving coordinates. It supports the more general short-range increment-covariance interpretation.

For these saved parameters, let `r=sqrt(h_lat^2+h_lon^2)` and `tau=1/range_time`. Apart from the common diagonal jitter at the origin, the two maps reduce to

- `K_delta,1(r) = 20 [exp(-r) - exp(-sqrt(r^2 + tau^2))]`,
- `K_delta,M(r) = 20 [exp(-r) - exp(-(r + tau))]`,
- with `tau=0.5` for the fixed one-step time lag.

This directly explains the observed radial symmetry. It also shows why the interpretation is specific to the available grid: the discrepancy is zero at `r=0`, largest at the shortest available nonzero radius here, and then decreases across the radii represented by this grid.

This map examines only the numerator discrepancy `Sigma_1 - Sigma_M`. The original generalized quotient also used the fitted-null `Sigma_0` in its denominator, so a strict orientation preference in that quotient must not be attributed to directional structure in this map.

| radius squared | radius | signed lag count | K_delta,1 | K_delta,M | delta K_delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 7.86938680775 | 7.86938680775 | -2.6645352591e-15 |
| 1 | 1 | 4 | 0.819150916394 | 2.89498562046 | -2.07583470407 |
| 2 | 1.41421 | 4 | 0.399731485716 | 1.91317962221 | -1.5134481365 |
| 4 | 2 | 4 | 0.16158143836 | 1.06500569225 | -0.903424253894 |
| 5 | 2.23607 | 8 | 0.114835912445 | 0.841063738017 | -0.726227825573 |
| 8 | 2.82843 | 4 | 0.0507204226363 | 0.465125982139 | -0.414405559502 |
| 9 | 3 | 4 | 0.0403641185701 | 0.391793698911 | -0.351429580341 |
| 10 | 3.16228 | 8 | 0.0326128267643 | 0.3331050024 | -0.300492175636 |
| 13 | 3.60555 | 8 | 0.0184311918503 | 0.213830607428 | -0.195399415578 |
| 16 | 4 | 4 | 0.0112272519104 | 0.14413284701 | -0.132905595099 |
| 17 | 4.12311 | 8 | 0.00963702274985 | 0.127437977766 | -0.117800955016 |
| 18 | 4.24264 | 4 | 0.00831549378489 | 0.113079909878 | -0.104764416093 |
| 20 | 4.47214 | 8 | 0.00627788910449 | 0.0898911476675 | -0.083613258563 |
| 25 | 5 | 8 | 0.00331903616333 | 0.0530235112124 | -0.0497044750491 |
| 32 | 5.65685 | 4 | 0.00152404746231 | 0.0274916184197 | -0.0259675709573 |

Maximum within-radius directional spread of `delta K_delta`: `6.661e-16`.

## Reproduction

From the diagnostic directory:

```bash
python analyze_k_delta_displacement_map.py
```
