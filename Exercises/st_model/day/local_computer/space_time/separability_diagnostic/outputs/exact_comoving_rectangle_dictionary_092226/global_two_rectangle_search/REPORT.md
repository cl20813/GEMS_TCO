# Global two-rectangle exhaustive search

This is a covariance-only exact-comoving oracle calculation. It reads no responses and is not a calibrated test.

## Objective boundary

The exhaustive target is `mu(w) = w' (Sigma1-SigmaM) w / (w' Sigma0 w)`, where SigmaM is the matched-margin advected-separable covariance and Sigma0 is the fitted null.
It therefore isolates the intrinsic interaction discrepancy. It is **not** an exhaustive minimization of `Sigma1-Sigma0`, `rho0`, or `g(rho0)`.

## Result

- Exhaustively evaluated `35,275,800` unordered pairs of `8,400` rectangles.
- Global minimum: `mu = -0.190275209275`.
- Strict ties within the declared tolerance `1e-12`: `14`.
- Strict-tie D4-canonical relative-geometry classes: `1`.
- Largest equivalence class: `14` pairs.
- Temporal placements represented among the ties: `7`; mirror-handed variants: `2`.
- Greedy pair `650;1271` has `mu = -0.0993278821166` and is not globally minimizing.
- The next symmetry-near level is `3.76364e-09` above the minimum.
- At sensitivity tolerance `1e-08`, `28` pairs occupy `1` D4-canonical class(es).

## Interpretation

Every strict tie belongs to one relative-geometry orbit.

Under the stored `p<q, k<l` endpoint orientation, the two rectangle coefficients have the same relative sign. This sign label is convention-dependent; reversing either atom reverses its coefficient without changing the final filter.

> Under the canonical endpoint orientation, the intrinsic optimum is a same-signed, one-hour temporal difference of two co-centered, unequal-length, non-collinear spatial contrasts.

`Co-centered` is deliberate: the two spatial segments share a center but are neither parallel nor geometrically concentric.

## Representative filter

The representative strict tie is `s000_024_t00_01` with `s005_019_t00_01`.
Its standardized coefficient ratio is `0.982878304498` and its fitted-null rectangle correlation is `g_ij=0.220146129864`.
The spatial squared lengths are `32` and `20`, their center displacement is `(0,0,0)`, and their unsigned angle is `18.43494882` degrees. They share `2` time endpoints, `0` spatial anchors, and `0` observation-support points.
In raw rectangle units, define `S_t = (0.156629367196)(Y[0,t]-Y[24,t]) + (0.154021959461)(Y[5,t]-Y[19,t])`. The filter is exactly `L=S_0-S_1`.

The intrinsic quadratic-form terms are

- first diagonal: `+0.00127411230299`;
- cross term: `-0.195516396778`;
- second diagonal: `+0.00396707519969`;
- total: `-0.190275209275`.

The negative cross term is `37.304` times the two positive diagonal terms combined.

For this fixed filter, the fitted-null comparison is

- intrinsic `v1-vM = -0.190275209275`;
- compensation `vM-v0 = +0.0281110148332`;
- total `v1-v0 = -0.162164194442`;
- `rho0 = 0.837835805558` and `g(rho0) = 0.00738446967218`.

## Geometry-equivalence rule

The saved class key retains both segment vectors, their time widths, center displacement, and relative dot/cross geometry. It is canonical under pair exchange, space/time translation, endpoint reversal, and the eight rotations/reflections of the square grid.
The strict ties have zero center displacement. The wider sensitivity set contains the 90-degree rotated counterparts in the same geometric class; their small objective gap is reported rather than silently declaring them exact ties.
The fitted-null longitude/latitude range ratio is `1.50000010743` versus the exact target ratio `1.5`. After preserving the fitted range product but imposing the exact ratio, the 28 screened motif variants span only `1.943e-16` in objective value.
That last calculation is a local axis-symmetry sensitivity audit of the 28 screened pairs, not a refit or a second global search.

## Numerical audit

- Ordinary closed-form pairs: `35,275,800`.
- Nearly singular positive-definite pairs: `0`.
- Non-positive two-column null Gram matrices: `0`.
- Tiny negative discriminants clamped to zero in the primary block partition: `5,814`.
- Tiny negative discriminants clamped in the verification partition: `5,814`.
- Maximum analytic/SciPy discrepancy among final ties: `8.327e-17`.
- Maximum reduced generalized-eigen residual among final ties: `2.088e-16`.
- Maximum null-normalization error among final ties: `4.441e-16`.

The clamp count is an execution-order diagnostic and may vary with BLAS reduction order; the minimum, validated tie set, and equivalence classes are the scientific reproducibility targets.
The primary and verification block partitions are required to return the same strict candidate indices and minimum before any files are written.

`global_pair_ties.csv` retains rectangle IDs, sizes, center displacement, temporal arrangement, overlap diagnostics, `g_ij`, coefficients, and fitted-null variance metrics for every strict tie.
