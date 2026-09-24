# Exact-comoving rectangle-dictionary oracle

This is a covariance-only oracle experiment.  It does not use response values and is not a calibrated real-data test.

## Geometry decision

- Across the `3` design days, the existing flow tube has `0` comoving groups out of `300` within tolerance `1.0e-12`.
- Its maximum unstandardized moving-coordinate drift is `0.0922768` degrees.
- Therefore the existing modes are audited only against the algebraic double-centering span; they are not claimed to be physical rectangle combinations.
- The main experiment uses a separate exact Cartesian product of moving anchors and hours.

## Matched-margin check

- Maximum pure-spatial Sigma1/SigmaM discrepancy: `0.000e+00`.
- Maximum pure-temporal Sigma1/SigmaM discrepancy: `5.329e-15`.
- All three models use the same known advection and zero statistical nugget in this experiment.
- A `1.0e-10` diagonal jitter is used only for numerical linear algebra; it is not a fitted or statistical nugget.

## Strong fitted null

- All `8` starts used the pilot optimizer tolerances; `8` converged.
- Known advection was fixed at `(0.08, -0.2)`.
- The KL-optimal fitted parameters are variance `8.962497`, latitude range `0.13229506`, longitude range `0.1984426`, and time range `1.556152`.
- The largest absolute optimizer-reported Jacobian component across starts is `1.332e-07`; attempt-level objective deltas and convergence messages are retained in the fit CSV.

## Dictionary and rank

- Actual rectangles: `8400`.
- Theoretical full rectangle-span rank `(m-1)(T-1)`: `168`.
- Numerical ranks over the requested tolerances: `168; 168; 168`.
- Individual rectangles were scaled to unit null variance. All fitted-null cross-correlations were accounted for exactly, without forming a rectangle-by-rectangle Gram matrix.

## Intrinsic objective comparison

The reported `mu` is a Sigma0-scaled intrinsic variance difference, not a variance ratio and not KL.

| branch | single | greedy-2 | greedy-4 | greedy-8 | dense dictionary | unrestricted full space |
|---|---:|---:|---:|---:|---:|---:|
| positive | 0.313382 | 0.458672 | 0.548518 | 0.66307 | 0.849854 | 1.05107 |
| negative | 0.00133828 | -0.0993279 | -0.225204 | -0.40858 | -0.665644 | -1.11146 |

The greedy rows are nested forward selections with all coefficients reoptimized after every addition.  They are not globally optimal k-rectangle subsets.

- Eight rectangles retain `78.0%` of the positive dense-oracle magnitude and `61.4%` of the negative magnitude.
- The best single rectangle on the minimizing branch is still positive (`mu=0.00133828`); the first negative interaction appears only after two rectangles are combined (`mu=-0.0993279`).  Within this finite oracle dictionary and geometry, off-diagonal cross-rectangle covariance terms are therefore necessary for the signed negative contrast.
- After fitted-null compensation, the dense positive filter has `rho0=1.4755`, `g=0.04325` and the dense negative filter has `rho0=0.4955`, `g=0.09884`.  These population-covariance contrasts remain nonzero after oracle KL refitting; this is not a significance or power result.

## Intrinsic, compensation, and total effects

| filter | intrinsic | compensation | total | rho0 | marginal g(rho0) |
|---|---:|---:|---:|---:|---:|
| `dense_dictionary_positive` | 0.849854 | -0.374339 | 0.475514 | 1.47551 | 0.0432539 |
| `dense_dictionary_negative` | -0.665644 | 0.161149 | -0.504495 | 0.495505 | 0.0988414 |
| `greedy_08_positive` | 0.66307 | -0.351075 | 0.311995 | 1.31199 | 0.0202231 |
| `greedy_08_negative` | -0.40858 | 0.133353 | -0.275227 | 0.724773 | 0.0233348 |

The two signed filters in each family are also evaluated jointly; joint projected KL retains their covariance and is not the sum of independent rectangle scores.

| filter pair | joint projected KL | sum of marginal g values |
|---|---:|---:|
| `dense_dictionary_positive;dense_dictionary_negative` | 0.142095 | 0.142095 |
| `greedy_08_positive;greedy_08_negative` | 0.043559 | 0.0435579 |

## Eight-rectangle geometry

- Positive branch: standardized spatial distances `[1.0]` and temporal lags `[1, 2]` hours.
- Negative branch: standardized spatial distances `[2.828427, 4.472136, 5.656854]` and temporal lags `[5, 7]` hours.
- More than one candidate lay within the predeclared score tolerance `1.0e-12` at `6` positive and `1` negative greedy steps.
- The saved path uses the smallest rectangle index within each tolerance tie. Later greedy selections and the reported k-rectangle objective are conditional on that deterministic branch; tied current-step scores do not imply identical future paths.
- Endpoint diagrams and the coefficient table show actual rectangles.  Pairwise lag attribution from the earlier atlas was not reinterpreted as a rectangle coefficient.

## Fixed-filter simulation check

- Under Sigma0, the maximum absolute relative error in the simulated squared-projection mean was `1.81%` and in its variance was `2.76%`.
- Across Sigma0, SigmaM, and Sigma1, the maximum absolute relative mean error was `1.81%`.
- Combining many rectangles still produces one Gaussian projection per filter; the simulation does not treat its atoms as independent replications.
- This Monte Carlo checks fixed-filter Gaussian moments only; it does not calibrate a test, select filters anew, or estimate power.

## Existing atlas span audit

| mode | fitted-null eigenvalue | algebraic retention | max anchor temporal sum | max time spatial sum |
|---:|---:|---:|---:|---:|
| 13 | 1.22624 | 0.3810 | 0.1211 | 0.2987 |
| 14 | 1.22425 | 0.3978 | 0.167 | 0.03967 |
| 39 | 0.834357 | 0.7490 | 0.13 | 0.3229 |
| 40 | 0.834768 | 0.8008 | 0.1039 | 0.5227 |

Basis-invariant two-dimensional cluster retention:

- `modes_13_14`: mean `0.3894`, minimum `0.3746`, maximum principal angle `52.26` degrees.
- `modes_39_40`: mean `0.7749`, minimum `0.5922`, maximum principal angle `39.69` degrees.

These retentions answer only whether the numerical weight arrays satisfy the double-centering constraints.  Because the source-coordinate anchors drift, they do not establish a combination of fixed physical rectangles.
The original 100-by-8 atlas and the exact 25-by-8 oracle use different observation geometries, so their objective values and mode numbers are not an apples-to-apples performance comparison.
The legacy audit reproduces its original fitted-advection covariance `(0.0409094, -0.161028)`, rather than the fixed truth advection `(0.08, -0.2)`; it is not part of the fixed-v0 exact-grid comparison.
For near-degenerate legacy mode pairs, the two-dimensional canonical-retention summaries are primary; individual mode retentions are basis-dependent descriptions.

## Interpretation boundary

- Intrinsic optimization uses `Sigma1-SigmaM`; fitted-null discrimination is evaluated afterward from `v1/v0` and `g(v1/v0)`.
- Intrinsic `mu` must not be identified with the existing total generalized eigenvalue `lambda-1`.
- Minimum-norm dense dictionary coefficients are not unique physical contributions when the dictionary is redundant.
- The single, greedy, dense, and full inequalities are checked only for the same intrinsic numerator and Sigma0 normalization.
- No nugget estimation, advection estimation, sparsity penalty, response-based selection, or p-value calibration was added.
- Every selected filter and percentage is specific to this regular 5-by-5-by-8 synthetic geometry; transfer to the warped 100-by-8 GEMS design requires a separate geometry-aware study.
