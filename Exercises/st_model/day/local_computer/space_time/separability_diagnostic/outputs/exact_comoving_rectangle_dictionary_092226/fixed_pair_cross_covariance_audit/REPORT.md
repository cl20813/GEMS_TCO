# Fixed-pair cross-covariance audit

This audit freezes the representative pair and time endpoints `(0,1)` already selected by the earlier exact-comoving global search. It performs no contrast search, coefficient optimization, parameter fit, simulation, bootstrap, or power calculation.

## Fixed inputs and normalization

- Oracle directory: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226`
- Parameter source: `experiment_manifest.json`; coordinate source: `exact_comoving_points.csv`; atom source: `rectangle_dictionary_metadata.csv`; coefficient source: `global_two_rectangle_search/global_pair_ties.csv`.
- Rectangles: `s000_024_t00_01` and `s005_019_t00_01`.
- `Sigma_1`: joint Matern-half with `(variance, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget)=(10, 0.20000000000000001, 0.29999999999999999, 2, 0.080000000000000002, -0.20000000000000001, 0)`.
- `Sigma_M`: advected-separable with the saved matched-margin tuple `(10, 0.20000000000000001, 0.29999999999999999, 2, 0.080000000000000002, -0.20000000000000001, 0)`.
- `Sigma_0`: advected-separable fitted null with tuple `(8.9624969604739437, 0.13229505989419363, 0.19844260405396283, 1.5561520090880183, 0.080000000000000002, -0.20000000000000001, 0)`.
- Saved numerical jitter ratio `1e-10` is retained solely to reproduce the original matrices; it is distinct from the zero statistical nugget.
- Every reported displacement `h=(h_lat,h_lon)` is in the standardized moving-coordinate grid.
- Raw atoms use `+p,k -q,k -p,l +q,l`, so they are exactly `Q_A=A_0-A_1` and `Q_B=B_0-B_1`.
- Full-precision raw coefficients from `global_pair_ties.csv`: `d1=0.15662936719606299`, `d2=0.15402195946103639`.
- Reconstructed unit-Sigma0 coefficients: `0.645658192065651`, `0.63460342910300083`.
- Each dictionary atom was divided by its fitted-null standard deviation before the 2-by-2 eigenproblem. The saved raw coefficient is therefore the standardized eigenvector coefficient divided by that standard deviation.
- Maximum saved-versus-reconstructed normalization error: `2.220e-16`.

## Direct full-matrix calculation

- `H_1 = [[15.7357255206, 1.63166376046], [1.63166376046, 15.7262178373]]`
- `H_M = [[15.6837903787, 5.6839242185], [5.6839242185, 15.5589913202]]`
- `H_0 = [[16.9925651909, 3.73904160564], [3.73904160564, 16.976163161]]`
- After the original unit-Sigma0 atom normalization, `D = [[0.00305634501509, -0.238587732132], [-0.238587732132, 0.00985066622771]]` and `G = [[1, 0.220146129864], [0.220146129864, 1]]`; these reproduce the saved reduced matrices within `2.220e-16`.
- Maximum error against the independent K_delta decomposition: `7.105e-15`.
- Maximum stationarity spread over all available placements: `3.553e-15`.

## A. Structural claim

Confirmed. With `I(s)=Y(s,0)-Y(s,1)`, the two atoms are `Q_A=I((-2,-2))-I((2,2))` and `Q_B=I((-1,-2))-I((1,2))`. Spatial stationarity and covariance symmetry, without an isotropy assumption, give

`Var(Q_A)=2[K_delta(0,0)-K_delta(4,4)]`,

`Var(Q_B)=2[K_delta(0,0)-K_delta(2,4)]`,

`Cov(Q_A,Q_B)=2[K_delta(1,0)-K_delta(3,4)]`.

Thus the short vector lag `(1,0)` is absent from both individual variance formulas and is present in the cross-covariance formula. The implementation and the direct 200-dimensional calculation agree to the error reported above.

## B. Numerical claim

The matched-margin check was not forced: `deltaK(0,0)=-2.6645352591003757e-15`, numerically zero up to roundoff.

For the raw `DeltaH_AB=H_1,AB-H_M,AB` discrepancy:

- short contribution `2 deltaK(1,0) = -4.151669408133122`;
- far contribution `-2 deltaK(3,4) = +0.09940895009819245`;
- sum `-4.0522604580349295`, versus direct `DeltaH_AB=-4.0522604580349322`.

The far term has the opposite sign and partially offsets the short-lag term; it does not reinforce it.

For the fixed final `L=d1 Q_A+d2 Q_B` numerator:

| contribution | raw value | divided by V0 |
|---|---:|---:|
| diagonal A | +0.00127411230299 | +0.00127411230299 |
| diagonal B | +0.00396707519969 | +0.00396707519969 |
| cross short | -0.200312751783 | -0.200312751783 |
| cross far | +0.0047963550053 | +0.0047963550053 |
| total N | -0.190275209275 | -0.190275209275 |

`V0=1.0000000000000002` and `N/V0=-0.19027520927526689`; the saved global-search objective is `-0.19027520927526689`.

## Answers to the four audit questions

1. **Yes.** The formulas, atom signs, indices, and direct implementation agree: `(1,0)` enters only the cross-covariance, not either individual variance.
2. The unweighted cross-covariance contributions are short `-4.15166940813` and far `+0.0994089500982`. At the final-L numerator scale they are respectively `-0.200312751783` and `+0.0047963550053`.
3. The dominant discrepancy is the negative short-lag cross contribution. The far cross term and both diagonal terms are positive offsets.
4. The proposed interpretation is supported for this fixed selected exact-comoving pair, but should be stated more precisely: **the standardized moving-coordinate lag `(1,0)` supplies the dominant negative truth-minus-matched contribution; the `(3,4)` cross lag and both diagonal terms partially offset it.** This is a result-specific decomposition, not a universal causal claim about all contrasts.

## Reproduction

From the diagnostic directory:

```bash
python audit_fixed_pair_cross_covariance.py
```

Inputs are read from the saved experiment manifest, exact-comoving point table, rectangle metadata, and global strict-tie table. Outputs are written only inside this audit directory.
