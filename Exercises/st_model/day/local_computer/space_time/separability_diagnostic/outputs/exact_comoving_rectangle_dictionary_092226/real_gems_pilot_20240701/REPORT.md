# Fixed canonical contrast on GEMS: one-day pilot

This is a descriptive application of the contrast fixed in the exact-comoving simulation. No contrast search, covariance refit, or coefficient re-optimization was performed.

## Inputs

- Data: `/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl`
- Existing fit: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/spectrum_diagnostics/outputs/real_july2024_st_heads_vecchia_lag432_one_day_gc_a075_b1_nugget0_061626/heads_vecchia_lag432_one_day_fit_summary.csv`; strategy `standard_432`, day `2024-07-01`
- Fixed coefficients: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226/global_two_rectangle_search/global_pair_ties.csv`; d1=`0.15662936719606299`, d2=`0.15402195946103639`
- Monthly centering mean reproduced from the selected July region: `257.97261042523161`
- Mean model removed: intercept, centered source latitude, and seven nominal time indicators using the saved GLS coefficients.

## Transfer from the simulation

The two mirror geometries use A endpoints (+/-2 fitted latitude ranges, +/-2 fitted longitude ranges) and B endpoints (+/-1 fitted latitude range, +/-2 fitted longitude ranges). Each endpoint follows the saved fitted advection over time. Continuous targets are mapped to nearest regular-grid cells; covariance expectations use the actual source coordinates of the retained observations.

## Descriptive result

- Complete translated contrast evaluations: `29576`
- Empirical mean L^2: `1.2952664`
- Fitted joint-GC mean Var(L): `1.3811817`; empirical/model ratio `0.937796`
- Matched-separable mean Var(L): `1.746683`; empirical/model ratio `0.741558`
- By absolute log variance ratio, the descriptive closer model is `joint`.
- Cross covariance Cov(QA,QB): empirical `0.19885039`, fitted joint GC `0.39086388`, matched separable `7.2253519`.

The individual QA and QB variances are comparatively similar under the two fitted constructions; most of their separation for this fixed diagnostic is in the cross covariance. The comparison is nevertheless about this one fixed second moment only. A ratio nearer one does not establish that the corresponding full covariance model is correct.

## Interpretation boundary

This is not a calibrated real-data hypothesis test. The translated contrasts overlap heavily, the covariance parameters and mean were fitted to the same day, nearest-cell mapping perturbs exact comoving geometry, and the real truth is unknown. The result is therefore a model-checking demonstration, not validation of the diagnostic or evidence that one model is true.

## Reproduction

```bash
python apply_fixed_canonical_contrast_real_gems.py
```
