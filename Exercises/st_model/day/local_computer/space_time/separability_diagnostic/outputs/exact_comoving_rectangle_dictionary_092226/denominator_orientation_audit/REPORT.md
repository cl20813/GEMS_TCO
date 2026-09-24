# Denominator orientation audit

This audit uses the saved exact-comoving population covariances. It performs no simulation, covariance refit, coefficient optimization, or unrestricted pair search.
Input oracle: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226`.

## Fixed candidate class

Only pairs with signature `(outer norm squared, inner norm squared, short cross distance, long cross distance) = (32, 20, 1, 5)` were retained. Both rectangles also use the same one-step time interval and have the same spatial center.

- Length-filtered pairs screened: `168` (not 35,275,800).
- Exact signature matches: `28` = `4` spatial orientations x `7` time translations.
- All centers are at grid location `(2, 2)`.
- Actual strict global ties among these pairs: `14`.

## Numerator

The raw-atom numerator matrix is, using outer contrast first and inner contrast second,

`Delta H = [[0.0519351419146849, -4.05226045803493], [-4.05226045803493, 0.167226517125969]]`.

The maximum entrywise deviation over all 28 rotation/reflection/time variants is `3.664e-15`. Thus the intrinsic numerator is the same to floating-point precision.

## Fitted-null denominator

The selected reference has `H0 = [[16.9925651908517, 3.73904160564139], [3.73904160564139, 16.9761631609835]]`.
The fitted longitude/latitude range ratio is `1.5000001074316183`, whereas the exact standardized-grid target is `1.5`; the relative ratio error is `+7.162e-08`.

| spatial orientation | axis assignment | time copies | strict ties | H0_AB | H0_BB | lambda_min |
|---|---|---:|---:|---:|---:|---:|
| `outer=(4,-4);inner=(2,-4)` | `inner_long_longitude` | 7 | 7 | 3.73904160564139 | 16.9761631609835 | -0.190275209275267 |
| `outer=(4,4);inner=(2,4)` | `inner_long_longitude` | 7 | 7 | 3.73904160564139 | 16.9761631609835 | -0.190275209275267 |
| `outer=(4,-4);inner=(4,-2)` | `inner_long_latitude` | 7 | 0 | 3.7390420127883 | 16.9761631667027 | -0.190275205511624 |
| `outer=(4,4);inner=(4,2)` | `inner_long_latitude` | 7 | 0 | 3.7390420127883 | 16.9761631667027 | -0.190275205511624 |

Between the two axis assignments, `(H0_AA, H0_AB, H0_BB)` changes by `(+0.000e+00, -4.071e-07, -5.719e-09)`, while `Delta H` does not change materially.
The resulting objective gap is `3.76364273080299e-09`. The more negative class is `inner_long_longitude` and contains all `14` strict ties.

Holding `Delta H` fixed at the reference while allowing only the actual `H0` to vary reproduces every direct generalized eigenvalue within `1.665e-16`. Conversely, holding `H0` fixed while allowing only the tiny computed `Delta H` roundoff to vary leaves a spread of only `1.943e-16`.

## Boundary and exact-tie checks

The finite 5 by 5 boundary permits four spatial variants, all at the same center. Each axis assignment has two mirror variants and seven time translations, so the boundary does not favor one assignment by availability or placement. It limits the class, but it does not create the objective split.

Within each spatial orientation, the maximum time-translation spread is `1.943e-16`. Mirror variants within an axis assignment are also tied to numerical precision.

As a denominator-only sensitivity check (not a refit), the fitted spatial-range product was preserved while its longitude/latitude ratio was set exactly to the target ratio. All 28 objectives then span only `1.943e-16`.

## Conclusion

The strict reported orientation was selected by `H0`, not by the isotropic numerator and not by unequal boundary availability. More precisely, the split is caused by the `+7.162e-08` relative residual in the fitted-null spatial range ratio. It creates a `3.764e-09` objective gap: large enough to exceed the stored strict tie tolerance, but not evidence of a scientifically meaningful directional interaction. With exact axis symmetry, all 28 variants are a symmetry tie.

## Reproduction

From the diagnostic directory:

```bash
python audit_denominator_orientation_preference.py
```
