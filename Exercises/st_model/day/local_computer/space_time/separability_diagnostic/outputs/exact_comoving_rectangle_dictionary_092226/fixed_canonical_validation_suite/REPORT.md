# Fixed canonical contrast: small validation suite

The 14 strict-tie filters from the original exact-comoving oracle were fixed before defining these scenarios. This run performs no filter search and no covariance fitting.
Input oracle: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226`.

## Statistic and calibration

For each replicate, the statistic is the mean of the 14 squared canonical filter outputs divided by their scenario-specific separable-null variances. Its population expectation is one under the known null. Because the filters are correlated, null quantiles are calibrated from the exact 14-dimensional Gaussian filter covariance rather than from an independence approximation.

Calibration, independent null evaluation, and alternative evaluation each use `50,000` reduced Gaussian replicates per scenario. The pre-specified direction is the lower tail, matching the negative discrepancy found in the training oracle; a two-sided rate is also reported descriptively.

| scenario | population mean energy | null lower-tail rate | alternative lower-tail rate | alternative two-sided rate |
|---|---:|---:|---:|---:|
| `separable_null` | 1.000000 | 0.0495 | 0.0483 | 0.0484 |
| `joint_matern_training` | 0.814927 | 0.0486 | 0.1061 | 0.0607 |
| `joint_matern_time_range_1` | 0.882895 | 0.0514 | 0.0767 | 0.0470 |
| `joint_matern_time_range_4` | 0.775643 | 0.0513 | 0.1312 | 0.0723 |
| `mixture_aligned` | 0.953585 | 0.0509 | 0.0611 | 0.0483 |
| `mixture_crossed` | 1.046415 | 0.0500 | 0.0420 | 0.0565 |

## Scenarios

- `separable_null`: an exactly separable process; its alternative covariance equals its null covariance.
- `joint_matern_training`: the original joint Matérn-half population benchmark.
- `joint_matern_time_range_1` and `joint_matern_time_range_4`: fixed-filter parameter perturbations with their own matched-margin separable comparators.
- `mixture_aligned`: an equal mixture of short-space/short-time and long-space/long-time separable components.
- `mixture_crossed`: an equal mixture of short-space/long-time and long-space/short-time components. The aligned and crossed mixtures have identical spatial and temporal margins and therefore share the same matched separable comparator.

Mixture settings: short spatial multiplier `0.6`, long spatial multiplier `2.0`, short temporal range `1.0`, long temporal range `4.0`, weight `0.5`.

## Interpretation boundary

This is a controlled known-parameter validation of a pre-fixed statistic. A null rate near 0.05 checks implementation and calibration without selection bias. Alternative rejection rates show whether the original canonical contrast transfers to these specified populations. They do not establish a universally powerful interaction diagnostic, and they do not yet include same-data covariance fitting or real-data uncertainty.

## Reproduction

```bash
python validate_fixed_canonical_contrast_suite.py
```
