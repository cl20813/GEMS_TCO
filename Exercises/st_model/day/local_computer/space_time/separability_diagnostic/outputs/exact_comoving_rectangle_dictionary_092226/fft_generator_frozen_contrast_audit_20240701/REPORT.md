# FFT generator audit for the frozen contrast

No simulation, refitting, or contrast selection was performed. Expectations were evaluated directly under the analytic and generated covariance matrices for every complete frozen contrast used in the real-data pilot.

Frozen translated contrast count: `29576`. Joint-to-separable target Var(L) gap: `0.36550128568390061`.

## Diagnostic-specific result

| factor | DGP | target Var(L) | FFT Var(L) | total bias | |bias| / |model gap| | negative spectral mass |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `joint` | 1.38118167 | 1.59359723 | 0.212415562 | 0.581162 | 0.045741 |
| 2 | `separable` | 1.74668295 | 1.79298338 | 0.0463004237 | 0.126677 | 0.00305619 |
| 4 | `joint` | 1.38118167 | 1.49129733 | 0.11011566 | 0.301273 | 0.0417199 |
| 4 | `separable` | 1.74668295 | 1.78116517 | 0.0344822149 | 0.0943423 | 0.00127524 |
| 8 | `joint` | 1.38118167 | 1.41415197 | 0.0329702996 | 0.0902057 | 0.032929 |
| 8 | `separable` | 1.74668295 | 1.77523627 | 0.0285533148 | 0.078121 | 0.000468005 |

`mapping_bias` compares analytic covariance at nearest FFT cells with analytic covariance at actual source coordinates. `spectral_correction_bias` then compares the clipped/rescaled embedding covariance with the analytic covariance at those FFT cells. Their sum is the total generator bias.

## Source-coordinate mapping

| HR factors | observations over 8 hours | duplicate assignments | exact matches | max distance (degrees) | max standardized distance |
|---|---:|---:|---:|---:|---:|
| 1x1 | 131428 | 1742 | 0 | 0.037692568 | 0.041768642 |
| 2x2 | 131428 | 16 | 0 | 0.019192658 | 0.021319591 |
| 4x4 | 131428 | 0 | 0 | 0.0095875589 | 0.010646429 |
| 8x8 | 131428 | 0 | 0 | 0.0047913025 | 0.0053228326 |
| 100x10 | 131428 | 0 | 0 | 0.003157576 | 0.0033171957 |

Frozen Var(L) mapping bias (without spectral clipping):

| HR factors | DGP | target Var(L) | mapped Var(L) | mapping bias | |bias| / |model gap| |
|---|---|---:|---:|---:|---:|
| 1x1 | `joint` | 1.38118167 | 1.38211338 | 0.000931710903 | 0.00254913 |
| 1x1 | `separable` | 1.74668295 | 1.77179004 | 0.0251070826 | 0.0686922 |
| 2x2 | `joint` | 1.38118167 | 1.3829133 | 0.00173163568 | 0.0047377 |
| 2x2 | `separable` | 1.74668295 | 1.78065934 | 0.0339763875 | 0.0929583 |
| 4x4 | `joint` | 1.38118167 | 1.38152482 | 0.000343157114 | 0.000938867 |
| 4x4 | `separable` | 1.74668295 | 1.76028077 | 0.0135978151 | 0.0372032 |
| 8x8 | `joint` | 1.38118167 | 1.38133658 | 0.000154911923 | 0.000423834 |
| 8x8 | `separable` | 1.74668295 | 1.75161006 | 0.00492710721 | 0.0134804 |
| 100x10 | `joint` | 1.38118167 | 1.38121549 | 3.38217575e-05 | 9.25353e-05 |
| 100x10 | `separable` | 1.74668295 | 1.7488992 | 0.00221624409 | 0.00606357 |

## Interpretation rule

The spectrally corrected FFT generator should be treated as a primary approximation only if its frozen-diagnostic bias is small relative to the joint-versus-separable target gap and is stable as the embedding expands. Nonnegative-eigenvalue status is reported rather than inferred from generic covariance error.
