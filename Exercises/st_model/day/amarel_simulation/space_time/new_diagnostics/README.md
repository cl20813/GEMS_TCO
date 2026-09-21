# Wavelet residual-energy diagnostic

`wavelet_residual_energy_one_day.py` runs a local-CPU, one-day comparison of
the fitted adapted and fixed lag-6/4/3 Vecchia models for 2024-07-03.

The script uses the regular cells in `tco_grid_24_07.pkl` and reapplies the
debiased-Whittle half-cell rule. A cell is missing unless its source location is
within `0.5 * 0.044` degrees in latitude and `0.5 * 0.063` degrees in longitude
of the regular cell. Missing cells are passed through the exact same mask in
all fitted-model simulations.

Run from the repository root:

```bash
python Exercises/st_model/day/amarel_simulation/space_time/new_diagnostics/wavelet_residual_energy_one_day.py
```

The default run uses 32 independent simulations to estimate every wavelet
coefficient's model variance and 64 additional simulations for the 95% null
envelopes. The finite-calibration inverse-variance bias is corrected by the
Gaussian factor `m/(m-2)`, so the independently simulated null energy remains
centered at one. A fast smoke test is:

```bash
python Exercises/st_model/day/amarel_simulation/space_time/new_diagnostics/wavelet_residual_energy_one_day.py \
  --calibration-simulations 8 \
  --envelope-simulations 8 \
  --output-root outputs/summer_26/wavelet_energy_smoke
```

Outputs include:

- `wavelet_cumulative_diagnostic.png`: coarse-to-fine cumulative energy against
  the fitted-model `y=x` reference;
- `wavelet_scale_orientation_diagnostic.png`: low/middle/high energy ratios,
  separated into latitude, longitude, diagonal, and pooled details;
- `wavelet_spatial_energy_maps.png`: spatial localization of excess or deficient
  energy;
- CSV files containing the plotted values; and
- `run_summary.json` with thresholds, fitted parameters, embedding checks, and
  timings.

The reference covariance is the full stationary fitted Matérn-0.5 covariance
on the regular lattice. It is deliberately not called an eigendecomposition of
the ordering-dependent Vecchia precision. The wavelet levels provide the
physical scale ordering.

## Controlled misspecification validation

`wavelet_simulation_misspecification.py` checks whether those physical scales
actually react to known covariance errors. It uses the existing smoothness-0.5
2024-07-13 simulation and its realistic missing-cell mask, then compares the
true covariance with longitude ranges 0.5x and 2x truth, Matérn smoothness 0.3
and 1.0, and nugget 0 instead of 1. The non-closed-form Matérn correlations use
the same spline-coefficient builder as `GEMS_TCO.vecchia_st_spline`.

```bash
python Exercises/st_model/day/amarel_simulation/space_time/new_diagnostics/wavelet_simulation_misspecification.py
```

In addition to the stored simulated day, the script generates independent
true-model days on the same regular grid and mask. These repeated simulations
estimate rejection rates, so one atypical realization is not mistaken for
diagnostic power. The principal outputs are:

- `localized_6x3_diagnostic.png`: one row per assumed model and columns for
  low D3, middle D2, and high D1, each against its own `y=x` envelope;
- `diagnostic_performance_heatmaps.png`: stored-day energy departures and
  repeated-simulation detection rates;
- `localization_summary.csv`: the strongest band ranked by absolute log2
  energy error, plus bandwise rejection rates; and
- full curve and orientation-level CSV files for reproducible plotting.

This validation uses the full stationary covariance and FFT simulation. It
does not use Vecchia, eigenvectors, Lanczos, or SLQ. Consequently it tests the
wavelet diagnostic itself, without confounding the result with a spectral
approximation.
