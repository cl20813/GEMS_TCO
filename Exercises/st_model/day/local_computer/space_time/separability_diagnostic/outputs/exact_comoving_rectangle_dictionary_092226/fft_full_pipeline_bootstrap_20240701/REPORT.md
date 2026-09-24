# Frozen-contrast FFT/circulant full-pipeline bootstrap

The latent residual field was generated on a complete lattice, sampled at the original source coordinates, masked with the original O3 missingness pattern, and then re-fitted with the unchanged 4/3/2 corridor-Vecchia plus GLS pipeline. Geometry, temporal lag, coefficients, advection path, and grid rule were frozen.

## Embedding validity

| DGP | generator classification | negative spectral mass | max audited error / variance | RMSE / variance |
|---|---|---:|---:|---:|
| `joint` | `spectrally_corrected_circulant_embedding` | 0.045741 | 0.0271408 | 0.0107451 |
| `separable` | `spectrally_corrected_circulant_embedding` | 0.00305619 | 0.00224124 | 0.000674938 |

A generator is called exact only when the unmodified embedding spectrum is nonnegative. Otherwise the report deliberately uses `spectrally_corrected_circulant_embedding`: negative eigenvalues were clipped and the spectrum was rescaled to restore the target marginal variance. See `embedding_diagnostics.csv` and `lag_covariance_audit.csv`; this approximation must be disclosed in any inferential use.

Observed statistic: `0.74155783`.

| generating model | replicates | mean | 2.5% | 97.5% | lower-tail p-value |
|---|---:|---:|---:|---:|---:|
| `joint` | 1 | 0.902221 | 0.902221 | 0.902221 | 0.500000 |
| `separable` | 1 | 4.922374 | 4.922374 | 4.922374 | 0.500000 |

**Status: computational pilot only.** Fewer than 99 replicates per requested DGP are available, so the tail probabilities are not confirmatory.

The earlier block-Vecchia-generated replicate is not pooled here; it remains a generator-sensitivity result. This directory contains only FFT/circulant-generated replicates.

## Run configuration

```json
{
  "data_file": "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl",
  "fit_csv": "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/spectrum_diagnostics/outputs/real_july2024_st_heads_vecchia_lag432_one_day_gc_a075_b1_nugget0_061626/heads_vecchia_lag432_one_day_fit_summary.csv",
  "coefficient_source": "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226/global_two_rectangle_search/global_pair_ties.csv",
  "day": "2024-07-01",
  "strategy": "standard_432",
  "dgp": "both",
  "requested_replicates_per_dgp": 1,
  "seed": 20260925,
  "device_for_refit": "cpu",
  "optimizer": {
    "max_steps": 4,
    "max_eval": 20,
    "grad_tol": 0.0001,
    "tolerance_grad": 1e-05
  },
  "fft_grid": {
    "lat_factor_hr": 1,
    "lon_factor_hr": 1,
    "base_lat_step": 0.044,
    "base_lon_step": 0.063,
    "pad": 0.1,
    "n_lat": 119,
    "n_lon": 163,
    "embedding_spatial_factor": 2,
    "embedding_temporal_factor": 2,
    "spectral_correction": "clip negative eigenvalues then rescale to target variance"
  },
  "mask_and_sampling": "nearest high-resolution cell at original source coordinates; preserve original O3 missingness exactly",
  "analysis_refit": "joint GC 4/3/2 corridor Vecchia plus GLS mean for both generating models",
  "frozen_geometry": {
    "range_lat": 0.8152988171176069,
    "range_lon": 0.9527144962204038,
    "advec_lat": -0.0126294742027098,
    "advec_lon": -0.1941944244440873,
    "temporal_lag": 1,
    "coefficient_first": 0.156629367196063,
    "coefficient_second": 0.1540219594610364
  }
}
```
