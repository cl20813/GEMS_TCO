# Frozen-contrast full-pipeline parametric bootstrap

The canonical A/B geometry, lag one, raw coefficient ratio, fitted spatial scale, fitted advection path, and nearest-grid mapping were frozen before this bootstrap. Every replicate uses the original GEMS missingness pattern, is simulated jointly through the fixed 4/3/2 block-Vecchia graph, and is analyzed after re-fitting the joint generalized-Cauchy covariance and GLS mean.

Observed statistic (empirical mean L^2 / refitted matched-separable Var(L)): `0.74155783`.

| generating model | replicates | mean | 2.5% | 97.5% | lower-tail p-value |
|---|---:|---:|---:|---:|---:|
| `separable` | 1 | 4.194806 | 4.194806 | 4.194806 | 0.500000 |

**Status: computational pilot only.** The current replicate count is too small for confirmatory tail calibration. The driver is resumable; run it to at least 99, preferably 199 or more, replicates per generating model on CUDA before interpreting p-values.

The `joint` generating distribution checks compatibility with the fitted joint GC. The `separable` distribution checks whether the observed lower standardized energy is unusual under the matched-separable construction. Both distributions repeat joint-GC refitting because that is the frozen analysis pipeline used to construct the comparator.

This remains a parametric, model-based calibration. It does not replace external validation on held-out days.

## Run configuration

```json
{
  "data_file": "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl",
  "fit_csv": "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/spectrum_diagnostics/outputs/real_july2024_st_heads_vecchia_lag432_one_day_gc_a075_b1_nugget0_061626/heads_vecchia_lag432_one_day_fit_summary.csv",
  "coefficient_source": "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226/global_two_rectangle_search/global_pair_ties.csv",
  "day": "2024-07-01",
  "strategy": "standard_432",
  "dgp": "separable",
  "requested_replicates_per_dgp": 1,
  "seed": 20260924,
  "device": "cpu",
  "max_steps": 4,
  "max_eval": 20,
  "grad_tol": 0.0001,
  "tolerance_grad": 1e-05,
  "frozen_geometry": {
    "range_lat": 0.8152988171176069,
    "range_lon": 0.9527144962204038,
    "advec_lat": -0.0126294742027098,
    "advec_lon": -0.1941944244440873,
    "temporal_lag": 1,
    "coefficient_first": 0.156629367196063,
    "coefficient_second": 0.1540219594610364,
    "nearest_grid_rule": "axis-wise nearest regular cell within half a grid step"
  },
  "simulation": "exact block-conditional simulation from the fixed Vecchia 4/3/2 graph",
  "analysis_refit": "joint GC plus GLS mean for both generating models"
}
```
