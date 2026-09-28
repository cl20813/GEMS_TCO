# Matérn-truth GC versus Matérn two-day audit

This audit uses two already generated, independent daily fields:

- `2024-07-13`
- `2024-07-19`

Both assets have analytic joint Matérn truth with `nu=0.5`, signal variance
`10`, latitude/longitude/time ranges `(0.2, 0.3, 2.0)`, advection
`(0.08, -0.2)`, and nugget `0`. No new simulation or contrast search is run.
The two dates are independent eight-hour FFT blocks with different recorded
seeds. The `gridded` asset reproduces the real-data nearest-cell assignment and
half-cell rejection rule. Covariance fitting nevertheless uses the retained
observations' original irregular `Source_Latitude`/`Source_Longitude`; the
regular `Latitude`/`Longitude` only index translated filter locations.

The frozen fixed-geographic A/B geometry, both mirrors, lag 1, and all valid
translated anchors are retained. GC and Matérn are fit independently using
corridor `4/3/2`, `4x4` target blocks, CPU target chunk `64`, float64, and the
same truth-based initial physical parameters. Matérn smoothness is fixed at
`0.5`; GC shape is fixed at `alpha=0.75`, `beta=1`.

Run from the repository root:

```bash
PYTHONPATH=src /opt/anaconda3/envs/faiss_env/bin/python \
  Exercises/st_model/day/local_computer/space_time/interaction_diagnostic/matern_truth_gc_vs_matern_two_days_092526/run_matern_truth_gc_vs_matern.py
```

The primary requested comparison is fitted versus analytic-truth `C_AB` over
the exact same valid contrast samples. The output also records empirical
`C_AB`, full `2x2` covariance error to truth, pointwise covariance RMSE,
observed bivariate contrast score, Vecchia NLL, fitted parameters, and timing.

The stored fields came from the historical circulant generator, which clipped
negative embedding eigenvalues. Therefore `analytic truth` means the intended
Matérn target covariance; it does not include the generator's unrecorded
clipping-induced covariance perturbation. This limitation is kept explicit in
the report.
