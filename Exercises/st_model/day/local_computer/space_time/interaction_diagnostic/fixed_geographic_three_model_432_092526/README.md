# Local fixed-geographic 4/3/2 audit

This is a one-day local reproduction of the frozen Amarel diagnostic on
`2024-07-02`. It changes only the Vecchia conditioning budget and compute
chunk relative to the Amarel study:

- fixed geographic A/B contrasts and lag 1;
- generalized Cauchy, Matérn `nu=0.5`, and advected-separable models;
- corridor conditioning counts `4/3/2`;
- spatial target blocks remain `4 x 4` (at most 16 points per block);
- `target_chunk_sizes = 64` for each model means 64 target blocks are evaluated together by
  vectorized tensor operations; it does not change the block geometry;
- CPU execution, so it is independent of the Amarel GPU OOM;
- the updated compiled CPU kernel is used for generalized Cauchy covariance.

The Matérn class is the package's spline-capable implementation. At
`nu=0.5`, it deliberately uses the exact exponential Matérn formula rather
than evaluating a Bessel function or spline table. Generalized Cauchy is a
different covariance family and uses its analytic power correlation.

Run:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1
/opt/anaconda3/envs/faiss_env/bin/python \
  Exercises/st_model/day/local_computer/space_time/interaction_diagnostic/fixed_geographic_three_model_432_092526/run_local_fixed_geo_three_model_432.py
```

The output is written under `outputs_current/2024-07-02/`. The versioned
directory is intentional: results under `outputs/2024-07-02/` have a study
signature from before the package/API update and must not be mixed with a new
fit. Completed fits with the current signature are reused on rerun.

Passing a strict subset such as `--models gc matern05` now forwards the shared
runner's explicit `--allow-model-subset` safety flag.

The older successful July-1 demonstration is not a like-for-like benchmark:
it moved the endpoints along fitted advection and compared a fitted joint GC
with a matched-separable covariance derived from that same fit. This audit
uses fixed geographic endpoints and independently fits all three models.
