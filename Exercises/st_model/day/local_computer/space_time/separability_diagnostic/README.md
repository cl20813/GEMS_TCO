# Advected-separable null diagnostic

This folder is an isolated, reproducible pilot for comparing a strong
advected-separable baseline with the joint space-time Matern model used by the
simulation generator.  It does not modify the production Vecchia likelihood.

## Models

For latitude/longitude lag `h`, time lag `u`, and advection `v`, the true
`nu=1/2` model is

```text
C1(h,u) = sigma1^2 exp(-sqrt(
    ((h_lat-v_lat*u)/range_lat)^2
  + ((h_lon-v_lon*u)/range_lon)^2
  + (u/range_time)^2))
```

The null is separable after moving to `h* = h-vu`:

```text
C0(h,u) = sigma0^2
          exp(-sqrt((h*_lat/range_lat)^2 + (h*_lon/range_lon)^2))
          exp(-abs(u)/range_time)
```

Thus the spatial margin at `u=0` and the temporal margin along `h=vu` use the
same exponential family.  The remaining difference is joint Euclidean
combination in `C1` versus a spatial-times-temporal product in `C0`.

The best null is the bounded multistart minimizer of

```text
log|Sigma0(theta)| + trace(Sigma0(theta)^-1 Sigma1),
```

with `sigma0^2` profiled analytically.  The diagnostic directions solve

```text
Sigma1 w_j = lambda_j Sigma0 w_j,
w_j' Sigma0 w_k = 1(j=k),
```

and are ranked by

```text
g(lambda) = 0.5 * (lambda - 1 - log(lambda)).
```

This criterion keeps important directions on both sides of one: the true
model can have either more or less variance than the null.

## Data contract

The input is the exact five-day nugget-zero asset under
`outputs/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_matched5_090226`.
The five dates are independent eight-hour simulation blocks, not one
continuous 40-hour process.  Time is reset to `0,...,7` within every date.

Covariances use `Source_Latitude` and `Source_Longitude`, because those are the
locations at which the latent field was sampled.  The known simulation mean is
removed at those source locations.  The statistical nugget is exactly zero in
both models.  A default `1e-10 * variance` diagonal regularizer is used only for
dense numerical linear algebra and is recorded separately in the run manifest.

## Subset and selection-bias control

The default dense problem has 100 spatial anchors times eight hours (`n=800`).
Anchors must remain observed in all five dates while their grid cells move by
the truth advection.  A deterministic anisotropic max-min rule then spreads
the anchors across the common-valid moving tube.

The first three independent dates are the design split: their theoretical
covariances determine the KL null and generalized eigenvectors.  Response
values are not used.  The final two dates are held out for projection scores.
The fixed directions are renormalized to unit null variance on each held-out
coordinate pattern.  A parametric bootstrap calibrates both the maximum
squared projection and a top-subspace Gaussian likelihood ratio.

This is an oracle alternative-specific simulation diagnostic.  If a future
analysis estimates a composite null, parameter uncertainty remains even with
an independent split.  A publication-size calibration must simulate the
complete training and evaluation experiment and repeat mean, nugget,
advection, range fitting, diagnostic construction, and every scale/mode rule
inside each bootstrap replicate.  The generalized directions used here also
depend on the known joint-Matern simulation alternative and are therefore an
alternative-specific power diagnostic, not a universal separability test.

## Run

From the repository root:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/run_nugget0_five_day.py
```

Useful smoke-test settings are:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/run_nugget0_five_day.py \
  --spatial-anchors 20 \
  --optimizer-starts 2 \
  --bootstrap-replicates 500 \
  --output-dir /tmp/gems_separability_smoke
```

The response-free mixed-lag geometry audit is available separately:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/analyze_mixed_lag_geometry.py
```

The audit marks the structural location `s=t` at a fixed standardized joint
radius without imposing a paired-anchor contrast.  It also records why
replacing `sqrt(s^2+t^2)` with `s^2+t^2` changes the margins and the covariance
objective even though it produces a separable expression.

To inspect all generalized eigen-directions before designing an interpretable
interaction contrast, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/explore_eigen_directions.py
```

This analysis reads coordinate columns only.  It compares the fitted-null
eigenbasis with a pure same-margin interaction eigenbasis, decomposes each
direction into intrinsic interaction and fitted-null compensation, checks
design-day stability and near-degenerate clusters, and attributes each
candidate's quadratic-form discrepancy to moving spatial and temporal lag
bins.  For interaction-dominant two-mode clusters it also writes raw
space-time weights, rotation-invariant subspace amplitude and Gram heatmaps,
a joint temporal-DCT by spatial graph-Fourier spectrum, and cluster-level lag
attribution.  The heatmap's Fiedler-ordered columns are a one-dimensional
display of irregular two-dimensional anchors, not spatial rectangles.  Its
joint spectrum is Euclidean filter-weight energy, not a covariance-variance
or KL decomposition, and is conditional on the chosen spatial graph.  Its
candidate labels organize exploration; they do not select a final held-out
test.

The rectangle-dictionary oracle is a separate experiment because the observed
flow tube uses rounded grid shifts: equal `anchor_rank` values are not exactly
equal physical locations in truth-moving coordinates.  Run the exact-comoving
5-by-5 spatial grid experiment with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/run_rectangle_dictionary_oracle.py
```

Its settings are predeclared in `rectangle_dictionary_exact_grid.toml`.  The
script fits a zero-nugget separable null while holding the known advection
fixed, checks the exact matched margins, builds all 8,400 physical rectangle
contrasts, removes dictionary redundancy by SVD, solves both signs of the
intrinsic generalized problem, and compares single, greedy 2/4/8, dense
dictionary, and unrestricted filters.  Existing atlas modes receive only an
algebraic double-centering-span audit; that audit is explicitly not a claim
that the warped flow-tube observations form fixed physical rectangles.

To inspect the first signed interaction that is absent from every single
rectangle but appears at the minimizing two-rectangle step, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/analyze_negative_two_rectangle.py
```

This deterministic post-processing step rehydrates the oracle covariance
matrices from its manifest, resolves the two-by-two generalized problem at
full precision, and separates both positive diagonal terms from the negative
cross-rectangle term.  It also saves the exact eight nonzero observation
weights, fitted-null compensation, intrinsic and total lag attributions, and
the constrained eigen residual.  Its pair is conditional on the declared
greedy tie-breaking path and is not an exhaustive globally optimal pair.

To replace that greedy pair by an exhaustive search over all
`choose(8,400, 2) = 35,275,800` unordered pairs, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/run_global_two_rectangle_search.py
```

The search evaluates float64 blocks without retaining all pair scores,
handles nearly singular two-column null Gram matrices separately, validates
the screened minima with SciPy, and repeats the search with a second block
partition.  Strict ties use the oracle's declared tolerance; a wider,
explicitly labeled geometry-sensitivity screen records symmetry-near rotated
copies.  The equivalence classifier is canonical under pair exchange,
translation, endpoint reversal, and square-grid rotations/reflections.  Its
objective is the intrinsic contrast `Sigma1-SigmaM`, normalized by `Sigma0`;
it is not a global search for `Sigma1-Sigma0` or for projected KL.

After a pilot finishes, evaluate the predeclared number-of-modes path with:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/analyze_mode_count_path.py \
  --pilot-dir \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/nugget0_five_day_092226
```

This recomputes directions from the design covariance only.  Held-out
responses enter solely through the displayed test statistic.  The complete
`K` grid is fixed before those responses are read, and the null/power
calibration uses the exact projected generalized eigenvalues with chunked
Monte Carlo.  A publication test must choose `K` using a training-only rule
such as 80% or 90% cumulative KL; choosing the smallest displayed p-value
would introduce selection bias.

The current mode path uses eigenvectors of the covariance averaged across the
three design days.  Its cumulative fraction is therefore the KL of that
reference average-covariance problem, not the sum of the three day-specific
KL values.  The full-dimension LLR is basis-invariant and does not depend on
this top-`K` approximation.

Run the numerical unit tests with:

```bash
/opt/anaconda3/bin/python -m pytest -q \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/tests
```

## Outputs

The default output folder is `outputs/nugget0_five_day_092226` inside this
directory.  Important files are:

- `run_manifest.json`: complete data, split, parameter, numerical, and timing record.
- `null_fit_attempts.csv`: every optimization start, including local minima.
- `day_kl_comparison.csv`: KL-null versus the simple margin-matched null.
- `generalized_eigenvalues.csv`: `lambda`, `log2(lambda)`, `g(lambda)`, and cumulative KL.
- `selected_directions_long.csv`: interpretable long-form weights; no opaque binary archive.
- `heldout_projection_scores.csv`: fixed-direction scores for the two held-out days.
- `bootstrap_summary.json`: critical values, p-values, and oracle power.
- `RESULTS.md`: concise interpretation and explicit limits.
- `figures/`: covariance slices, spectrum, direction maps, and bootstrap distributions.
- `mixed_lag_geometry_audit/`: response-free occupancy and analytic
  same-margin gap summaries in standardized moving-lag coordinates.
- `eigen_direction_atlas/`: metrics for all fitted-null and same-margin
  interaction eigen-directions, design-day stability, near-degenerate
  clusters, raw and rotation-invariant space-time cluster heatmaps, candidate
  weights, lag attribution, figures, and an interpretation report.
- `../exact_comoving_rectangle_dictionary_092226/`: separate exact-grid
  rectangle metadata, rank diagnostics, greedy paths and coefficients,
  intrinsic-versus-fitted-null metrics, fixed-filter simulations, figures,
  and a report.  It contains no response-based inference.
- `mode_count_path/`: exact held-out LLR, calibration, and oracle power over a
  predeclared generalized-eigenmode count grid.

## Publication-scale continuation

The pilot should be accepted only after symmetry, generalized-eigen residual,
`W' Sigma0 W = I`, and `sum g(lambda) = KL` checks pass.  The next scale-up is
to repeat the immutable pipeline for nested 100/200/400-location tubes and
several coordinate-only seeds, then use at least 1,000 complete null and
alternative simulations.  For the full GEMS domain, fit the null with fixed
corridor Vecchia conditional KL and replace dense eigenanalysis with a
matrix-free null-whitened operator; the dense pilot remains its reference.
