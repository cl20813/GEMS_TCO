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
analysis fits both models or selects directions from the same observed
responses being tested, every fit and selection step must be repeated inside
each bootstrap draw, or an independent split/cross-fit must be retained.

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

To audit the square-root contrast without refitting or reading any response
column, run:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/analyze_balanced_mixed_lags.py
```

This audit reports the analytic same-margin gap separately from the actual
gap after the null has compensated through its fitted ranges and advection.
Literally removing the outer square root gives
`exp(-(s^2+t^2)) = exp(-s^2) exp(-t^2)`, which is separable.  It also changes
the axis margins from exponential to squared-exponential, so it does not in
general preserve the likelihood objective or its optimizer.  The useful
consequence is instead geometric: at a fixed joint radius, the original
nonseparable-versus-separable contrast is largest at balanced mixed lags
`s=t`.

To repeat the full fit with 50 non-overlapping local trajectory pairs at the
near-balanced `(2, 2)` grid offset, while retaining the original `n=800`
dimension, run:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/run_nugget0_five_day.py \
  --anchor-design balanced_pairs \
  --pair-offset 2 2 \
  --output-dir \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/nugget0_balanced_rectangles_092226
```

The paired-anchor order supports the adjacent-time moving rectangle

```text
Z(A,t) - Z(B,t) - Z(A,t+1) + Z(B,t+1).
```

The runner saves both the individual rectangle variance ratios and the
generalized spectrum of their joint subspace.  The KL-optimal null is refit
for the changed geometry; parameters from the max-min-tube fit are never
reused.

After a pilot finishes, evaluate the predeclared number-of-modes path with:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/analyze_mode_count_path.py \
  --pilot-dir \
  Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/nugget0_balanced_rectangles_092226
```

This recomputes directions from the design covariance only.  Held-out
responses enter solely through the displayed test statistic.  The complete
`K` grid is fixed before those responses are read, and the null/power
calibration uses the exact projected generalized eigenvalues with chunked
Monte Carlo.  A publication test must choose `K` using a training-only rule
such as 80% or 90% cumulative KL; choosing the smallest displayed p-value
would introduce selection bias.

Run the numerical unit tests with:

```bash
/opt/anaconda3/envs/gems_gpu/bin/python -m pytest -q \
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
- `balanced_mixed_lag_audit/`: response-free square-root geometry, selected-pair
  occupancy, and fitted-null compensation audit.
- `selected_anchor_pairs.csv`: deterministic local pair endpoints and max-min centres
  for a paired design.
- `moving_rectangle_contrasts.csv`: adjacent-time double-difference variances,
  variance ratios, and per-contrast `g(lambda)`.
- `rectangle_generalized_eigenvalues.csv`: full correlated rectangle-subspace
  generalized spectrum.
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
