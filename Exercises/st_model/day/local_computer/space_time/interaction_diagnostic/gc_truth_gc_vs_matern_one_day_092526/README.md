# One-day GC-truth FFT audit: GC versus Matérn

This study is the reverse control for the existing two-day Matérn-truth audit.
It generates one eight-hour generalized-Cauchy (GC) field locally, fits GC and
Matérn independently with the same Vecchia design, and compares their fitted
cross term against known population truth.

## Fixed experiment

- calendar day: `2024-07-13` (not the July 1 design-discovery day);
- fixed-geographic frozen A/B filters, both mirrors, temporal lag 1;
- all valid translated anchors;
- source-coordinate covariance fitting;
- corridor `4/3/2`, 4x4 target blocks, CPU chunk 64;
- signal variance `10`, temporal range `2`, advection `(0.08, -0.2)`;
- GC shape `alpha=0.75`, `beta=1`, nugget fixed to zero;
- spatial ranges `(0.8152988171, 0.9527144962)`.

The spatial ranges intentionally match the scale used to freeze the A/B
geometry. The historical GC generator defaults `(0.2, 0.3)` make analytic
`C_AB` only about `0.07`, which is too weak for an informative one-realization
mechanism check. This study therefore tests recovery of the cross term rather
than reproducing that low-signal default verbatim.

## Why the generator is different

The historical generator inserts advection directly into an even BCCB first
column, discards the FFT imaginary component, and clips negative real
eigenvalues. At the Nyquist faces that advected column is not centrally
symmetric, so the correction can manufacture or erase a substantial fraction
of the targeted cross term.

The new local generator in
`simulate_data/generate_one_day_gc_fft_local.py` uses the equivalent moving
coordinate construction:

```text
Z(s,t) = W(s - v * (t - t0), t)
```

`W` is a zero-advection joint GC field. Its covariance is coordinate-wise even,
so the odd `2n-1` real FFT embedding has no advection/Nyquist imaginary
artifact. Any small negative spectrum is measured, thresholded, clipped, and
renormalized to variance 10. The report compares fitted models with both:

1. intended analytic GC covariance;
2. effective post-clipping FFT covariance, the exact covariance that generated
   the finite lattice field.

The effective FFT target is primary. The generator also preserves the real
template's `Hours_elapsed`; local `t=0,...,7` is used only inside simulation.

The production setting uses a 10x10 source-sampling grid and usually peaks at
several GiB on the 48GB local Mac. A 2x2 or 4x4 run is suitable only as a wiring
smoke test because coordinate snapping is coarser.

## Run

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /opt/anaconda3/envs/faiss_env/bin/python \
  Exercises/st_model/day/local_computer/space_time/interaction_diagnostic/\
gc_truth_gc_vs_matern_one_day_092526/run_gc_truth_gc_vs_matern.py
```

Useful modes:

```bash
# Generate/audit truth and instantiate both current model APIs, but do not fit.
.../run_gc_truth_gc_vs_matern.py --prepare-only

# Deterministically replace the generated day in a fresh output root.
.../run_gc_truth_gc_vs_matern.py --regenerate --output-root /path/to/fresh-output

# Rebuild only the comparison CSV and Markdown report from completed fits.
.../run_gc_truth_gc_vs_matern.py --summary-only
```

Use a fresh `--output-root` after changing the config, package source, shared
runner/core, native extension, or generator. The study signature includes all
of those inputs and refuses to mix incompatible cached fits.

## Outputs

- `simulation/gc_fft_2024-07-13_{real_locations,gridded}.pkl`
- `simulation/gc_fft_2024-07-13_manifest.csv`
- `simulation/gc_fft_2024-07-13_truth.json`
- `run_manifest.json`
- `empirical_contrasts.csv.gz`
- analytic and effective-FFT truth prediction files
- `model_gc/` and `model_matern05/` fit/prediction/error records
- `model_summary.csv`
- `cross_term_comparison.csv`
- `RESULTS.md`
- `COMPLETE`

The comparison reports `C_AB`, the derived contribution
`2*d1*d2*C_AB`, full 2x2 covariance errors, population score regrets,
one-realization empirical contrast scores, Vecchia NLL, convergence, and FFT
integrity diagnostics.

## Interpretation boundary

This is one simulated day and one field realization. It is a mechanism audit,
not a power calculation or a universal GC-versus-Matérn ranking. Overlapping
within-day contrasts are not independent replicates; the population truth
columns are more informative than the noisy empirical `C_AB` for this run.
