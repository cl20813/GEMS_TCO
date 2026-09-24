# Pilot status

Do not use this directory for tail calibration.

- The `joint` row is a valid end-to-end pipeline pilot generated with the
  2x spectrally corrected circulant approximation.  The later
  frozen-contrast generator audit showed that this approximation biases
  expected `Var(L)` by about 58% of the joint-versus-separable model gap, so it
  is not an acceptable primary bootstrap generator.
- The `separable` row is a failed refit-path pilot.  It was generated from the
  matched-separable covariance but incorrectly re-fitted with the joint-GC
  model, whose ranges ran to the boundary.  Its statistic must not be used.
- No replicate in this directory is pooled with the matrix-free Lanczos
  sensitivity or with a future calibrated bootstrap.

The generator-specific audit is in
`../fft_generator_frozen_contrast_audit_20240701/`; the unclipped matrix-free
reference is in `../lanczos_generator_sensitivity_20240701/`.
