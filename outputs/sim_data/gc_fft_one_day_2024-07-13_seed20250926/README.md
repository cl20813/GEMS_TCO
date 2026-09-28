# Preserved one-day GC FFT simulation

This directory is a non-destructive preservation copy of the simulation assets
created by the local study
`gc_truth_gc_vs_matern_one_day_092526` for `2024-07-13`.

- seed: `20250926`
- truth family: generalized Cauchy (`alpha=0.75`, `beta=1`)
- signal variance: `10`
- ranges (latitude, longitude, time):
  `(0.8152988171176069, 0.9527144962204038, 2.0)`
- advection (latitude, longitude): `(0.08, -0.2)`
- nugget: `0`
- construction: zero-advection comoving FFT field sampled at
  `Z(s,t) = W(s - v*(t-t0), t)`

The original study output remains at:

`Exercises/st_model/day/local_computer/space_time/interaction_diagnostic/gc_truth_gc_vs_matern_one_day_092526/outputs/simulation/`

From this directory, verify the preserved files with:

```bash
shasum -a 256 -c GC_FFT_DATA_SHA256.txt
```
