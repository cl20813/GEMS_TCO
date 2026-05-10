# July ST Circulant Simulation Assets

This folder contains reusable simulation generators and runbooks for GEMS July
space-time experiments.  The goal is to generate common simulated data assets
once, then reuse the same pickles for pure-space, space-time, gridded, and
real-location model fitting tests.

## Griddification Rule

For the gridded output, each real source location is mapped to its nearest
regular GEMS grid cell.  The assignment is accepted only if both coordinate
differences are within half of a native grid cell:

```text
abs(source_lat - grid_lat) <= 0.044 / 2
abs(source_lon - grid_lon) <= 0.063 / 2
```

This is not a radial distance threshold.  It is an axis-wise half-cell
threshold in latitude and longitude separately.  If multiple source locations
map to the same grid cell, only the closest source is kept.

## Circulant Embedding Block Design

The high-resolution simulation grid uses:

```text
latitude  x100: dlat = 0.044 / 100
longitude x10:  dlon = 0.063 / 10
```

Running one 3D FFT for all 248 hours at this high resolution would require too
much memory.  The generator therefore creates independent daily 8-hour
space-time circulant-embedding blocks.  With `range_time = 2.0`, overnight
cross-day correlation is negligible for the current daily GEMS pure-space and
space-time testing goals.

## July Template Hour Counts

The local July real-location template pickles currently contain:

```text
2022: 240 hours
2023: 248 hours
2024: 248 hours
2025: 247 hours
```

The generator uses the available ordered July keys up to 248 hours for each
year; it does not invent missing hour keys.
