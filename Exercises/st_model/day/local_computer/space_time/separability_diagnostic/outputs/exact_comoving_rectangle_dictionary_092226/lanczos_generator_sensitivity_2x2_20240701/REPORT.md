# Matrix-free Lanczos generator sensitivity

The finite regular-lattice target covariance was not replaced by a clipped circulant covariance. Zero-padded FFT was used only for exact BTTB matrix-vector products, and Lanczos approximated the square-root action on a fixed Gaussian vector.

| DGP | m | replicates | max relative field error vs largest m | mean L squared | analytic target | clipped 8x expectation |
|---|---:|---:|---:|---:|---:|---:|
| `joint` | 200 | 1 | 0.000794419 | 1.30905271 | 1.3829133 | nan |
| `joint` | 240 | 1 | 0.000197224 | 1.30904372 | 1.3829133 | nan |
| `joint` | 280 | 1 | 0 | 1.3090468 | 1.3829133 | nan |

These few fields assess numerical generator convergence only; they are not a bootstrap calibration and their empirical contrast moments have Monte Carlo variation. The analytic target moments remain the reference for generator bias.
