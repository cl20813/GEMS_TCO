# Matrix-free Lanczos generator sensitivity

The finite regular-lattice target covariance was not replaced by a clipped circulant covariance. Zero-padded FFT was used only for exact BTTB matrix-vector products, and Lanczos approximated the square-root action on a fixed Gaussian vector.

| DGP | m | replicates | max relative field error vs largest m | mean L squared | analytic target | clipped 8x expectation |
|---|---:|---:|---:|---:|---:|---:|
| `joint` | 120 | 5 | 0.000545463 | 1.40277589 | 1.38211338 | 1.41415197 |
| `joint` | 160 | 5 | 0 | 1.40277203 | 1.38211338 | 1.41415197 |

These few fields assess numerical generator convergence only; they are not a bootstrap calibration and their empirical contrast moments have Monte Carlo variation. The analytic target moments remain the reference for generator bias.
