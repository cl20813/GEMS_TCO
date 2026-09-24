# FFT-accelerated matrix-free Lanczos full-pipeline pilot

The generator targets the finite joint-GC BTTB covariance and uses FFT only for exact zero-padded matrix-vector products. No circulant eigenvalue is clipped. Each field is restored to the saved GLS mean and original missingness mask, then re-fitted with the unchanged 4/3/2 corridor-Vecchia pipeline.

Replicates: `1`. Observed empirical/joint statistic: `0.93779584`. Pilot bootstrap mean: `1.02293542`.

This remains a computational pilot. Do not interpret its tail probability until at least 99 replicates have been run and the separable-null refit path has been implemented separately.

See `bootstrap_replicates.csv` for Lanczos convergence, optimizer diagnostics, fitted parameters, and timing.
