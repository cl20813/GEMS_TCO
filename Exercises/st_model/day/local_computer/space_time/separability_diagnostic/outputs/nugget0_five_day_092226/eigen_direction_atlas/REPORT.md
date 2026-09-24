# Eigen-direction interaction atlas

No response value was used to reconstruct, rank, or annotate these directions.

## Global structure

- Total reference-covariance KL: `1.5585045`.
- Same-margin intrinsic-interaction KL before null refitting: `4.8102497`.
- Fraction of that KL remaining after the separable null is refitted: `32.400%`.
- Null-whitened Frobenius cosine between intrinsic interaction and fitted-null compensation: `-0.8560`.
- Modes needed for 50%, 80%, and 90% cumulative KL: `101`, `251`, `354`.
- KL from lambda > 1 directions: `0.71654707`.
- KL from lambda < 1 directions: `0.84195741`.

## Two leading pattern families

Among the 20 strongest modes in each variance branch:

| branch | median rank-1 | median effective rank | median temporal roughness | median high-frequency weight energy |
|---|---:|---:|---:|---:|
| lambda > 1 | 0.472 | 3.681 | 0.720 | 0.189 |
| lambda < 1 | 0.726 | 2.259 | 3.118 | 0.970 |

## Covariance-only exploration candidates

These are transparent exemplars, not a finalized selected test.

| basis | mode | role | total g | intrinsic g | interaction share | rank-1 | high-frequency weight energy |
|---|---:|---|---:|---:|---:|---:|---:|
| fitted_null | 1 | top_total_gt | 0.017179 | 0.0036787 | 0.505 | 0.554 | 0.106 |
| fitted_null | 5 | top_total_lt | 0.013853 | 0.055276 | 0.632 | 0.981 | 0.992 |
| fitted_null | 14 | interaction_dominant_gt | 0.010961 | 0.0060735 | 0.769 | 0.424 | 0.168 |
| fitted_null | 39 | interaction_dominant_lt | 0.0077254 | 0.010798 | 0.852 | 0.592 | 0.784 |
| margin_matched_interaction | 1 | top_intrinsic_lt | 0.01202 | 0.064214 | 0.604 | 0.992 | 0.988 |
| margin_matched_interaction | 48 | top_intrinsic_gt | 0.0096847 | 0.025509 | 0.748 | 0.986 | 0.026 |

## Space-time cluster heatmaps

Raw eigenvector signs are arbitrary and close eigenvalues permit within-cluster rotation.  The pointwise amplitude, temporal/spatial Gram, joint graph-Fourier/DCT spectrum, and cluster lag-attribution panels are therefore the stable interpretation targets.

Heatmap columns use one-dimensional Fiedler graph seriation only.  Adjacent columns are not a spatial rectangle and must not define a spatial contrast.

- `gt_008`: modes `13;14` (interaction_dominant_gt), eigenvalue range `[1.2242, 1.2262]`; high-frequency Euclidean filter-weight energy `0.169`, mean adjacent-hour similarity `0.630`, peak energy hour `7`, dominant joint frequency `(temporal=0, graph-rank=0)`.
- `lt_014`: modes `39;40` (interaction_dominant_lt), eigenvalue range `[0.8344, 0.8348]`; high-frequency Euclidean filter-weight energy `0.841`, mean adjacent-hour similarity `-0.277`, peak energy hour `2`, dominant joint frequency `(temporal=6, graph-rank=10)`.

## Dominant intrinsic-interaction lag cells

- `fitted_null` mode `1` (top_total_gt): time lag `2` hours, moving spatial norm `[2, 3)`.
- `fitted_null` mode `5` (top_total_lt): time lag `1` hours, moving spatial norm `[2, 3)`.
- `fitted_null` mode `14` (interaction_dominant_gt): time lag `2` hours, moving spatial norm `[2, 3)`.
- `fitted_null` mode `39` (interaction_dominant_lt): time lag `1` hours, moving spatial norm `[2, 3)`.
- `margin_matched_interaction` mode `1` (top_intrinsic_lt): time lag `1` hours, moving spatial norm `[2, 3)`.
- `margin_matched_interaction` mode `48` (top_intrinsic_gt): time lag `2` hours, moving spatial norm `[2, 3)`.

## Interpretation limits

- A non-rank-one weight matrix describes the filter; it does not by itself prove covariance nonseparability.
- Near-degenerate eigenvectors may rotate.  Interpret their labelled cluster or subspace before interpreting one mode.
- The Fiedler ordering is only a display device; the 2D anchor-order panel records the original spatial geometry.
- The graph-Fourier/DCT panel decomposes Euclidean filter-weight energy, not covariance variance or KL.  It is conditional on the standardized coordinates and k-nearest-neighbor graph; individual graph-frequency cells can rotate inside a repeated graph-Laplacian eigenspace.
- The intrinsic/compensation split uses the known simulation truth.  A real-data analogue must remove the fitted-null nuisance tangent space.
- Candidate construction and all tuning must remain on the covariance-only design split.  Held-out responses belong only in the final test.
