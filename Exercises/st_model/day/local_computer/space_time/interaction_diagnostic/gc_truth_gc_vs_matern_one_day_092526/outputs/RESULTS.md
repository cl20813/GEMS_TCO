# One-day GC-truth FFT audit: GC versus Matérn

A generalized-Cauchy field was generated for one eight-hour GEMS day. 
The FFT is applied to a zero-advection comoving field and observations 
are sampled at `s - v(t-t0)`. This is covariance-equivalent to constant 
advection while avoiding the non-centrosymmetric Nyquist artifact in the 
historical direct-advected embedding.

The primary population target is the effective, post-clipping and variance-
renormalized FFT covariance—the covariance that actually generated the field. 
The intended analytic GC covariance is reported as a sensitivity target.

| quantity | value |
|---|---:|
| analytic truth C_AB | 1.03254947 |
| effective FFT truth C_AB | 1.03884108 |
| empirical one-field C_AB | 1.37204311 |
| fitted GC C_AB | 1.03952434 |
| fitted Matérn C_AB | 0.0380406893 |
| fitted GC abs(C_AB - FFT truth) | 0.000683258404 |
| fitted Matérn abs(C_AB - FFT truth) | 1.00080039 |
| fitted GC abs(C_AB - analytic truth) | 0.00697486611 |
| fitted Matérn abs(C_AB - analytic truth) | 0.994508784 |
| analytic truth 2*d1*d2*C_AB | 0.0498191946 |
| effective FFT truth 2*d1*d2*C_AB | 0.0501227567 |
| fitted GC 2*d1*d2*C_AB | 0.050155723 |
| fitted Matérn 2*d1*d2*C_AB | 0.00183541472 |

- Smaller C_AB error to effective FFT truth: **GC**.
- Smaller C_AB error to intended analytic truth: **GC**.
- Effective-FFT population score regret, GC versus Matérn: `4.56483803e-06` versus `0.00134889204`.
- Analytic-GC population score regret, GC versus Matérn: `6.19833815e-06` versus `0.0013305219`.
- Observed one-field contrast score, GC versus Matérn: `3.95018725` versus `3.9525034`.
- Vecchia NLL per target, GC versus Matérn: `0.777259994` versus `0.785129083`.

## FFT integrity

- Simulation grid: `[1307, 1841, 8]`; embedding: `[2613, 3681, 15]`.
- Negative spectral mass fraction before correction: `0.00165882905`.
- Maximum imaginary/real spectrum ratio: `4.63530531e-17`.
- Variance before renormalization and scale: `10.0166435`, `0.998338415`.
- FFT minus analytic truth C_AB: `0.0062916077`.

## Interpretation boundary

This is a one-realization mechanism audit, not a power study or a general 
claim that GC always estimates cross covariance better. The empirical C_AB 
is noisy because overlapping within-day filters are not independent. Model 
recovery should therefore be read primarily against the two population-truth 
columns and their population score regrets.
