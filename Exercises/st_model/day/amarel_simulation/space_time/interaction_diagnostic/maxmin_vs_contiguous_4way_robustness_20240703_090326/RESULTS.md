# Global max-min versus contiguous 20x20 robustness

Both experiments use 400 locations x 8 times, the same stored adapted 4/3/2
parameters, and the same four-way E1--E2--V1--V2 construction.

| metric | global max-min | contiguous 20x20 |
|---|---:|---:|
| mean B nonzeros/row | 9.10 | 102.19 |
| E1--V1 real curve RMSE/n | 0.0116505 | 0.00156352 |
| real 20-band RMSE | 0.128955 | 0.0429808 |
| covariance relative Frobenius error | 0.164002 | 0.08971 |
| KL exact-to-Vecchia / observation | 0.00494304 | 0.00138867 |
| max expected 20-band bias | 0.113284 | 0.0313462 |
| simulated curve RMSE median | 0.0105188 | 0.00311178 |
| simulated band RMSE median | 0.11972 | 0.0668547 |

The observed E1--V1 curve discrepancy is 7.45 times larger under
global max-min thinning, and the maximum expected band bias is
3.61 times larger.  Conditioning density is therefore a major
driver of the apparent Vecchia spectral error.  The contiguous result is much
closer to the full-data operator, whose B has about 131 nonzeros per row, but it
still does not directly prove the full-data Vecchia approximation error.

Hard-band Lanczos errors remain around 0.02 at m=512 and 32 probes, so smooth
spectral filters and probe convergence are still required before statistical
calibration.
