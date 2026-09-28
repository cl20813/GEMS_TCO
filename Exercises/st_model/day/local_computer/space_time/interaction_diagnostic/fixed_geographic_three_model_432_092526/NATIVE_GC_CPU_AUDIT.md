# Native generalized-Cauchy CPU audit

Date: 2026-09-25  
Data day: 2024-07-02  
Geometry: corridor 4/3/2, 4x4 target blocks  
Target chunk size: 64  
Model: no-nugget generalized Cauchy, alpha=0.75, beta=1

The original Torch fit and the fused C++ fit used the same data, conditioning
graph, initial parameters, optimizer settings, and frozen diagnostic. The
native run was written under `outputs_native_cpu/`; no earlier result was
overwritten.

| quantity | Torch | native CPU | native - Torch |
|---|---:|---:|---:|
| profiled NLL / target | 1.3968643314285498 | 1.3968643314285512 | 1.4e-15 |
| max absolute gradient | 1.1807627345e-05 | 1.1807597356e-05 | -3.0e-11 |
| fit seconds | 213.2830 | 205.6856 | -7.5974 (-3.56%) |
| diagnostic score | 4.6739956264424602 | 4.6739956264396314 | -2.83e-12 |
| fitted C_AB | 0.6333153176386909 | 0.6333153176171041 | -2.16e-11 |
| fitted Var(L) | 2.2148549338914023 | 2.2148549338454386 | -4.60e-11 |

The six raw fitted parameters differ by at most approximately `7.6e-11`.
These differences are ordinary floating-point accumulation-order effects and
do not change the fitted model or diagnostic result.

The CPU speedup is intentionally reported as a one-run local audit, not a
general benchmark. The principal production benefit is expected on Amarel:
the fused CUDA backward recomputes GC derivatives instead of retaining every
pairwise Torch intermediate, which directly targets the observed GPU OOM.
Full lag-6/4/3 chunk-256 peak memory and runtime remain to be measured in the
two-date Amarel smoke job.

The command requested only `gc`, so the frozen three-model runner deliberately
returned a nonzero final status after completing and saving the GC fit; its
publication guard requires all three models before creating task `COMPLETE`.
