# M3 / M3+Q / gated-Q confirmatory experiment: interim report

Status: stopped by user because the full 72-dataset run was too slow.

## Usable sample

- 30 completely paired synthetic datasets
- 3 nuisance starts per dataset
- 3 initializer methods per start
- 270 successful full seven-parameter fits
- One incomplete dataset was excluded.
- Coverage is an unbalanced prefix of the planned factorial: all selected
  conditions at 22.5 degrees, and the 0.75/2.0-cell speeds at 112.5 degrees.
  The opposite two quadrants were not reached. These are descriptive interim
  results, not the final confirmatory result.

## Initializer accuracy

| Method | Mean seed error | Median seed error | Mean initializer time |
|---|---:|---:|---:|
| M3 FFT | 0.02714 | 0.02257 | 0.00586 s |
| M3 FFT + gated Q | 0.01970 | 0.00956 | 0.00710 s |
| M3 FFT + Q | **0.01650** | **0.00897** | 0.00597 s |

## Downstream fit summary

| Method | Numerical success | Gradient convergence | Mean common regret | Mean standardized parameter error | Mean final advection error | Mean fit time |
|---|---:|---:|---:|---:|---:|---:|
| M3 FFT | 100% | 44.4% | 0.000739 | 0.1790 | 0.02184 | 7.135 s |
| M3 FFT + gated Q | 100% | 46.7% | 0.000769 | 0.1830 | 0.02169 | 7.211 s |
| M3 FFT + Q | 100% | 46.7% | **0.000500** | **0.1697** | **0.01366** | 7.152 s |

The low gradient-convergence rates are not numerical failures. Approximately
87--90% of fits reached the 20-evaluation budget, so this interim experiment
mainly compares equal-budget optimization outcomes rather than fully converged
solutions.

## Paired cluster-bootstrap results: Q minus M3

| Metric | Mean paired difference | 95% dataset-cluster bootstrap CI |
|---|---:|---:|
| Common objective NLL | -0.000240 | [-0.000864, 0.000143] |
| Standardized seven-parameter error | -0.00929 | [-0.03966, 0.01041] |
| Final advection error | -0.00818 | [-0.02502, 0.000815] |
| Optimizer iterations | +0.122 | [-0.178, 0.433] |
| Full-fit time | +0.017 s | [-0.065, 0.108] |
| End-to-end time | +0.021 s | [-0.058, 0.110] |

All intervals include zero. The point estimates favor Q for common NLL and
parameter/advection accuracy, while optimizer work and time are effectively
the same at the current sample size.

## Nuisance-start robustness

The mean trace of standardized final-parameter variance across the three
nuisance starts was:

- M3 FFT: 0.3208
- gated Q: 0.3319
- Q: **0.1634**

The Q estimate is much smaller, but this mean is affected by one important
fast/strong case in which Q avoided a poor solution reached from the
long-time/low-nugget start. Removing or replicating that case is necessary
before treating the reduction as stable evidence.

## What the gate did

Gated Q differed from unconditional Q on four datasets (12 nuisance-start
comparisons). Q had lower common NLL in six and M3 had lower common NLL in six.
The gate therefore did not classify the better downstream seed reliably. In
the most consequential case it rejected Q even though Q avoided a large poor
solution, so gated Q lost most of Q's observed tail-risk benefit.

## Interim conclusion

1. M3+Q remains the most promising method.
2. The current JQ90 gate is not supported as a hard rule; keep it as an
   uncertainty diagnostic.
3. No method had a numerical optimizer failure.
4. The 20-evaluation budget was too short for a clean convergence comparison.
5. Because direction coverage is incomplete and the confidence intervals
   include zero, this partial run cannot establish a final publication claim.

