# July 2025 local GC versus Matérn audit

Frozen fixed-geographic A/B contrasts, temporal lag 1, corridor 4/3/2, 
4x4 target blocks, CPU target chunk 64, and nugget fixed at zero were used.
The third date (July 23) was selected reproducibly from the complete late-July 
dates with seed 20250925; July 7 and July 15 were specified in advance.

Lower contrast score is better. `Matérn-GC > 0` therefore favors GC.
The covariance Frobenius error compares the fitted and empirical 2x2 second-
moment matrices and is descriptive rather than an independent test.

| date | empirical CAB | GC CAB | Matérn CAB | GC score | Matérn score | Matérn-GC | GC cov. error | Matérn cov. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-07 | 0.474567 | 0.550652 | 0.0138871 | 4.498 | 4.51443 | 0.0164351 | 4.14844 | 10.6063 |
| 2025-07-15 | 0.730606 | 1.03318 | 0.0503232 | 4.65213 | 4.6541 | 0.00196765 | 0.795094 | 3.64256 |
| 2025-07-23 | 0.160281 | 0.00546524 | 3.23802e-07 | 3.77219 | 3.77218 | -4.81376e-06 | 0.733267 | 1.08239 |

- GC has the lower diagnostic score on 2/3 days.
- GC has the smaller raw 2x2 covariance error on 3/3 days.
- GC has the smaller absolute C_AB error on 3/3 days.
- GC has the smaller absolute Var(L) error on 3/3 days.
- GC has the lower fitted Vecchia NLL on 3/3 days.
- Mean `Score(Matérn)-Score(GC)`: `0.0061326353`.
- Mean absolute C_AB error, GC versus Matérn: `0.17782595` versus `0.43374781`.
- Mean absolute Var(L) error, GC versus Matérn: `0.047412915` versus `0.14523992`.
- Mean raw 2x2 covariance error, GC versus Matérn: `1.8922662` versus `5.1104084`.
- Mean fitted Vecchia NLL, GC versus Matérn: `1.2409203` versus `1.2487021`.
- July 23 is an effective score tie: Matérn is lower by only `4.81e-06`.

Within these three frozen-design days, GC reproduces the targeted covariance more closely and has no meaningful score loss. This supports GC for this specific interaction diagnostic, but three in-sample days are not a general or calibrated model-selection result.

- Results are a three-day descriptive audit, not a calibrated hypothesis test.
