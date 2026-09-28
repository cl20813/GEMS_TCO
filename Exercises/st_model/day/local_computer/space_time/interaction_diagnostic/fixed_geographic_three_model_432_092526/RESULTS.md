# 2024-07-02 local 4/3/2 result

The frozen fixed-geographic lag-1 diagnostic completed on CPU for all three
models. The run used 4x4 target blocks, corridor conditioning counts 4/3/2,
and `target_chunk_size=64`. All fits met the outer `1e-4` gradient criterion.

There were 45,117 complete contrasts out of 55,440 possible contrasts.

| model | score | fitted VA | fitted VB | fitted CAB | fitted rhoAB | fitted Var(L) | Vecchia NLL | max abs gradient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generalized Cauchy | 4.673996 | 45.2616 | 45.2691 | 0.633315 | 0.013991 | 2.21486 | 1.396864 | 1.18e-5 |
| Matérn nu=0.5 | 4.697838 | 51.2747 | 51.2752 | 0.010684 | 0.000208 | 2.47481 | 1.404078 | 7.45e-6 |
| advected separable | 4.724828 | 56.5363 | 56.5368 | 0.035151 | 0.000622 | 2.72990 | 1.411969 | 2.37e-5 |

The empirical moments were `VA=37.9868`, `VB=40.0338`, `CAB=0.649127`, and
`Var(L)=1.91295`. Thus the GC cross moment is very close to the empirical
cross moment, whereas the Matérn and separable cross moments are both close
to zero. The score ordering is

```text
GC < Matérn < separable
```

with `Score(Matérn)-Score(GC)=0.0238428` and
`Score(separable)-Score(Matérn)=0.0269892`.

This local result shows that the small Matérn `CAB` in the Amarel partial run
is reproducible under the smaller 4/3/2 Vecchia graph; it is not explained by
the 6/4/3 GPU OOM. The fitted Matérn has much shorter ranges than the fitted GC,
especially spatially, so the selected cross-geometry covariance is nearly
zero under that fitted specification. In contrast, the longer-tailed GC
retains the relevant cross covariance.

The older 2024-07-01 result is not directly comparable because it used
translated, advection-following endpoints and a matched-separable covariance
derived from the same GC fit. This audit uses fixed geographic endpoints and
fits all three models independently.

Machine-readable results are in
`outputs/2024-07-02/task_000_2024-07-02/daily_scores.csv`.

