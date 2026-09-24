# Fixed-short-scale long-contrast path

This is an analytic population-covariance mechanism audit. It performs no simulation, covariance refit, dictionary search, or coefficient search.
Input oracle: `/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/space_time/separability_diagnostic/outputs/exact_comoving_rectangle_dictionary_092226`.

## Path

The path fixes

`phi=0`, `r_a=s+r_star/2`, `r_b=s-r_star/2`,

so `d_minus=r_star`, `d_plus=2s`, `2r_a=2s+r_star`, and `2r_b=2s-r_star`.

For the saved one-step temporal scale, `r_star=0.544955056123904` maximizes `|k_delta(r)|`, with `k_delta(r_star)=-2.51215092911315` and `C_short=-5.0243018582263`.

The denominator uses the fitted null after restoring the exact spatial range ratio while preserving the fitted spatial-range product. In truth-standardized moving coordinates its spatial decay is `1.51177219314045` and its one-step temporal decay is `0.642610743783346`.

## What the path shows

- The exact numerator threshold is `s=0.808681437725839`. Below it, `C^2<AB`; above it, `C^2>AB` and one negative generalized eigenvalue exists.
- `A` and the far-lag cancellation decrease along the full admissible path. `B` first rises because the shorter contrast grows from zero length, reaches its maximum at `s=r_star`, and then decreases. Thus all three unwanted terms decrease together on the long-contrast branch `s>r_star`.
- On the evaluated path, `lambda_min` is monotonically decreasing and approaches `-0.205470984724064` as `s` tends to infinity.

At the mean half-length of the selected discrete pair,

- `s_selected=(sqrt(8)+sqrt(5))/2=2.53224755112299`;
- path geometry: `(r_a,r_b,phi,d_minus,d_plus)=(2.80472507918, 2.25977002306, 0, 0.544955056124, 5.06449510225)`;
- numerator components: `(A,B,C_short,C_far,C)=(0.0544297612531, 0.159607371147, -5.02430185823, 0.0932778386444, -4.93102401958)`;
- path `lambda_min=-0.197448361299253`, which is `96.095%` of the limiting magnitude;
- original discrete geometry has `(phi,d_minus,d_plus)=(18.43494882 degrees, 1, 5)` and symmetric-null `lambda_min=-0.190275207393445`.

At the same mean half-length, tuning the collinear separation to `r_star` makes the objective magnitude `3.770%` larger than for the selected lattice geometry. This is a comparison on the specified path, not a claim of a global continuous optimum.

The analytic discrete value agrees with the saved axis-symmetrized matrix audit within `3.331e-16`.

## Limit

As `s` tends to infinity, `A`, `B`, and `C_far` vanish, while `C` approaches the fixed short-lag contribution. The limiting values are

- `C=-5.0243018582263`;
- `(P,Q,R)=(16.9958484922223, 16.9958484922223, 7.45676151971517)`;
- `lambda_min=-0.205470984724064`.

Therefore the path confirms the proposed mechanism: once `d_minus` is held at the optimal nonzero scale, longer near-parallel contrasts progressively remove diagonal penalties and far-lag cancellation. The improvement has a finite asymptote rather than growing without bound.

## Reproduction

From the diagnostic directory:

```bash
python analyze_fixed_short_scale_length_path.py
```
