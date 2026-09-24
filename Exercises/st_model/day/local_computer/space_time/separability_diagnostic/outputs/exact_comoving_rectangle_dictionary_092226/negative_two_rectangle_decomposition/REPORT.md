# First negative two-rectangle contrast

This is a population-covariance decomposition of the exact-comoving oracle. It reads no responses and is not a calibrated test.

## The two actual rectangles

- `Q1 = s000_024_t00_07` compares anchors `0` and `24` between hours `0` and `7`.
- `Q2 = s001_023_t01_06` compares anchors `1` and `23` between hours `1` and `6`.

With the orientation `+p,k -q,k -p,l +q,l`, the final unit-null-variance projection is

`L = (+0.127556335109) Q1 + (-0.126735788204) Q2`.

Equivalently,

`Q1 = Y[24,7] - Y[0,7] - Y[24,0] + Y[0,0]`,

`Q2 = Y[23,6] - Y[1,6] - Y[23,1] + Y[1,1]`.

Writing `A_t=Y[0,t]-Y[24,t]`, `B_t=Y[1,t]-Y[23,t]`, `a=0.127556335109`, and `b=0.126735788204`, the same filter is exactly `L=a(A_0-A_7)-b(B_1-B_6)`.

## Two-by-two generalized problem

In the original raw rectangle coordinates,

```text
D_raw  = [[ 0.047434218,  3.195709986],
          [ 3.195709986,  0.200705679]]
G0_raw = [[ 35.444194427,  3.980977887],
          [ 3.980977887,  34.367762698]]
```

After scaling each rectangle to unit fitted-null variance,

```text
D  = [[ 0.001338279,  0.091562823],
      [ 0.091562823,  0.005839940]]
G0 = [[ 1.000000000,  0.114062157],
      [ 0.114062157,  1.000000000]]
```

Both single-rectangle intrinsic differences are positive (`d11=0.0013382789`, `d22=0.00583994021`), but `d12^2=0.00838375056` exceeds `d11*d22=7.81546877e-06`. Thus `det(D)=-0.00837593509<0`, and D has one negative direction.
The generalized eigenvalues are `-0.0993278821` and `0.0854376811`.

## Why the combined direction is negative

| contribution | intrinsic value | fitted-null normalization value |
|---|---:|---:|
| alpha1^2 M11 | 0.000771784064 | 0.57669897 |
| alpha2^2 M22 | 0.0032237266 | 0.55201363 |
| 2 alpha1 alpha2 M12 | -0.103323393 | -0.1287126 |
| total | -0.0993278821 | 1 |

The two positive diagonal terms sum to `0.00399551066`, while the cross term is `-0.103323393`: its magnitude is `25.860` times larger.
The raw cross-covariance is `11.2382385` under truth and `8.0425285` under the matched-margin model. Because the rectangles enter with opposite signs, the larger positive truth cross-covariance creates stronger cancellation.
This concerns a covariance difference; no variance itself is negative, and it does not assert that either model's correlation is negative.

## Fitted-null compensation

| quantity | value |
|---|---:|
| v0 | 1 |
| vM | 0.951940299 |
| v1 | 0.852612417 |
| intrinsic v1-vM | -0.0993278821 |
| compensation vM-v0 | -0.0480597006 |
| total v1-v0 | -0.147387583 |
| rho0=v1/v0 | 0.852612417 |
| rhoM=v1/vM | 0.895657446 |
| g(rho0) | 0.00603131401 |

Here the fitted-null refitting effect has the same negative sign as the intrinsic difference, so it reinforces rather than offsets it. Truth variance is about 14.74% below fitted-null variance for this fixed filter.

## What temporal pattern does the final filter represent?

The observation-weight matrix has rank `2` with nonzero singular values `0.25511267` and `0.253471576`. Its best rank-one approximation retains only `50.32%` of Frobenius energy.
The two rectangles use different spatial contrasts, so the filter cannot be written exactly as one spatial vector times a single temporal pattern such as `W0-W1+W6-W7`.
It is exactly the sum of two separable four-point contrasts: an outer diagonal contrast over hours 0 and 7 and a different inner diagonal contrast over hours 1 and 6. Rectangle widths 7 and 5 hours therefore describe the atoms, not a unique temporal scale of the combined filter.

Quadratic-form attribution for the final filter:

| absolute time lag | intrinsic Sigma1-SigmaM | total Sigma1-Sigma0 |
|---:|---:|---:|
| 0 | 0 | 0.125347609 |
| 1 | -0.131017302 | -0.280192653 |
| 5 | 0.0032237266 | -0.0257711932 |
| 6 | 0.0276939088 | 0.0455515901 |
| 7 | 0.000771784064 | -0.0123229358 |

These lag sums are quadratic-form attributions of the final filter, not rectangle coefficients and not independent pieces of information.
The atom endpoint spans are 5 and 7 hours, but the strongest negative intrinsic contribution occurs at lag 1; endpoint width, weight support, and quadratic-form lag attribution are distinct summaries.

## Selection and interpretation boundary

- The first minimizing rectangle had `2` candidates within the predeclared tie tolerance. `Q1` is the lexicographically selected representative.
- `Q2` is the best second addition conditional on that Q1. This is a greedy path, not an exhaustive globally optimal search over every rectangle pair.
- The overall sign of L is arbitrary; the relative minus sign between Q1 and Q2 is the identified feature.
- This is an exact-grid oracle population result, not evidence of significance or performance on the warped GEMS observation geometry.

## Numerical checks

- Null normalization error: `2.220e-16`.
- Reconstruction error from the two raw rectangles: `0.000e+00`.
- Basis-invariant constrained residual: `1.406e-16`.
- Direct reduced 2-by-2 residual: `1.928e-16`.
- Full observation-space relative residual: `6.064e-01`; it need not vanish because this is a two-rectangle constrained eigenproblem.
- The saved oracle k=2 objective, recomputed generalized eigenvalue, intrinsic quadratic sum, and pair-attribution sum are required to agree before outputs are written.
