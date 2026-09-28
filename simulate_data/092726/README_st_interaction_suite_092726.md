# Controlled space-time interaction simulation suite (092726)

## Decision

Phase 1 uses the real July 1--30, 2024 GEMS source locations: 30 independent
eight-hour day blocks, or 240 hourly templates per scenario.  This is preferable
to immediately mixing 2024 and 2025 because it gives 30 independent replicates
for day-level uncertainty while holding the sampling year and calendar month
fixed.  Use 2025 later as an external observation-geometry robustness check.

All production scenarios have nugget exactly zero.  They share signal variance,
ranges, advection, deterministic mean, observation locations, selected dates,
and day seeds.  The only within-family manipulation is the space-time coupling.

## Why the old generator is not reused unchanged

The historical multi-day generator puts advection directly into the first
column of an even BCCB embedding, discards the imaginary part of the FFT, and
clips negative eigenvalues.  The advected column is not centrally symmetric on
the Nyquist faces, so this can alter precisely the small cross-covariance being
studied.  It also replaced `Hours_elapsed` with consecutive integers, making
independently generated adjacent days appear one hour apart.

The new generator instead uses an odd `2n-1` embedding for a zero-advection
field and then samples

```text
Z(s,t) = W(s - v * local_time, local_time).
```

It checks central symmetry and imaginary leakage, measures negative spectral
mass, refuses excessive clipping, and renormalizes the requested variance.
The original time is retained as `Template_Hours_elapsed`; analysis
`Hours_elapsed` is put on the exact integer FFT lattice.  The generator also
adds `Simulation_Block` and `Simulation_Time_Index`.  Downstream fitting and
diagnostics must treat days as independent blocks and never construct
cross-day pairs.

## Controlled interaction parameter

Let

```text
d_s^2 = ((h_lat - v_lat*u)/range_lat)^2
      + ((h_lon - v_lon*u)/range_lon)^2
d_t   = abs(u)/range_time.
```

For radial correlation `R`, define

```text
C_sep   = sigmasq * R(d_s) * R(d_t)
C_joint = sigmasq * R(sqrt(d_s^2 + d_t^2))
C_eta   = (1-eta)*C_sep + eta*C_joint.
```

Both endpoints are valid covariance functions, and their convex mixture is
valid.  Within the analytic covariance family, every value of eta has exactly the same
pure spatial margin `C(h,0)` and the same flow-following temporal margin
`C(v*u,u)`.  Thus eta changes the targeted interaction rather than variance or
range.  With nonzero advection, fixed-geographic temporal covariance is not a
controlled margin; the controlled temporal margin is explicitly Lagrangian.
Consequently eta=0 is a **Lagrangian separability null**, not a generic
Eulerian no-interaction null.  At a fixed site and one-hour lag, for example,
the configured Matérn correlations are about `0.279` (eta=0) and `0.397`
(eta=1).  Fixed-coordinate diagnostics must therefore be interpreted as
detecting interaction beyond/alongside advection; flow-aligned contrasts are
the clean primary estimand.
For the sampled circulant field, any departure caused by spectral clipping and
variance renormalization is bounded and saved as
`relative_covariance_distortion_bound`; a run is refused above `1e-4`.

The six production scenarios form a `2 families x 3 eta levels` design:

| Scenario | Family | eta | Role |
|---|---|---:|---|
| `matern05_lagrangian_separable_eta0` | Matérn nu=0.5 | 0 | Lagrangian separability null |
| `matern05_lagrangian_mixture_eta0p5` | Matérn nu=0.5 | 0.5 | intermediate interaction |
| `matern05_lagrangian_joint_eta1` | Matérn nu=0.5 | 1 | nonseparable Lagrangian interaction |
| `gencauchy_a1_b5_lagrangian_separable_eta0` | normalized GC a=1,b=5 | 0 | matched-family Lagrangian separability null |
| `gencauchy_a1_b5_lagrangian_mixture_eta0p5` | normalized GC a=1,b=5 | 0.5 | matched-family intermediate interaction |
| `gencauchy_a1_b5_lagrangian_joint_eta1` | normalized GC a=1,b=5 | 1 | matched-family nonseparable interaction |

The generalized-Cauchy kernel is normalized as

```text
R_GC(r) = {1 + [exp(1/5)-1] r}^-5,
```

so both families have `R(1)=exp(-1)` and local roughness exponent one.  At the
controlled corner `(d_s,d_t)=(1,1)`, the eta `0/0.5/1` correlations are
`0.135335/0.189226/0.243117` for Matérn and
`0.135335/0.195742/0.256149` for GC.  This is a close robustness pairing, not
a claim that the two families have identical full marginal curves.

The saved `range_lat/range_lon/range_time` values are effective e-fold ranges.
An existing unnormalized GC fitter using `(1+r^a)^(-b/a)` must instead use the
saved kernel ranges `(0.903331, 1.354997, 9.033311)`.  Both forms are written
to every truth JSON.  The `fitter_truth_parameters` object is the required
adapter for downstream fitting and is numerically validated against
`R(1)=exp(-1)`; do not feed the generic GC `range_*` fields directly to the
existing unnormalized fitter.

Do not create the interaction experiment by changing Matérn smoothness or GC
`a,b`: those changes also alter marginal roughness and correlation decay.

## Shared truth

```text
sigmasq    = 10
range_lat  = 0.2
range_lon  = 0.3
range_time = 2.0
advec_lat  = 0.08
advec_lon  = -0.2
nugget     = 0
mean       = 260 + 1 * (Source_Latitude + 0.5)
```

High-resolution spacing is the previously tested configuration:

```text
dlat = 0.044 / 100
dlon = 0.063 / 10
```

The factors refine grid spacing; they do not multiply latitude or longitude
coordinates.  Production preflight resolves an embedding of approximately
`26145 x 3681 x 15`, with a conservative working-set estimate near 61 GiB.
The final threaded code used 4.5--4.6 GiB peak RSS in a Mac-only,
reduced-resolution x10/x10 code smoke test; this is not a data-generation
setting.  The Slurm job requests 160 GiB to leave FFT and
allocator headroom.  The remote one-day gate uses the full production x100/x10
resolution; it reduces days only, not the numerical experiment.  Its universal
axes are still constructed from all 30 dates, so it uses the exact production
embedding shape.  SciPy's FFT is assigned all eight requested CPU workers.

Real source coordinates are evaluated at their nearest high-resolution FFT
lattice point.  At production spacing the deterministic worst-case bounds are
`0.00022` degree latitude and `0.00315` degree longitude, only `0.11%` and
`1.05%` of their respective e-fold ranges; all 240 production maps have zero
FFT-cell collisions.  These tolerances and every realized maximum mapping
error are saved, so “exact margins” refers to the analytic/lattice covariance,
not an unrecorded claim of exactness at off-lattice source coordinates.

The Mac-only reduced-resolution smoke test generated and deeply validated all
six scenario pickles.  Imaginary-spectrum ratios were about `1e-16`; the
three Matérn scenarios and GC eta=0/0.5 needed no clipping, while joint GC had negative spectral mass
`3.88e-7`, far below the configured `5e-4` refusal threshold.  Its direct
relative covariance-distortion bound was `7.77e-7`, also far below `1e-4`.
Both real-location and gridded outputs passed the strict time, coordinate,
manifest, seed, and completion checks.

The validator enforces the production contract by default: x100/x10 spacing,
axes constructed from all July 1--30 dates, exact selected dates, and matching
generator/config/input hashes.  Reduced local smoke tests are accepted only
with the explicit `--allow-nonproduction-grid` flag, which the Amarel helper
never uses.

Downstream code must use `Hours_elapsed` together with `Simulation_Block`.
It must not reconstruct analysis time from the hour-key: the original
04:48--07:48 keys are intentionally shifted by five minutes onto the integer
FFT time lattice, while their original values remain in
`Template_Hours_elapsed`.

## Files

```text
st_interaction_scenarios_092726.json
generate_st_interaction_suite_092726.py
validate_st_interaction_suite_092726.py
plot_st_interaction_design_092726.py
slurm_generate_st_interaction_suite_092726.sbatch
submit_st_interaction_suite_092726.sh
amarel_st_interaction_suite_092726.sh
```

Each scenario writes monthly files with the standard downstream schema under

```text
<suite>/<scenario>/2024_july_st_circulant/
```

plus atomic daily checkpoints and `COMPLETE.json`.  A failed or timed-out job
reuses matching completed days on resubmission.

Daily checkpoints remain on Amarel when consolidated results are downloaded,
avoiding a duplicate local copy while preserving remote restart capability.

The monthly contract contains `real_locations.pkl`, `gridded.pkl`,
`manifest.csv`, `griddification_diag.csv`, `embedding_diag.csv` (plus a JSON
copy), and `truth.json`, all with the standard `sim_july2024_st_circulant_`
prefix.

## Analysis contract and limits

- The 30 daily fields are independent blocks, but they are not identically
  distributed because real observation geometry and missingness vary by day.
  A monthly Vecchia fit is valid only if neighbor construction and likelihood
  factorization honor `Simulation_Block`.  Otherwise fit/analyze each day
  separately; never permit conditioning across two dates.
- Common random numbers deliberately pair all six scenarios within day.
  Estimate effects with within-day paired differences and do not count the
  six scenarios as independent replicates.  The suite therefore contains six
  monthly bundles and 180 paired day-scenario blocks, not 180 independent
  replicates.
- The one-day Amarel pilot is engineering QA only.  It has the same seed and
  field as day 1 of the full run.  If its diagnostic outcome is inspected to
  alter a contrast or threshold, exclude July 1 from final inference.
- Thirty days are appropriate for an exploratory phase-1 paired effect study,
  not for definitive type-I-error or power calibration.  Reserve 2025 and/or
  additional Monte Carlo replicates for that later stage.
- A correctly specified fitter currently exists for the joint Matérn,
  separable exponential/Matérn, and joint GC endpoints, but not separable GC
  or the eta=0.5 mixtures.  Generation and oracle-covariance diagnostics cover
  all six scenarios; claiming a correctly specified fitted-model benchmark
  for every scenario requires implementing those fitters first.

## Recommended run sequence

The helper defaults to the current Amarel login endpoint
`jl2815@amarel-new.hpc.rutgers.edu`.  Override it with `AMAREL_HOST` if needed.

From the local repository:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1

# 1. Upload and submit six one-day production-resolution x100/x10 pilots.
./simulate_data/092726/amarel_st_interaction_suite_092726.sh upload-submit-pilot

# 2. Monitor, download, and validate the pilots.
./simulate_data/092726/amarel_st_interaction_suite_092726.sh status
./simulate_data/092726/amarel_st_interaction_suite_092726.sh download-pilot
./simulate_data/092726/amarel_st_interaction_suite_092726.sh validate-pilot

# 3. Only after pilot PASS, submit the six 30-day x100/x10 jobs.
./simulate_data/092726/amarel_st_interaction_suite_092726.sh submit-full

# 4. Download into the requested local simulate_data root and validate.
./simulate_data/092726/amarel_st_interaction_suite_092726.sh download
./simulate_data/092726/amarel_st_interaction_suite_092726.sh validate
```

The remote production root is

```text
/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726
```

The one-day pilot is isolated beside it at

```text
/home/jl2815/tco/exercise_output/fall_26/st_interaction_july2024_nugget0_092726_pilot
```

Slurm stdout/stderr logs are written under

```text
/home/jl2815/tco/exercise_output/fall_26/logs
```

and the downloaded local root is

```text
/Users/joonwonlee/Documents/GEMS_TCO-1/simulate_data/092726/st_interaction_july2024_nugget0_092726
```

## Follow-up design

The six scenarios are sufficient for the first interaction-diagnostic study.
If a clean advection false-positive study is needed, add a second velocity
stratum with `v=(0,0)` and keep the same `2 families x 3 eta levels` design,
producing twelve scenarios.  That is a phase-2 factorial extension rather than
a reason to double the first production run now.
