# Six-scenario two-contrast cross-center diagnostic

This study applies the previously selected **two-rectangle contrast** to the
six July 2024 simulation scenarios. It is intentionally not the later
seven-point odd/even contrast experiment and it does not search for a new
contrast.

## Quantity being tracked

For the eight ordered points

```text
A-_t, A+_t, A-_t+1, A+_t+1,
B-_t, B+_t, B-_t+1, B+_t+1,
```

the two contrasts are

```text
Q_A = A-_t - A+_t - A-_t+1 + A+_t+1
Q_B = B-_t - B+_t - B-_t+1 + B+_t+1.
```

The primitive off-diagonal center is the raw second cross moment

```text
H_AB = E[Q_A Q_B].
```

The primary reported diagnostic is its contribution to the original selected
linear contrast:

```text
cross term = 2*d1*d2*H_AB
cross excess = 2*d1*d2*(H_AB - H_sep,AB).
```

`H_AA=E[Q_A^2]` and `H_BB=E[Q_B^2]` are retained as controls. The centered
sample covariance is saved only as a descriptive sensitivity quantity. Since
the translated contrasts overlap and are correlated, subtracting their sample
means does not preserve the raw oracle expectation; the centered sensitivity
must not be interpreted as an unbiased eta estimator.

The selected pair is co-centered and unequal-length. In standardized moving
coordinates its clockwise geometry is

```text
A: (-2,-2) -> ( 2, 2)
B: (-1,-2) -> ( 1, 2),
```

with the reflected counterclockwise version also included. Physical offsets
are recomputed from each simulation truth as
`(standardized_lat*range_lat, standardized_lon*range_lon)`. The old real-data
physical offsets based on ranges near `(0.815,0.953)` are not reused.

At time slot `j`, every endpoint is translated by
`j*(advec_lat,advec_lon)`. Thus the diagnostic is flow-following in the
Lagrangian coordinate system used to define the eta experiment. It removes
the known simulation mean before forming contrasts. The generator evaluates
each response at the nearest fine, de-advected FFT lattice point. The
diagnostic reconstructs that sampling coordinate from the truth JSON and uses
it for the analytic pointwise covariance. Raw source and regular-grid snapping
errors remain in the audit output. If the generator used spectral correction,
the truth table is an analytic target with the recorded covariance-distortion
bound, rather than a claim that the corrected FFT field has mathematically
exact unmodified covariance.

## Why the cross center diagnoses eta

For each actual sample, the code computes the two endpoint matrices

```text
H_sep   = W K_sep W'
H_joint = W K_joint W'
H_eta   = (1-eta) H_sep + eta H_joint.
```

Therefore the daily moment estimator

```text
eta_hat = sum(Q_A*Q_B - H_sep,AB) / sum(H_joint,AB - H_sep,AB)
```

targets the known eta. The separable endpoint does **not** have zero `H_AB`;
the signal is movement away from its separable center.

For the ideal unsnapped flow geometry, the no-jitter analytic targets are:

| family | eta | H_AA | H_AB | H_BB |
|---|---:|---:|---:|---:|
| Matérn 0.5 | 0 | 15.683790 | 5.683924 | 15.558991 |
| Matérn 0.5 | 0.5 | 15.709758 | 3.657794 | 15.642605 |
| Matérn 0.5 | 1 | 15.735726 | 1.631664 | 15.726218 |
| normalized GC | 0 | 16.055030 | 5.616577 | 15.813509 |
| normalized GC | 0.5 | 16.192201 | 3.541036 | 16.065318 |
| normalized GC | 1 | 16.329373 | 1.465494 | 16.317127 |

The corresponding ideal weighted cross terms for eta `(0, 0.5, 1)` are
`(0.274242, 0.176484, 0.078726)` for Matérn and
`(0.270993, 0.170850, 0.070708)` for normalized GC. Relative to the matched
separable center, their ideal signed excesses are respectively
`(0, -0.097758, -0.195516)` and `(0, -0.100142, -0.200284)`.

Actual daily oracle values need not equal these constants exactly because the
real GEMS endpoints are snapped first to the observation grid and then to the
fine simulation lattice. The output therefore compares the empirical center
with a pointwise analytic oracle at the reconstructed sampling lattice points.

The original selected-filter sign is also retained as

```text
empirical_l_cross_excess_vs_separable
  = 2*d1*d2*(empirical H_AB - separable H_AB),
```

where `d1=0.156629367196063` and `d2=0.1540219594610364`. Its ideal eta-one
target is `-0.1955164` for Matérn and `-0.2002845` for normalized GC.

## Restart contract

One 12-hour Slurm job processes scenarios 0 through 5 sequentially, and each
scenario processes its 30 days sequentially. For every completed day it:

1. validates the generator's day checkpoint and hashes its source files;
2. atomically writes
   `scenarios/<scenario>/day_json/YYYY-MM-DD.json`;
3. scans the valid JSONs and atomically rebuilds that scenario's
   `daily_two_contrast_centers.csv` and `daily_two_contrast_strata.csv`;
4. rebuilds the shared master CSV under a file lock.

The JSON is the source of truth. If a job stops after the JSON write but before
the CSV update, the next invocation repairs the CSV before doing new work.
CSV files are never concurrently appended.

A cached day is skipped only when the design/code signature, scenario, date,
block, truth hash, generator day-request hash, and hashes of `SUCCESS.json`,
`manifest.csv`, and `gridded.pkl` still match. A rebuild or final aggregation
rechecks those source hashes before accepting a day. Scenario and master
completion markers are invalidated first and recreated only after current
inputs pass, so a stale CSV cannot remain marked final.

## Amarel run

From the local repository:

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1

# Deterministic analytic regression test.
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/\
sim_two_contrast_cross_center_092726/amarel_two_contrast_diagnostic.sh validate-local

# Upload and run one date for all six scenarios.
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/\
sim_two_contrast_cross_center_092726/amarel_two_contrast_diagnostic.sh upload-submit-pilot

# Inspect and then run/restart all 30 dates.
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/\
sim_two_contrast_cross_center_092726/amarel_two_contrast_diagnostic.sh progress
bash Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/\
sim_two_contrast_cross_center_092726/amarel_two_contrast_diagnostic.sh submit-full
```

This is CPU-only: one `main` job, two CPUs, 16 GiB, and a 12-hour limit.
Aggregation runs at the end of that same job; it writes
provisional summaries for pilot output and writes
`FINAL_COMPLETE.json` only after all 180 unique scenario-days exist. In full
mode the single job exits nonzero after writing the provisional report if
the 180-day contract is incomplete, so Slurm does not silently label a partial
study successful.

## Main outputs

```text
<output>/scenarios/<scenario>/day_json/YYYY-MM-DD.json
<output>/scenarios/<scenario>/daily_two_contrast_centers.csv
<output>/scenarios/<scenario>/daily_two_contrast_strata.csv
<output>/all_daily_two_contrast_centers.csv
<output>/all_daily_two_contrast_strata.csv
<output>/scenario_day_summary.csv
<output>/paired_eta_effects_by_day.csv
<output>/master_progress.json
<output>/RESULTS.md                 # complete only
<output>/FINAL_COMPLETE.json        # 180/180 only
```

Days are the independent inferential units. The many translated, overlapping
within-day contrasts are used to estimate one daily center; they are not
treated as independent replicates. Same-date scenarios use common random
numbers, so eta effects should be compared by paired date within family.
