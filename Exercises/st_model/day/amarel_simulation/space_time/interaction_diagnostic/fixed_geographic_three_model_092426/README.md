# Fixed-geographic three-model interaction diagnostic

This directory is the frozen Amarel study for July 2024--2025. It compares
three independently fitted covariance specifications on the same observed
mixed space--time feature:

1. joint generalized Cauchy with fixed `(alpha, beta) = (0.75, 1)`;
2. joint Matérn with fixed smoothness `nu = 0.5`;
3. advected-separable exponential spatial × temporal covariance.

The third model is fitted independently with the same grouped-batched
lag-6/4/3 corridor Vecchia engine. All three use a nugget fixed at zero.
The Amarel production setting fixes `target_chunk_size = 256`, matching the
lag-6/4/3 benchmark in which 256 was the fastest tested batch size without
changing the final NLL.
The scheduler requests exactly one GPU from Amarel's `gpu` partition for at
most 12 hours and selects the mixed-state A100 node `gpu020`. This node choice
is not a separate partition or an array. There is no Slurm array or concurrent
GPU work.

## Frozen diagnostic

- Coordinates are fixed geographic coordinates. Observation points never
  follow a fitted advection path; advection appears only inside each model's
  covariance.
- Both mirror versions of the July 1 physical A/B geometry are fixed.
- `Q_A = A_t - A_{t+1}` and `Q_B = B_t - B_{t+1}`; lag 1 is fixed.
- Time follows the established package convention: raw `Hours_elapsed` is
  rounded to nominal integer-hour GEMS slots after subtracting `477700.0`.
  Raw and rounded adjacent increments are both recorded; covariance scoring
  uses exactly the same rounded times as the Vecchia fits.
- `d1 = 0.156629367196063` and `d2 = 0.1540219594610364` are used only for
  the targeted secondary `Var(L)` decomposition.
- The primary diagnostic is the constant-free mean bivariate Gaussian
  contrast score using each sample's own fitted `2 × 2` covariance.
- `||W X||_inf` is audited on the actual snapped source coordinates. Raw
  monthly-centered `Y` is used only if it is numerically zero; otherwise one
  common daywise OLS mean is removed for all three models.
- The score treats that common fitted mean as a shared plug-in nuisance
  estimate. The output records the RMS and maximum change it makes to both
  contrasts; it does not pretend that same-day mean estimation has no effect.
- Every actual source endpoint is compared with its frozen desired physical
  endpoint using Euclidean distance after latitude/longitude are divided by
  the frozen ranges. Per-point/per-sample errors and their daily median, 95th
  percentile, and maximum are retained in the audit outputs.
- Desired-endpoint to regular-grid snapping error is recorded with the same
  standardized Euclidean metric.
- Complete contrast counts and fractions are recorded for every mirror and
  adjacent time pair because cloud/missingness coverage differs by day. Model
  comparisons remain paired within exactly the same available contrasts.
- The unit of comparison is a day. Overlapping anchors are never counted as
  independent inferential replicates.
- A model/day is complete only after its final absolute gradient is below the
  frozen outer threshold `1e-4`; unconverged fits are failed rather than
  silently scored.

The exact locked values and provenance hashes are in `frozen_design.json`.
Every date manifest, model fit, model score summary, and daily score row records
the date, model where applicable, lag pattern `6/4/3`, batch size `256`, frozen
design identifier and SHA-256, study signature, and Git revision. The manifest
also records hashes of the executable source files, which remains informative
when uploaded working-tree code is newer than the recorded Git commit.

## Design-held-out evaluation dates

The predeclared window is all of July 2024 and July 2025. The discovery day
`2024-07-01` is excluded, and `2025-07-24` has only seven hourly frames. Both
July 31 dates have eight frames, so the locked `evaluation_dates.csv` contains
exactly 60 design-held-out evaluation days.

“Design-held-out” means that these dates were not used to select the A/B
geometry, lag, signs, or coefficients. Each covariance model is nevertheless
fit and scored on the same day's data. The score differences are therefore
targeted in-sample covariance-adequacy comparisons among three
equal-dimensional covariance specifications, not out-of-sample predictive
scores.

## Local preflight

```bash
/opt/anaconda3/envs/faiss_env/bin/python validate_fixed_geo_three_model.py --check-data
```

This checks the manifest, frozen signs/geometry, both mean-handling branches,
the covariance formulas, and all local date/slot counts. It does not fit a
model.

## Upload and run

The upload is package-first. It creates a clean archive of the current local
working-tree package, excluding caches and local binaries, verifies the SCP
checksum, and replaces the remote `src/GEMS_TCO` tree so modules deleted by the
recent refactor cannot remain importable. It also copies all C++ sources,
packaging metadata, tests, package documentation, and Amarel build helpers.
The helper then performs a build-isolated, no-runtime-dependency editable
install, rebuilds the Linux max-min pybind extension, verifies the new public
imports and package origin, and checks that a representative deleted legacy
module is absent. Build isolation is intentional: it honors the
`setuptools>=77` requirement in `pyproject.toml`, which is needed to parse the
modern SPDX license metadata even when Amarel's persistent environment has an
older Setuptools release.

Only after those checks pass does it replace and upload this code-only study
directory and run the remote no-fit preflight over all 60 date/slot entries.
Data and `/home/jl2815/tco/exercise_output` are never replaced or deleted. The
older 55 MB `interaction_diagnostic` artifacts are not uploaded.

```bash
bash scp_fixed_geo_three_model.sh push
```

Run that command in the local Mac terminal, not inside an Amarel login shell.
The default endpoint is `jl2815@amarel-new.hpc.rutgers.edu`; set
`AMAREL_HOST` only when Rutgers explicitly provides another login endpoint.

Start with one GPU job that processes two dates sequentially, one from each
year (`2024-07-02` and `2025-07-01`):

The simplest local command uploads, verifies, and then submits the smoke job:

```bash
bash scp_fixed_geo_three_model.sh push-smoke
```

The helper opens one reusable SSH control connection, so password-based login
is requested once rather than once per verification/SCP step. macOS extended
attributes are omitted from the transfer archive, and the remote data
preflight explicitly checks `/home/jl2815/tco/data` rather than the local
`/Users/.../GEMS_DATA` path. Before rebuilding the package, it also verifies
and prints the sizes of `pickle_2024/tco_grid_24_07.pkl` and
`pickle_2025/tco_grid_25_07.pkl` on Amarel.

After checking the single job log and the two date directories, submit all 60
dates. One Slurm job holds one GPU and processes task IDs `0` through `59` in
order; no job array or concurrent GPU work is used:

```bash
bash scp_fixed_geo_three_model.sh submit-full
```

Completed dates and model fits are reused on resubmission. Therefore, if the
12-hour job stops or one date fails, rerunning `full` skips completed dates and
continues in manifest order. An incompatible configuration signature is
rejected rather than mixed with old output. Aggregation runs at the end of
that same one-GPU job; no second scheduler job is submitted.

Download results:

```bash
bash scp_fixed_geo_three_model.sh pull
```

Each pull goes into a new timestamped subdirectory under
`downloaded_results/`, so files from an older or partial run cannot silently
survive in a later download. Set `DOWNLOAD_TAG` explicitly if a named snapshot
is preferred.

## Outputs

Each sequential step owns one date directory and writes separate model checkpoints.
The larger per-anchor empirical and fitted tables are gzip-compressed CSVs so
the 60-day result transfer remains manageable.
The final step of the same sequential GPU job always creates completeness and
available descriptive tables. It creates inferential intervals, final plots,
`RESULTS.md`, and `FINAL_COMPLETE` only after all 60 tasks are complete;
otherwise it writes a visibly provisional report. Final outputs are:

- `completeness.csv` and `missing_task_ids.txt`;
- `all_daily_model_scores.csv`;
- `daily_pairwise_score_differences.csv`;
- `daily_moment_diagnostics.csv`, with empirical/model `Q_A`, `Q_B`, cross,
  correlation, and `L` decomposition columns;
- `day_block_bootstrap_intervals.csv`;
- `daily_paired_contrast_scores.{png,pdf}`;
- `RESULTS.md` and, only when all dates complete, `FINAL_COMPLETE`.

The two primary reported differences are

```text
Score(JM)  - Score(GC)  # positive favors GC
Score(Sep) - Score(JM)  # positive favors joint Matérn
```

Day-level uncertainty uses circular moving blocks within contiguous
calendar-day segments of each year. Blocks never bridge the excluded
`2025-07-24` date, and overlapping within-day contrasts are never resampled
as if they were iid observations.

The resulting language must remain specification-specific: this study does
not rank every possible generalized-Cauchy, Matérn, or separable model, and it
is not a formal separability test.

Because all three models are independently fitted, their fitted marginal
behavior is not forced to match. The primary `2 x 2` score is therefore a
diagnostic of targeted mixed-lag covariance adequacy, not a mathematically
isolated “interaction-only” estimand. The diagonal/cross decomposition is what
shows whether a daily score advantage is actually driven by `C_AB` rather
than by the two individual second moments.

Empirical quantities used by the zero-mean Gaussian score are labeled as
second moments (`E[Q_A^2]`, `E[Q_B^2]`, and `E[Q_A Q_B]`), not silently called
centered variances. Centered variance/covariance summaries are also retained
as a sensitivity description.
