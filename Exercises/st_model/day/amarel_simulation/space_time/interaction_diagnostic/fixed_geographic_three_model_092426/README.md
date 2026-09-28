# Fixed-geographic three-model interaction diagnostic

This directory is the frozen Amarel study for July 2024--2025. It compares
three independently fitted covariance specifications on the same observed
mixed space--time feature:

1. joint generalized Cauchy with fixed `(alpha, beta) = (0.75, 1)`;
2. joint Matérn with fixed smoothness `nu = 0.5`;
3. advected-separable exponential spatial × temporal covariance.

The third model is fitted independently with the same grouped-batched
lag-6/4/3 corridor Vecchia engine. All three use a nugget fixed at zero.
The Amarel production setting uses model-specific target chunks: GC uses 128,
while Matérn 0.5 and separable use 256. The earlier lag-6/4/3 benchmark
establishing 256 was for the fused smoothness-0.5 Matérn kernel, not GC. GC now
has its own exact fused CUDA forward/backward, but its full-likelihood peak
memory has not yet been measured on Amarel. Starting at 128 gives materially
more headroom; chunking changes only computational partitioning, not the 4x4
target blocks or the likelihood definition.
The scheduler requests exactly one GPU from Amarel's `gpu` partition for at
most 5 hours. The default `compatible` profile selects the combined pool
`gpu[015-017,019-048]` (excluding invalid `gpu018`) and accepts either an A100
`sm_80` or an actual L40S `sm_89`. Explicit `a100` and `l40s` profiles remain
available for controlled comparisons. The allocated model and capability are
checked before building or fitting, so a node-name assumption cannot silently
select another device. There is no Slurm array or concurrent GPU work.

The default request is `1 node / 1 task / 1 GPU / 12 CPUs / 128 GiB host RAM /
5 hours`.  The `128G` Slurm memory request is system RAM, not GPU VRAM; the
allocated A100 or L40S contributes its own full device memory.  The wrapper
prints every request before submission and the job records the effective
Slurm resources plus `nvidia-smi` output.  Override only when needed with
`GEMS_TCO_SLURM_TIME`, `GEMS_TCO_SLURM_MEM`, or `GEMS_TCO_SLURM_CPUS`.

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
the date, model where applicable, lag pattern `6/4/3`, the actual model-specific
chunk size, frozen design identifier and SHA-256, study signature, and Git
revision. The manifest records the complete chunk map and hashes of the Python,
C++, and CUDA executable sources, which remains informative when uploaded
working-tree code is newer than the recorded Git commit. Each model fit also
records peak CUDA allocated and reserved bytes, so GC can be promoted from 128
to 256 later only from measured A100/L40S full-likelihood headroom.

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
The helper first makes the persistent Amarel build backend satisfy the
`setuptools>=77` requirement in `pyproject.toml`, then performs a
build-isolated, no-runtime-dependency editable install, rebuilds the Linux
max-min pybind extension, verifies the new public imports and package origin,
and checks that a representative deleted legacy module is absent.  The
persistent-backend check is required because the later CUDA build must use the
active Torch environment with build isolation disabled; both paths therefore
parse the same modern SPDX license metadata correctly.

Inside the allocated GPU job, the Slurm prologue loads CUDA 12.1 and performs
a forced native rebuild followed by a non-isolated editable install with
`GEMS_TCO_BUILD_CUDA_EXT=1`.  The architecture list is explicitly reset to
`8.0;8.9`, so a stale user environment or an older `sm_80`-only object cannot
remove L40S support.  This creates the Linux x86-64 CUDA binary locally on
Amarel for both A100 (`sm_80`) and L40S (`sm_89`); no macOS binary is uploaded.
Matérn and GC CUDA
covariance/gradient parity tests and a small benchmark must pass before any
daily fit starts. The Amarel configuration sets
`gc_covariance_backend = "native"`, so a missing or stale GC kernel fails
immediately instead of silently reverting to the high-memory Torch path.

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

The simplest local command uploads, verifies, and then submits the two-date
smoke job to the combined compatible pool:

```bash
bash scp_fixed_geo_three_model.sh push-smoke compatible
```

The helper opens one reusable SSH control connection, so password-based login
is requested once rather than once per verification/SCP step. macOS extended
attributes are omitted from the transfer archive, and the remote data
preflight explicitly checks `/home/jl2815/tco/data` rather than the local
`/Users/.../GEMS_DATA` path. Before rebuilding the package, it also verifies
and prints the sizes of `pickle_2024/tco_grid_24_07.pkl` and
`pickle_2025/tco_grid_25_07.pkl` on Amarel.

The corresponding A100 smoke command is

```bash
bash scp_fixed_geo_three_model.sh submit-smoke a100
```

The currently idle `gpuk[...]` and `volta[...]` nodes are not A100/L40S nodes.
`volta` is the V100 generation, and the `gpuk` name does not establish an L40S
device. They are deliberately excluded. A `MIX` node can still have a free GPU
slot, so the combined compatible pool is preferable to an idle but wrong
architecture.

After checking the single L40S smoke log and the two date directories, submit
all 60 dates. A100 remains the recommended production profile because this
likelihood is dominated by float64 Cholesky/solve. One Slurm job holds one GPU
and processes task IDs `0` through `59` in order; no job array or concurrent
GPU work is used:

```bash
bash scp_fixed_geo_three_model.sh submit-full a100
```

An L40S full run is supported with `submit-full l40s`, but its two-date smoke
time should be used to decide whether 5 hours is sufficient. The run is
restartable: if the wall-time expires, resubmission skips only dates with a
validated completion marker.

For example, after verifying that `gpu029` is currently an L40S, a named-node
24-hour run can be submitted from the local Mac with

```bash
GEMS_TCO_L40S_NODELIST=gpu029 \
GEMS_TCO_SLURM_TIME=24:00:00 \
bash scp_fixed_geo_three_model.sh submit-full l40s
```

These overrides are forwarded by the SCP helper to the remote submit wrapper.

The GC-native run writes to
`fixed_geographic_three_model_gc128_native_092526`, leaving the earlier partial
Torch-GC/OOM output untouched. Completed dates and model fits within this new
signature are reused on resubmission. Therefore, if the
5-hour job stops or one date fails, rerunning `full` skips completed dates and
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
