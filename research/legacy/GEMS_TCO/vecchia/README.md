# Archived Vecchia implementations

This directory holds historical Vecchia implementations that are no longer
part of the installed `GEMS_TCO` package.  They were moved here on
2026-09-21 while the supported Vecchia path was narrowed to the grouped,
GPU-batched engine and the 4-by-4 corridor configurations.  The move did not
intentionally change any statistical or numerical logic.

The pre-refactor tag is a private/local rollback snapshot only. It contains
historical credentials and must not be included in a clean public release.

## What remains active

The grouped block-target calculation formerly embedded in a class named
`ClusterHybridVecchiaFit` is still required by the corridor models. Its
maintained implementation has names that describe the method rather than an
experiment or machine:

- `GEMS_TCO.vecchia.grouped_batched.GroupedBatchedVecchia` is the grouped,
  block-target, GPU-batched calculation engine;
- the private base in `GEMS_TCO.vecchia.corridor_neighbors._geometry`
  provides the corridor package's internal conditioning geometry; and
- `GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432` and
  `GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643` expose the supported
  fixed-longitude 4/3/2 and 6/4/3 corridor configurations;
- `GEMS_TCO.vecchia.corridor_neighbors.directional_lag432` and
  `GEMS_TCO.vecchia.corridor_neighbors.directional_lag643` expose the
  directional 4/3/2 and 6/4/3 configurations.

The adapted-direction, generalized-Cauchy, and spline variants also live under
`GEMS_TCO.vecchia.corridor_neighbors`.  There are no top-level compatibility
modules in the installed package.

## Archived files

| File | Historical role | Reason it is archived |
| --- | --- | --- |
| `matern_vecchia_hybrid.py` | Point-target hybrid Vecchia engine | Superseded for current work by grouped corridor fits |
| `matern_vecchia_engine_point_target.py` | Original point-target likelihood, preprocessing, and optimizer base | The supported grouped engine now uses a minimal grouped-only base under `vecchia/_base.py` |
| `matern_vecchia_col_batch.py` | Point-target column-batch implementation | Not part of the selected grouped corridor path |
| `matern_vecchia_cluster_hybrid.py` | Original grouped/block-target implementation | Preserved under its old name for history; the same active calculation core was renamed and moved to `vecchia/grouped_batched.py` |
| `matern_vecchia_cluster_column_batch.py` | Cluster-column batching experiment | Not part of the selected grouped corridor path |
| `vecchia_mm_space_spline.py` | Max-min spatial spline experiment | No current package caller |
| `vecchia_realdata_calibrated_shifted_center_4x4_lag643.py` | Calibrated shifted-center corridor candidate | Retired in favor of the explicit corridor configurations |
| `vecchia_st_spline_full.py` | Snapshot of the former all-in-one spline module | Retains the retired point-target and cluster-hybrid spline classes; the active package keeps only corridor spline classes |

Files in this directory are reference snapshots and should not be added to the
installed package import path.  Older scripts and notebooks under `Exercises/`
that import the retired module names are historical callers; they are not part
of the supported installed-package API. Use this source archive when
reproducing one of those experiments.

The retired exploratory candidate package is preserved separately at
`research/legacy/GEMS_TCO/vecchia_candidate/`.
