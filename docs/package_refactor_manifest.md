# Package refactor manifest

This refactor changes package organization without intentionally changing any
statistical or numerical behavior.  The pre-refactor source is recoverable at
the local annotated Git tag `pre-package-refactor-2026-09-21` (commit
`ee987c84fc4b19f901464f45a7be1de0f96225e5`).

## Debiased Whittle filters

The public filter names describe the operation rather than the historical
coefficient shorthand.

| Legacy module suffix | Public filter name | Preserved operation |
| --- | --- | --- |
| `raw` | `identity` | No convolution; preserve the historical per-time-slice spatial demeaning |
| `lat1` | `latitude_difference` | First difference in latitude |
| `lon1` | `longitude_difference` | First difference in longitude |
| `1111` | `cross_difference` | Historical stencil `[[-1, 1], [1, -1]]`, equal to `-D_lat D_lon` under the forward-difference definitions |
| `2110` | `summed_first_differences` | `D_lat + D_lon` summed first differences |

New code should construct
`GEMS_TCO.debiased_whittle.DebiasedWhittleEngine` with a public filter name.
The five historical module paths remain as compatibility wrappers.  Their
stencils, output shapes, frequency masks, CPU dtype/device placement, jitter,
parameter ordering, and optimizer flows are protected by characterization
tests in `tests/test_debiased_whittle_engine.py`.  The historical ``1111``
sign is intentionally documented rather than normalized, because changing it
would change the preprocessing output.

`debiased_whittle_grad_filter.py` and `debiased_whittle_mixed.py` remain
experimental and their numerical implementations were not changed.

## Vecchia corridor configurations

The existing implementations remain the source of truth and are re-exported
through descriptive paths:

| Configuration | Canonical import path |
| --- | --- |
| Local 4/3/2 | `GEMS_TCO.vecchia.corridor_neighbors.local_lag432` |
| Amarel 6/4/3 | `GEMS_TCO.vecchia.corridor_neighbors.amarel_lag643` |

The original `vecchia_realdata_*` module paths remain valid.  Five unused
candidate copies/wrappers were removed after a repository-wide caller audit:

- `kernel_vecchia_col_batch.py`, whose only computational counterpart is
  `matern_vecchia_col_batch.py` (the removed copy differed only in progress
  message precision);
- `kernels_vecchia_cluster_hybrid.py`, `kernels_vecchia_hybrid.py`, and the
  byte-identical `kernels_vecchia_hybrid_fresh.py`, which only re-exported the
  canonical Matérn implementations; and
- `kernels_vecchia_same_spatial.py`, whose former implementation duplicated
  `matern_vecchia_engine.py`.

The three files left in `vecchia_candidate/` are not duplicates: the pointwise
Cauchy model, the regular-grid reverse-L template-reuse model, and the
missing-aware intersection model each have distinct behavior and current
research callers.

## Data and diagnostics

`GEMS_TCO.data.loading` and `GEMS_TCO.data.preprocessing` currently re-export
the existing loading functions and preprocessing functions/classes.  This
provides clearer import paths without rewriting data behavior.  The
`research/diagnostics/` holding area is outside the installed package; it
currently contains its policy only, because existing diagnostic code is not
moved until callers have compatibility coverage.  Existing `GEMS_TCO.evaluate`
callers are not migrated in this phase.

The seven historical modules formerly stored under the literal
`src/GEMS_TCO/not used/` directory were moved unchanged to
`research/legacy/GEMS_TCO/not_used/`.  They are not installed.  Four are named
by old notebooks through package paths that were already stale before this
move; the remaining three have no repository callers.

The obsolete package-internal `src/GEMS_TCO/setup.py` was removed.  The
repository's `src/setup.py` remains solely as the native-extension build
declaration, while project metadata and dependencies live in
`src/pyproject.toml`.

## Deferred work

- remove tracked platform-specific native binaries after the new reproducible
  C++ extension build has been exercised on every supported platform;
- migrate the native extension declarations from the build-only `src/setup.py`
  if a fully declarative backend is adopted later;
- promote the three actively used `vecchia_candidate` implementations to
  stable or explicitly experimental namespaces and migrate their callers;
- migrate research scripts from compatibility imports before removing any
  legacy module path.
