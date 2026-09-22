# Package refactor manifest

This refactor makes `src/GEMS_TCO` the maintained, installable research
library rather than a compatibility archive for historical experiments. The
pre-refactor tag is a private/local rollback snapshot only: it contains a
historical API credential and must not be pushed, mirrored, or included in the
clean public repository. Revoke and rotate that credential before release,
then publish from a verified clean export or scrubbed history without old tags.

The numerical paths retained in the package are protected by deterministic
regression tests. Historical notebooks and scripts under `Exercises/` and
`GEMS_TCO_EDA/` were deliberately not rewritten in this change; update their
imports when an experiment is brought back into active use.

## Maintained package layout

```text
src/GEMS_TCO/
├── __init__.py
├── data/
│   ├── loading.py
│   └── preprocessing.py
├── debiased_whittle/
│   ├── engine.py
│   ├── filters.py
│   ├── mixed_frequency.py
│   └── vector_gradient.py
├── spatial/
├── vecchia/
│   ├── grouped_batched.py
│   └── corridor_neighbors/
└── orderings.py
```

There are no compatibility wrappers for retired top-level module names.

## Debiased Whittle

The five formerly duplicated scalar implementations now use one configured
engine. Public filter names describe the statistical operation:

| Removed suffix | Public filter name | Preserved operation |
| --- | --- | --- |
| `raw` | `identity` | No convolution; per-time-slice spatial demeaning |
| `lat1` | `latitude_difference` | First latitude difference |
| `lon1` | `longitude_difference` | First longitude difference |
| `1111` | `cross_difference` | Historical stencil `[[-1, 1], [1, -1]]` |
| `2110` | `summed_first_differences` | `D_lat + D_lon` |

Construct scalar variants with
`GEMS_TCO.debiased_whittle.DebiasedWhittleEngine`. The old abbreviated filter
names and five top-level modules are not accepted.

The two genuinely different estimators remain separate:

- `GEMS_TCO.debiased_whittle.mixed_frequency` combines identity-filtered low
  frequencies with cross-differenced high frequencies.
- `GEMS_TCO.debiased_whittle.vector_gradient` uses the two-component spatial
  gradient and its cross-spectrum.

Common preprocessing, taper, covariance, expected-periodogram, and optimizer
logic is inherited from the private `_core` implementation. Obsolete
comparison orchestration and the undefined full/Vecchia experiment path were
removed from the library.

The scalar API accepts descriptive names only. Its grid, time-axis, frequency
mask, parameter-order, and numerical-loading assumptions are recorded in
[`method_assumptions.md`](method_assumptions.md).

## Vecchia

The supported spatio-temporal Vecchia implementation is the grouped,
GPU-batched engine plus corridor-neighbor geometries:

| Role | Canonical path |
| --- | --- |
| Grouped block-target engine | `GEMS_TCO.vecchia.grouped_batched` |
| Shared corridor geometry | `GEMS_TCO.vecchia.corridor_neighbors._geometry` |
| Fixed-longitude 4/3/2 corridor | `GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432` |
| Directional 4/3/2 corridor | `GEMS_TCO.vecchia.corridor_neighbors.directional_lag432` |
| Fixed-longitude 6/4/3 corridor | `GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643` |
| Directional 6/4/3 corridor | `GEMS_TCO.vecchia.corridor_neighbors.directional_lag643` |
| Generalized Cauchy | `GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy` |
| Matérn spline variants | `GEMS_TCO.vecchia.corridor_neighbors.spline` |

`vecchia/_base.py` contains only behavior required by grouped fits. Broken
point-target methods that assumed a different batch representation were not
carried into the supported base.

Retired point-target, hybrid, column-batch, and calibrated-corridor source is
stored under `research/legacy/GEMS_TCO/vecchia/`. The former
`vecchia_candidate` package is archived under
`research/legacy/GEMS_TCO/vecchia_candidate/`; it is not installed.

Public class names describe the model rather than the machine or dataset. The
principal classes are `GroupedBatchedVecchia`, `Lag432CorridorVecchia`,
`DirectionalLag432CorridorVecchia`, `Lag643CorridorVecchia`, and
`DirectionalLag643CorridorVecchia`. There are no `RealData`, `Amarel`,
`Hybrid`, or `Fit` compatibility aliases in the installed API.

## Pure-spatial models

Dated top-level filenames were replaced with the `GEMS_TCO.spatial` package:

| Former module | Canonical module |
| --- | --- |
| `kernels_space_base_engine_052126` | `spatial.base` |
| `kernels_space_iso_cluster_052426` | `spatial.isotropic` |
| `kernels_space_aniso_cluster_060326` | `spatial.anisotropic_matern` |
| `kernels_space_aniso_cauchy_cluster_060326` | `spatial.anisotropic_cauchy` |
| `matern_bessel_anisotropic` | `spatial.matern_bessel` |
| `matern_spline` | `spatial.matern_spline` |
| `torch_bessel_full_likelihood` | `research/diagnostics/spatial/torch_matern_bessel.py` |

`spatial.__init__` provides the curated model-level API. Public classes use
`SpatialVecchia` names, and the numerical methods are named
`profiled_negative_log_likelihood`, `estimate_gls_coefficients`,
`make_lbfgs_optimizer`, and `fit_lbfgs`. The direct and block-Vecchia paths use
the same standard `sqrt(2 * nu)` Matérn range convention. The `latlon` mean
design consistently means intercept plus centered latitude and longitude;
`latlon_hour` explicitly adds the seven hourly indicators.

## Data and diagnostics

The implementations now live directly in `GEMS_TCO.data.loading` and
`GEMS_TCO.data.preprocessing`; the former top-level data modules were removed.
Public classes use descriptive names: `ProcessedDataLoader`,
`CoordinateDeviationFilter`, `GEMSOrbitReader`, `GeographicBounds`, and
`MonthlyOrbitAggregator`. NetCDF groups are aligned by shared dimension
indices, and processed aggregate tensors are checked against hourly
coordinate/time order.

The package root is intentionally small and dependency-free. Experiment result
logging was removed from the publication API because it was orchestration, not
a statistical method.

The former `GEMS_TCO.evaluate` diagnostic collection was moved to
`research/diagnostics/evaluate.py`. The project-specific downloader was moved
to `research/data_acquisition/` and no longer embeds an API credential.
Machine-specific path constants were archived as
`research/legacy/GEMS_TCO/configuration.py`.

## Packaging and generated artifacts

`pyproject.toml`, `setup.py`, `README.md`, and `LICENSE` now live at the
repository root, following the standard `src` layout. `src/` contains package
source only.

Tracked macOS and Windows extensions and compiler intermediates (`.so`, `.pyd`,
`.obj`, `.lib`, `.exp`) were removed. Wheels compile one private pybind11
extension, `GEMS_TCO._maxmin`, from `cpp/maxmin_order.cpp`. Its adapted upstream
code and license are identified in `THIRD_PARTY_NOTICES.md`.

`CITATION.cff` supplies software-citation metadata, and
`.github/workflows/tests.yml` builds and checks a wheel before running the test
suite against the installed artifact. The workflow uses Python isolated mode
and explicitly rejects imports from the checkout's `src` directory. Historical
machine-specific setup and deployment notes live under
`research/legacy/setup/` and are not maintained installation instructions.

## Verification

- The complete unit/regression suite passes from both the source tree and an
  installed wheel.
- Scalar, mixed-frequency, vector-gradient, and 4/3/2 versus 6/4/3 corridor
  likelihood goldens are preserved.
- Data-group alignment, invalid optimizer states, public imports, deterministic
  orderings, mean designs, and shared Matérn conventions have focused
  regression tests.
- Black, isort, Pyflakes, and `git diff --check` pass for the maintained source
  and tests.
- The built wheel contains only maintained package modules, metadata, and the
  single locally compiled native extension.
