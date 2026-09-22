# GEMS-TCO

GEMS-TCO is a research package for covariance inference with geostationary
total-column ozone observations. The maintained library is organized by
statistical method and kept separate from experiment notebooks, generated
outputs, and retired implementations.

## Maintained package

- `GEMS_TCO.debiased_whittle` provides scalar-filter, mixed-frequency, and
  vector-gradient Debiased Whittle estimators.
- `GEMS_TCO.vecchia` provides the grouped, batched spatio-temporal Vecchia
  engine and reviewed corridor-neighbor configurations.
- `GEMS_TCO.spatial` provides pure-spatial Matérn and generalized-Cauchy
  block-Vecchia models and a direct Matérn likelihood.
- `GEMS_TCO.data` provides raw-orbit preprocessing and model-tensor loading.
- `GEMS_TCO.orderings` provides max-min ordering and deterministic predecessor
  neighbors.

Only maintained APIs live under `src/GEMS_TCO`. Work in progress belongs under
`research/diagnostics`; retired code is preserved under `research/legacy` and
is not installed.

## Installation

Python 3.10 or later and a C++ compiler are required. The build compiles one
private pybind11 extension, `GEMS_TCO._maxmin`, for max-min ordering.

```bash
python -m pip install .
```

For an editable checkout with development tools:

```bash
python -m pip install -e '.[dev]'
```

The smoothness-0.5 Vecchia path has optional fused PyTorch C++/CUDA
accelerators. Build the CPU extension only after PyTorch is installed in the
active environment:

```bash
GEMS_TCO_BUILD_TORCH_EXT=1 python -m pip install -e . --no-build-isolation
```

On an NVIDIA build host, the CUDA extension is independently enabled with
`GEMS_TCO_BUILD_CUDA_EXT=1`. A single Amarel build can target both A100 and
L40S as follows (CUDA 11.8 or newer is required for L40S `sm_89`):

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
GEMS_TCO_BUILD_TORCH_EXT=1 \
GEMS_TCO_BUILD_CUDA_EXT=1 \
python -m pip install -e . --no-build-isolation
```

The build and parity-smoke workflow used on Amarel is documented in
[`scripts/amarel/README.md`](scripts/amarel/README.md). Because this likelihood
uses float64 Cholesky factorizations, A100 is generally preferable to L40S for
the full fit even though both architectures are supported.

`covariance_backend="auto"` uses the accelerator matching supported CPU or
CUDA float64 tensors and otherwise falls back to the reference PyTorch
implementation.
`covariance_backend="torch"` forces the portable path for reproducibility
checks, while `"native"` requires the compiled accelerator. The native path
provides the parameter gradients needed for fitting; use the Torch path for
higher-order derivatives or coordinate gradients.

The source and wheel tests were last validated with Python 3.12.3, NumPy
1.26.4, SciPy 1.15.2, pandas 2.2.3, scikit-learn 1.6.1, PyTorch 2.5.1,
xarray 2025.4.0, netCDF4 1.7.2, and pybind11 2.13.6. These versions are a
reproducibility record, not a claim that they are the oldest supported
versions; minimum-version CI remains a release-policy decision.

## Public entry points

```python
from GEMS_TCO.debiased_whittle import DebiasedWhittleEngine
from GEMS_TCO.spatial import (
    AnisotropicMaternSpatialVecchia,
    fit_full_matern,
    fit_vecchia_matern_from_batches,
    vecchia_batches_to_numpy,
)
from GEMS_TCO.vecchia import GroupedBatchedVecchia
from GEMS_TCO.vecchia.corridor_neighbors import (
    Lag432CorridorVecchia,
    Lag643CorridorVecchia,
)

whittle = DebiasedWhittleEngine("cross_difference")
```

For SciPy fitting of a precomputed pure-spatial Vecchia graph, use the paired
public API `vecchia_batches_to_numpy(model)` followed by
`fit_vecchia_matern_from_batches(...)`. The adapter requires
`model.precompute_conditioning_sets()` to have completed successfully.

The supported scalar-filter names are `identity`, `latitude_difference`,
`longitude_difference`, `cross_difference`, and
`summed_first_differences`. Numeric experiment-era names are deliberately not
part of the package API.

The 4/3/2 and 6/4/3 Vecchia labels state the conditioning-block budgets at the
current, first-lag, and second-lag layers. Both fixed-longitude and directional
corridor variants have explicit class names under
`GEMS_TCO.vecchia.corridor_neighbors`.

All maintained Matérn implementations use the same standard range convention,
with Bessel argument `sqrt(2 * nu) * distance / range`. See
[the numerical and scientific conventions][method-conventions] before comparing
fits: it records coordinate units, mean designs, nugget and jitter policies,
conditioning-graph assumptions, objective normalization, and data alignment
requirements.

## Verification

Run the complete package and numerical-regression suite from the repository
root:

```bash
python -m unittest discover -s tests -v
```

The suite checks public-package boundaries, data alignment, filter behavior,
optimizer state, Matérn conventions, ordering determinism, and likelihood
goldens for the maintained Whittle and corridor-Vecchia paths. The GitHub
Actions workflow builds a wheel, validates its metadata, installs that wheel,
and runs the same tests against the installed package.

## Reproducibility and archive policy

The source package is the publication artifact; it is not an archive of every
exploratory run. Older notebooks and scripts under `Exercises` and
`GEMS_TCO_EDA` may require import updates. A pre-refactor tag may be retained
locally as a private rollback snapshot, but it contains historical credentials
and must not be pushed or included in the clean public release.
Machine-specific environment and cloud-deployment notes are preserved only as
unmaintained historical material under `research/legacy/setup`; they are not
installation instructions for this release.

See [the refactor manifest][refactor-manifest] for the canonical module map and
archive policy. Large input data and generated outputs are not part of the
Python distribution.

Before any public repository release, revoke and rotate the historical data-API
credential that appeared in earlier Git history, then publish from a verified
clean export or scrubbed history. Removing the credential from the current tree
does not invalidate a credential already present in old commits.

## Citation and license

Citation metadata is provided in [`CITATION.cff`][citation-metadata]. GEMS-TCO
is distributed under the MIT License; adapted third-party ordering code is
identified in [`THIRD_PARTY_NOTICES.md`][third-party-notices].

[method-conventions]: https://github.com/cl20813/GEMS_TCO/blob/main/docs/method_assumptions.md
[refactor-manifest]: https://github.com/cl20813/GEMS_TCO/blob/main/docs/package_refactor_manifest.md
[citation-metadata]: https://github.com/cl20813/GEMS_TCO/blob/main/CITATION.cff
[third-party-notices]: https://github.com/cl20813/GEMS_TCO/blob/main/THIRD_PARTY_NOTICES.md
