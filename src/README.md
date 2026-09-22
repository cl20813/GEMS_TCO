# GEMS-TCO

GEMS-TCO provides research implementations of Debiased Whittle and Vecchia
inference for spatio-temporal total column ozone models.

The stable package code is organized around the inference method:

- `GEMS_TCO.debiased_whittle` provides a common engine with named spatial
  filters while preserving the historical module imports.
- `GEMS_TCO.vecchia` provides descriptive paths for Vecchia conditioning
  configurations while preserving the established implementations.
- `GEMS_TCO.data` provides stable paths to the existing data loading and
  preprocessing logic.

The holding area for experimental diagnostics is outside the installed package
under `research/diagnostics`.  Existing diagnostic implementations have not
yet been moved; see the repository-level documentation and
`docs/package_refactor_manifest.md` for the compatibility policy.
