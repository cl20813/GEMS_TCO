# Archived package source

This directory contains historical implementations and machine-specific
configuration that are intentionally excluded from the installed package.
They are retained only for reproducing older experiments.

- `vecchia/` contains retired point-target, hybrid, and column-batch engines.
- `vecchia_candidate/` contains exploratory candidate implementations.
- `not_used/` contains older modules that had already fallen out of use.
- `configuration.py` records former workstation/HPC paths and must not be
  imported by maintained package code.

New library work belongs under `src/GEMS_TCO`; archived files should not be
added to the package import path.
