#!/usr/bin/env python3
"""Verify an Amarel editable install after an exact GEMS_TCO source sync."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import sysconfig
from pathlib import Path

import numpy as np
import torch

import GEMS_TCO
from GEMS_TCO import _maxmin
from GEMS_TCO.data import ProcessedDataLoader
from GEMS_TCO.orderings import maxmin_order
from GEMS_TCO.vecchia.corridor_neighbors import (
    NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia,
    NoNuggetGeneralizedCauchyLag643CorridorVecchia,
    NoNuggetSplineMaternLag643CorridorVecchia,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, required=True)
    result.add_argument("--require-distribution-metadata", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    expected_init = project_root / "src" / "GEMS_TCO" / "__init__.py"
    installed_init = Path(GEMS_TCO.__file__).resolve()
    if installed_init != expected_init:
        raise AssertionError(f"GEMS_TCO imported from {installed_init}, expected {expected_init}")
    try:
        installed_version = importlib.metadata.version("GEMS-TCO")
    except importlib.metadata.PackageNotFoundError:
        if args.require_distribution_metadata:
            raise AssertionError("editable GEMS-TCO distribution metadata is missing") from None
        installed_version = GEMS_TCO.__version__
    if installed_version != "0.3.0":
        raise AssertionError("unexpected installed GEMS-TCO distribution version")

    # Native extensions are not portable across operating systems or Python
    # ABIs.  Confirm that the source-only sync was rebuilt for this exact host
    # instead of accidentally loading a copied macOS/Windows artifact.
    extension_path = Path(_maxmin.__file__).resolve()
    expected_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not expected_extension_suffix or not extension_path.name.endswith(
        expected_extension_suffix
    ):
        raise AssertionError(
            f"native extension {extension_path.name!r} does not match this "
            f"interpreter suffix {expected_extension_suffix!r}"
        )
    if extension_path.parent != expected_init.parent:
        raise AssertionError(
            f"native extension imported from {extension_path}, expected directory "
            f"{expected_init.parent}"
        )
    binary_header = extension_path.read_bytes()[:4]
    system = platform.system()
    if system == "Linux" and binary_header != b"\x7fELF":
        raise AssertionError(f"Amarel extension is not an ELF binary: {extension_path}")
    if system == "Darwin" and binary_header not in {
        b"\xcf\xfa\xed\xfe",  # little-endian 64-bit Mach-O
        b"\xfe\xed\xfa\xcf",  # big-endian 64-bit Mach-O
    }:
        raise AssertionError(f"macOS extension is not a Mach-O binary: {extension_path}")
    if system == "Windows" and binary_header[:2] != b"MZ":
        raise AssertionError(f"Windows extension is not a PE binary: {extension_path}")

    for retired_module in (
        "GEMS_TCO.configuration",
        "GEMS_TCO.data_loader",
        "GEMS_TCO.vecchia_candidate",
    ):
        if importlib.util.find_spec(retired_module) is not None:
            raise AssertionError(f"retired module remains importable: {retired_module}")

    locations = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    order = maxmin_order(locations)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(locations)))

    rows = torch.zeros((4, 11), dtype=torch.float64)
    input_map = {"slot_0": rows}
    models = (
        NoNuggetGeneralizedCauchyLag643CorridorVecchia(
            gc_alpha=0.75,
            gc_beta=1.0,
            input_map=input_map,
            target_chunk_size=256,
        ),
        NoNuggetSplineMaternLag643CorridorVecchia(
            smooth=0.5,
            input_map=input_map,
            target_chunk_size=256,
        ),
        NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(
            input_map=input_map,
            target_chunk_size=256,
        ),
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.3, 1.0], [0.4, 0.1, 2.0]]],
        dtype=torch.float64,
    )
    for model in models:
        if model.covariance_parameter_count != 6:
            raise AssertionError(f"unexpected parameter count for {type(model).__name__}")
        if model.resolved_covariance_backend() != "torch":
            raise AssertionError(f"unexpected covariance backend for {type(model).__name__}")
        raw = torch.tensor(
            [0.1, -0.2, 0.3, -0.1, 0.01, -0.126],
            dtype=torch.float64,
            requires_grad=True,
        )
        covariance = model.batched_covariance(raw, coordinates)
        (gradient,) = torch.autograd.grad(covariance.square().sum(), raw)
        if not torch.isfinite(covariance).all() or not torch.isfinite(gradient).all():
            raise AssertionError(f"non-finite covariance/gradient for {type(model).__name__}")

    print(
        "GEMS_TCO package verified:",
        GEMS_TCO.__version__,
        installed_init,
        extension_path,
        f"{system}/{platform.machine()}",
        expected_extension_suffix,
        ProcessedDataLoader.__name__,
    )


if __name__ == "__main__":
    main()
