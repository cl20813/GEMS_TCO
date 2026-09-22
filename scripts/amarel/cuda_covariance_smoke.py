#!/usr/bin/env python3
"""Parity and microbenchmark smoke test for the native CUDA covariance path."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from collections.abc import Callable

import torch

from GEMS_TCO.vecchia._native_covariance import (
    native_covariance,
    native_covariance_available,
    torch_covariance_reference,
)

TARGET_CAPABILITIES = {(8, 0): "A100", (8, 9): "L40S"}


def _nll(
    covariance: torch.Tensor,
    response: torch.Tensor,
    is_dummy: torch.Tensor,
) -> torch.Tensor:
    response = response.masked_fill(is_dummy.unsqueeze(-1), 0.0)
    factor = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(factor, response, upper=False)
    log_determinant = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum()
    quadratic = whitened.square().sum()
    return 0.5 * (log_determinant + quadratic) / (~is_dummy).sum()


def _representative_inputs(
    batch_size: int,
    points: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    latitude = 33.0 + 4.0 * torch.rand(
        batch_size,
        points,
        generator=generator,
        dtype=torch.float64,
    )
    longitude = 124.0 + 8.0 * torch.rand(
        batch_size,
        points,
        generator=generator,
        dtype=torch.float64,
    )
    time_coordinate = torch.randint(
        0,
        3,
        (batch_size, points),
        generator=generator,
    ).to(torch.float64)
    coordinates = torch.stack((latitude, longitude, time_coordinate), dim=-1).to(
        device=device,
        dtype=torch.float64,
    )
    # Exercise exact-zero distance and dummy identity rows.
    if points > 1:
        coordinates[:, 1] = coordinates[:, 0]
    is_dummy = torch.zeros((batch_size, points), dtype=torch.bool, device=device)
    if points > 4:
        is_dummy[:, -2:] = True
    response = torch.randn(
        (batch_size, points, 1),
        generator=generator,
        dtype=torch.float64,
    ).to(device)
    params = torch.tensor(
        [0.25, -0.35, 0.45, -0.20, 0.018, -0.110, -2.50],
        dtype=torch.float64,
        device=device,
    )
    return params, coordinates, is_dummy, response


def _correctness_check(device: torch.device) -> dict[str, float]:
    initial, coordinates, is_dummy, response = _representative_inputs(2, 12, device)
    torch_params = initial.clone().requires_grad_(True)
    native_params = initial.clone().requires_grad_(True)

    torch_nll = _nll(
        torch_covariance_reference(torch_params, coordinates, is_dummy, smooth=0.5),
        response,
        is_dummy,
    )
    native_matrix = native_covariance(
        native_params,
        coordinates,
        is_dummy,
        smooth=0.5,
        backend="native",
    )
    native_nll = _nll(native_matrix, response, is_dummy)
    torch_gradient = torch.autograd.grad(torch_nll, torch_params)[0]
    native_gradient = torch.autograd.grad(native_nll, native_params)[0]

    torch_matrix = torch_covariance_reference(initial, coordinates, is_dummy, smooth=0.5)
    covariance_difference = float((native_matrix.detach() - torch_matrix).abs().max())
    nll_difference = float((native_nll.detach() - torch_nll.detach()).abs())
    gradient_difference = float((native_gradient - torch_gradient).abs().max())
    torch.testing.assert_close(native_matrix.detach(), torch_matrix, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(native_nll, torch_nll, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(native_gradient, torch_gradient, rtol=2e-8, atol=2e-10)
    return {
        "covariance_max_abs_difference": covariance_difference,
        "nll_abs_difference": nll_difference,
        "gradient_max_abs_difference": gradient_difference,
    }


def _time_step(step: Callable[[], None], iterations: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    elapsed: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
    return statistics.median(elapsed) * 1000.0


def _benchmark(
    device: torch.device,
    batch_size: int,
    points: int,
    iterations: int,
) -> dict[str, float | int]:
    initial, coordinates, is_dummy, _ = _representative_inputs(batch_size, points, device)

    def torch_step() -> None:
        params = initial.detach().clone().requires_grad_(True)
        covariance = torch_covariance_reference(params, coordinates, is_dummy, smooth=0.5)
        torch.autograd.grad(covariance.square().mean(), params)

    def native_step() -> None:
        params = initial.detach().clone().requires_grad_(True)
        covariance = native_covariance(
            params,
            coordinates,
            is_dummy,
            smooth=0.5,
            backend="native",
        )
        torch.autograd.grad(covariance.square().mean(), params)

    torch_milliseconds = _time_step(torch_step, iterations)
    native_milliseconds = _time_step(native_step, iterations)
    return {
        "batch_size": batch_size,
        "points": points,
        "iterations": iterations,
        "torch_forward_backward_median_ms": torch_milliseconds,
        "native_forward_backward_median_ms": native_milliseconds,
        "speedup": torch_milliseconds / native_milliseconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--points", type=int, default=144)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--allow-other-gpu",
        action="store_true",
        help="run even when the compute capability is not A100 sm_80 or L40S sm_89",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA in this allocation")
    if not native_covariance_available("cuda"):
        try:
            importlib.import_module("GEMS_TCO._vecchia_covariance_cuda")
        except (ImportError, OSError) as error:
            raise RuntimeError(
                f"GEMS_TCO._vecchia_covariance_cuda failed to import: {error}"
            ) from error
        raise RuntimeError("GEMS_TCO._vecchia_covariance_cuda is not importable")

    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability not in TARGET_CAPABILITIES and not args.allow_other_gpu:
        raise RuntimeError(
            f"unsupported smoke-test GPU capability sm_{capability[0]}{capability[1]}; "
            "expected A100 sm_80 or L40S sm_89"
        )

    metadata = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "target_family": TARGET_CAPABILITIES.get(capability, "other"),
    }
    result = {
        "environment": metadata,
        "parity": _correctness_check(device),
        "benchmark": _benchmark(
            device,
            batch_size=args.batch_size,
            points=args.points,
            iterations=args.iterations,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
