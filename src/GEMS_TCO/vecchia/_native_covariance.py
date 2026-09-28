"""Optional fused CPU/CUDA covariance assembly for grouped Vecchia likelihoods.

The compiled backends implement float64, fixed-coordinate hot paths for the
seven-parameter smoothness-0.5 Matérn model and six/seven-parameter
fixed-shape generalized-Cauchy models. Separate CPU and CUDA extensions can
be built; ``backend="auto"`` selects the one matching the input device. Every
other configuration uses the exact PyTorch formulation, so installing the
package without a compiler does not remove functionality. Analytic backward
is first-order only; callers requiring higher-order or coordinate derivatives
must select the Torch path.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

try:
    from GEMS_TCO import _vecchia_covariance_cpu
except (ImportError, OSError):  # Optional accelerator; Torch remains authoritative.
    _vecchia_covariance_cpu = None

try:
    from GEMS_TCO import _vecchia_covariance_cuda
except (ImportError, OSError):  # Optional accelerator; Torch remains authoritative.
    _vecchia_covariance_cuda = None


Backend = Literal["auto", "native", "torch"]


def _normalized_device_type(device: str | torch.device) -> str:
    try:
        return torch.device(device).type
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device {device!r}") from error


def _native_extension_for(device: str | torch.device):
    device_type = _normalized_device_type(device)
    if device_type == "cpu":
        return _vecchia_covariance_cpu
    if device_type == "cuda":
        return _vecchia_covariance_cuda
    return None


def native_covariance_available(device: str | torch.device | None = None) -> bool:
    """Return whether an optional native covariance backend is importable.

    With no argument this preserves the original CPU-availability API.
    Supplying ``cpu``, ``cuda``, or a concrete :class:`torch.device` checks
    that device explicitly.
    """

    if device is None:
        return _vecchia_covariance_cpu is not None
    return _native_extension_for(device) is not None


def native_generalized_cauchy_covariance_available(
    device: str | torch.device | None = None,
) -> bool:
    """Return whether the selected extension contains the fused GC kernel.

    This is intentionally stricter than :func:`native_covariance_available`:
    an older installed extension may provide the Matérn accelerator but
    predate the generalized-Cauchy entry points.  In that case callers must
    fall back to the authoritative Torch implementation rather than failing
    with a missing-symbol error.
    """

    extension = (
        _vecchia_covariance_cpu
        if device is None
        else _native_extension_for(device)
    )
    return (
        extension is not None
        and hasattr(extension, "generalized_cauchy_forward")
        and hasattr(extension, "generalized_cauchy_backward")
    )


def _normalized_dummy_mask(is_dummy: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    if not isinstance(is_dummy, torch.Tensor):
        raise TypeError("is_dummy must be a torch.Tensor")
    if is_dummy.ndim == 3 and is_dummy.shape[-1] == 1:
        is_dummy = is_dummy.squeeze(-1)
    if is_dummy.ndim != 2 or tuple(is_dummy.shape) != tuple(coordinates.shape[:2]):
        raise ValueError("is_dummy must have shape (batch, points) or (batch, points, 1)")
    if is_dummy.dtype != torch.bool:
        raise TypeError("is_dummy must have dtype torch.bool")
    if is_dummy.device != coordinates.device:
        raise ValueError("is_dummy and coordinates must be on the same device")
    return is_dummy


def _validate_arguments(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    smooth: float,
) -> torch.Tensor:
    if not isinstance(params, torch.Tensor):
        raise TypeError("params must be a torch.Tensor")
    if params.ndim != 1 or params.numel() != 7:
        raise ValueError("params must have shape (7,)")
    if not params.is_floating_point():
        raise TypeError("params must have a floating-point dtype")
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a torch.Tensor")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (batch, points, 3)")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must have a floating-point dtype")
    if coordinates.dtype != params.dtype or coordinates.device != params.device:
        raise ValueError("params and coordinates must have the same dtype and device")
    if smooth not in (0.5, 1.5):
        raise ValueError("smooth must be 0.5 or 1.5")
    return _normalized_dummy_mask(is_dummy, coordinates)


def torch_covariance_reference(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    smooth: float = 0.5,
) -> torch.Tensor:
    """Evaluate covariance plus padding decoupling using ordinary PyTorch ops."""

    dummy = _validate_arguments(params, coordinates, is_dummy, smooth)
    phi1, phi2, phi3, phi4 = torch.exp(params[:4]).unbind()
    nugget = torch.exp(params[6])
    time = coordinates[..., 2]
    advected_latitude = coordinates[..., 0] - params[4] * time
    advected_longitude = coordinates[..., 1] - params[5] * time
    latitude_difference = advected_latitude.unsqueeze(2) - advected_latitude.unsqueeze(1)
    longitude_difference = advected_longitude.unsqueeze(2) - advected_longitude.unsqueeze(1)
    time_difference = time.unsqueeze(2) - time.unsqueeze(1)
    squared_distance = (
        phi3 * latitude_difference.square()
        + longitude_difference.square()
        + phi4 * time_difference.square()
    )
    positive = squared_distance > 0
    distance = torch.where(
        positive,
        torch.sqrt(squared_distance.clamp_min(torch.finfo(squared_distance.dtype).eps)),
        torch.zeros_like(squared_distance),
    )
    scaled_distance = phi2 * distance
    if smooth == 0.5:
        correlation = torch.exp(-scaled_distance)
    else:
        sqrt_three_distance = 3.0**0.5 * scaled_distance
        correlation = (1.0 + sqrt_three_distance) * torch.exp(-sqrt_three_distance)
    covariance = (phi1 / phi2) * correlation
    identity = torch.eye(
        coordinates.shape[1],
        device=coordinates.device,
        dtype=coordinates.dtype,
    ).unsqueeze(0)
    covariance = covariance + identity * (nugget + 1.0e-6)

    touches_dummy = dummy.unsqueeze(2) | dummy.unsqueeze(1)
    covariance = covariance.masked_fill(touches_dummy, 0.0)
    return covariance + identity * dummy.to(coordinates.dtype).unsqueeze(1)


def _validate_generalized_cauchy_arguments(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    gc_alpha: float,
    gc_beta: float,
) -> torch.Tensor:
    if not isinstance(params, torch.Tensor):
        raise TypeError("params must be a torch.Tensor")
    if params.ndim != 1 or params.numel() not in (6, 7):
        raise ValueError("generalized-Cauchy params must have shape (6,) or (7,)")
    if not params.is_floating_point():
        raise TypeError("params must have a floating-point dtype")
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a torch.Tensor")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (batch, points, 3)")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must have a floating-point dtype")
    if coordinates.dtype != params.dtype or coordinates.device != params.device:
        raise ValueError("params and coordinates must have the same dtype and device")
    alpha = float(gc_alpha)
    beta = float(gc_beta)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 2.0:
        raise ValueError("gc_alpha must be finite and in (0, 2]")
    if not math.isfinite(beta) or not beta > 0.0:
        raise ValueError("gc_beta must be finite and positive")
    return _normalized_dummy_mask(is_dummy, coordinates)


def torch_generalized_cauchy_covariance_reference(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    gc_alpha: float,
    gc_beta: float,
) -> torch.Tensor:
    """Evaluate the exact GC covariance with ordinary differentiable ops."""

    dummy = _validate_generalized_cauchy_arguments(
        params,
        coordinates,
        is_dummy,
        gc_alpha,
        gc_beta,
    )
    phi1, phi2, phi3, phi4 = torch.exp(params[:4]).unbind()
    nugget = torch.exp(params[6]) if params.numel() == 7 else params.new_tensor(0.0)
    time = coordinates[..., 2]
    advected_latitude = coordinates[..., 0] - params[4] * time
    advected_longitude = coordinates[..., 1] - params[5] * time
    latitude_difference = advected_latitude.unsqueeze(2) - advected_latitude.unsqueeze(1)
    longitude_difference = advected_longitude.unsqueeze(2) - advected_longitude.unsqueeze(1)
    time_difference = time.unsqueeze(2) - time.unsqueeze(1)
    squared_distance = (
        phi3 * latitude_difference.square()
        + longitude_difference.square()
        + phi4 * time_difference.square()
    )
    positive = squared_distance > 0
    distance = torch.where(
        positive,
        torch.sqrt(squared_distance.clamp_min(torch.finfo(squared_distance.dtype).eps)),
        torch.zeros_like(squared_distance),
    )
    scaled_distance = phi2 * distance
    alpha = scaled_distance.new_tensor(float(gc_alpha))
    beta = scaled_distance.new_tensor(float(gc_beta))
    scaled = scaled_distance.clamp_min(torch.finfo(scaled_distance.dtype).eps)
    correlation = torch.pow(1.0 + torch.pow(scaled, alpha), -beta / alpha)
    correlation = torch.where(scaled_distance > 0, correlation, torch.ones_like(correlation))
    covariance = (phi1 / phi2) * correlation

    identity = torch.eye(
        coordinates.shape[1],
        device=coordinates.device,
        dtype=coordinates.dtype,
    ).unsqueeze(0)
    covariance = covariance + identity * (nugget + 1.0e-6)
    touches_dummy = dummy.unsqueeze(2) | dummy.unsqueeze(1)
    covariance = covariance.masked_fill(touches_dummy, 0.0)
    return covariance + identity * dummy.to(coordinates.dtype).unsqueeze(1)


class _NativeCovariance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        params: torch.Tensor,
        coordinates: torch.Tensor,
        dummy: torch.Tensor,
    ) -> torch.Tensor:
        contiguous_params = params.contiguous()
        contiguous_coordinates = coordinates.contiguous()
        contiguous_dummy = dummy.contiguous()
        native_extension = _native_extension_for(contiguous_params.device)
        if native_extension is None:  # Defensive: native_covariance checks first.
            raise RuntimeError(
                f"native covariance extension is unavailable for {contiguous_params.device.type}"
            )
        ctx.native_extension = native_extension
        ctx.save_for_backward(contiguous_params, contiguous_coordinates, contiguous_dummy)
        return native_extension.forward(
            contiguous_params,
            contiguous_coordinates,
            contiguous_dummy,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        params, coordinates, dummy = ctx.saved_tensors
        grad_params = ctx.native_extension.backward(
            grad_output.contiguous(),
            params,
            coordinates,
            dummy,
        )
        return grad_params, None, None


class _NativeGeneralizedCauchyCovariance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        params: torch.Tensor,
        coordinates: torch.Tensor,
        dummy: torch.Tensor,
        gc_alpha: float,
        gc_beta: float,
    ) -> torch.Tensor:
        contiguous_params = params.contiguous()
        contiguous_coordinates = coordinates.contiguous()
        contiguous_dummy = dummy.contiguous()
        native_extension = _native_extension_for(contiguous_params.device)
        if native_extension is None:  # Defensive: public wrapper checks first.
            raise RuntimeError(
                f"native covariance extension is unavailable for {contiguous_params.device.type}"
            )
        ctx.native_extension = native_extension
        ctx.gc_alpha = float(gc_alpha)
        ctx.gc_beta = float(gc_beta)
        ctx.save_for_backward(contiguous_params, contiguous_coordinates, contiguous_dummy)
        return native_extension.generalized_cauchy_forward(
            contiguous_params,
            contiguous_coordinates,
            contiguous_dummy,
            ctx.gc_alpha,
            ctx.gc_beta,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        params, coordinates, dummy = ctx.saved_tensors
        grad_params = ctx.native_extension.generalized_cauchy_backward(
            grad_output.contiguous(),
            params,
            coordinates,
            dummy,
            ctx.gc_alpha,
            ctx.gc_beta,
        )
        return grad_params, None, None, None, None


def native_covariance(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    smooth: float = 0.5,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Return covariance matrices with dummy rows replaced by identity rows.

    Parameters
    ----------
    backend
        ``"auto"`` selects the native kernel matching the input device for
        its supported float64, smooth=0.5 configuration.  ``"native"``
        requires that configuration and raises instead of silently falling
        back.  ``"torch"`` always uses the portable reference path.
    """

    dummy = _validate_arguments(params, coordinates, is_dummy, smooth)
    if backend not in ("auto", "native", "torch"):
        raise ValueError("backend must be 'auto', 'native', or 'torch'")

    native_supported = (
        _native_extension_for(params.device) is not None
        and smooth == 0.5
        and params.device.type in {"cpu", "cuda"}
        and params.dtype == torch.float64
        and not coordinates.requires_grad
    )
    if backend == "native" and not native_supported:
        raise RuntimeError(
            "native covariance requires the compiled extension, smooth=0.5, "
            "CPU or CUDA float64 inputs, and fixed coordinates"
        )
    if backend != "torch" and native_supported:
        return _NativeCovariance.apply(params, coordinates, dummy)
    return torch_covariance_reference(params, coordinates, dummy, smooth)


def native_generalized_cauchy_covariance(
    params: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
    gc_alpha: float,
    gc_beta: float,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Return a fused fixed-shape generalized-Cauchy covariance batch.

    Six parameters select the no-nugget model; seven parameters include the
    log-nugget as the final entry.  ``gc_alpha`` and ``gc_beta`` are fixed
    model-shape constants and therefore receive no gradients.
    """

    dummy = _validate_generalized_cauchy_arguments(
        params,
        coordinates,
        is_dummy,
        gc_alpha,
        gc_beta,
    )
    if backend not in ("auto", "native", "torch"):
        raise ValueError("backend must be 'auto', 'native', or 'torch'")

    native_supported = (
        native_generalized_cauchy_covariance_available(params.device)
        and params.device.type in {"cpu", "cuda"}
        and params.dtype == torch.float64
        and not coordinates.requires_grad
    )
    if backend == "native" and not native_supported:
        raise RuntimeError(
            "native generalized-Cauchy covariance requires the updated compiled "
            "extension, CPU or CUDA float64 inputs, and fixed coordinates"
        )
    if backend != "torch" and native_supported:
        return _NativeGeneralizedCauchyCovariance.apply(
            params,
            coordinates,
            dummy,
            float(gc_alpha),
            float(gc_beta),
        )
    return torch_generalized_cauchy_covariance_reference(
        params,
        coordinates,
        dummy,
        gc_alpha,
        gc_beta,
    )


__all__ = [
    "native_covariance",
    "native_covariance_available",
    "native_generalized_cauchy_covariance",
    "native_generalized_cauchy_covariance_available",
    "torch_generalized_cauchy_covariance_reference",
    "torch_covariance_reference",
]
