"""Matrix-free BTTB covariance products and Lanczos square-root actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.linalg import eigh_tridiagonal


Array = np.ndarray


@dataclass(frozen=True)
class BTTBOperator:
    """Finite stationary-grid covariance operator evaluated by FFT convolution."""

    grid_shape: tuple[int, int, int]
    convolution_shape: tuple[int, int, int]
    kernel_spectrum: Array

    @property
    def size(self) -> int:
        return int(np.prod(self.grid_shape))

    def matvec(self, vector: Array) -> Array:
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"expected vector length {self.size}, got {flat.size}")
        padded = np.zeros(self.convolution_shape, dtype=np.float64)
        slices = tuple(slice(0, size) for size in self.grid_shape)
        padded[slices] = flat.reshape(self.grid_shape)
        product = irfftn(
            rfftn(padded) * self.kernel_spectrum,
            s=self.convolution_shape,
        )
        return np.asarray(product[slices], dtype=np.float64).reshape(-1)


def build_bttb_operator(
    grid_shape: tuple[int, int, int],
    spacings: tuple[float, float, float],
    covariance: Callable[[Array, Array, Array], Array],
) -> BTTBOperator:
    """Build an exact zero-padded FFT matvec for a 3-D BTTB covariance."""

    if len(grid_shape) != 3 or len(spacings) != 3:
        raise ValueError("grid_shape and spacings must both have length three")
    if any(size < 1 for size in grid_shape):
        raise ValueError("grid dimensions must be positive")
    convolution_shape = tuple(next_fast_len(2 * size - 1) for size in grid_shape)
    lag_axes = []
    valid_axes = []
    for size, padded_size, spacing in zip(grid_shape, convolution_shape, spacings):
        index = np.arange(padded_size, dtype=np.int64)
        signed = np.where(index < size, index, index - padded_size)
        valid = np.abs(signed) < size
        lag_axes.append(signed.astype(np.float64) * float(spacing))
        valid_axes.append(valid)
    h0 = lag_axes[0][:, None, None]
    h1 = lag_axes[1][None, :, None]
    h2 = lag_axes[2][None, None, :]
    valid = (
        valid_axes[0][:, None, None]
        & valid_axes[1][None, :, None]
        & valid_axes[2][None, None, :]
    )
    kernel = np.zeros(convolution_shape, dtype=np.float64)
    values = np.asarray(covariance(h0, h1, h2), dtype=np.float64)
    kernel[valid] = np.broadcast_to(values, convolution_shape)[valid]
    spectrum = rfftn(kernel)
    return BTTBOperator(
        grid_shape=tuple(int(value) for value in grid_shape),
        convolution_shape=tuple(int(value) for value in convolution_shape),
        kernel_spectrum=spectrum,
    )


@dataclass(frozen=True)
class LanczosDecomposition:
    basis: Array
    diagonal: Array
    off_diagonal: Array
    input_norm: float
    breakdown: bool


def lanczos_decomposition(
    matvec: Callable[[Array], Array],
    vector: Array,
    max_dimension: int,
    breakdown_tolerance: float = 1.0e-13,
    full_reorthogonalization: bool = True,
) -> LanczosDecomposition:
    """Compute a symmetric Lanczos decomposition for one starting vector."""

    initial = np.asarray(vector, dtype=np.float64).reshape(-1)
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    norm = float(np.linalg.norm(initial))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("the starting vector must be finite and nonzero")
    basis = np.empty((max_dimension, initial.size), dtype=np.float64)
    diagonal = np.empty(max_dimension, dtype=np.float64)
    off_diagonal = np.empty(max(0, max_dimension - 1), dtype=np.float64)
    current = initial / norm
    previous = np.zeros_like(current)
    previous_beta = 0.0
    completed = 0
    breakdown = False
    for iteration in range(max_dimension):
        basis[iteration] = current
        work = np.asarray(matvec(current), dtype=np.float64).reshape(-1)
        if work.size != initial.size:
            raise ValueError("matvec returned the wrong vector length")
        if iteration > 0:
            work -= previous_beta * previous
        alpha = float(np.dot(current, work))
        work -= alpha * current
        if full_reorthogonalization:
            active = basis[: iteration + 1]
            # Two passes suppress loss of orthogonality for clustered spectra.
            work -= np.dot(active, work) @ active
            work -= np.dot(active, work) @ active
        beta = float(np.linalg.norm(work))
        diagonal[iteration] = alpha
        completed = iteration + 1
        scale = max(abs(alpha), abs(previous_beta), 1.0)
        if iteration == max_dimension - 1:
            break
        if beta <= breakdown_tolerance * scale:
            breakdown = True
            break
        off_diagonal[iteration] = beta
        previous = current
        current = work / beta
        previous_beta = beta
    return LanczosDecomposition(
        basis=basis[:completed].copy(),
        diagonal=diagonal[:completed].copy(),
        off_diagonal=off_diagonal[: max(0, completed - 1)].copy(),
        input_norm=norm,
        breakdown=breakdown,
    )


def sqrt_action(
    decomposition: LanczosDecomposition,
    dimension: int | None = None,
    negative_tolerance: float = 1.0e-10,
) -> tuple[Array, dict[str, float]]:
    """Evaluate ``A^(1/2)b`` from a leading Lanczos tridiagonal block."""

    available = len(decomposition.diagonal)
    if dimension is None:
        dimension = available
    if dimension < 1 or dimension > available:
        raise ValueError(f"dimension must lie in [1, {available}]")
    eigenvalues, eigenvectors = eigh_tridiagonal(
        decomposition.diagonal[:dimension],
        decomposition.off_diagonal[: max(0, dimension - 1)],
        check_finite=False,
    )
    maximum = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum = float(eigenvalues.min())
    if minimum < -negative_tolerance * maximum:
        raise ArithmeticError(
            f"Lanczos tridiagonal has materially negative eigenvalue {minimum:.6g}"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    first_projection = eigenvectors[0, :]
    coefficients = eigenvectors @ (np.sqrt(clipped) * first_projection)
    result = decomposition.input_norm * (
        coefficients @ decomposition.basis[:dimension]
    )
    diagnostics = {
        "dimension": int(dimension),
        "ritz_min": minimum,
        "ritz_max": float(eigenvalues.max()),
        "ritz_negative_count": int((eigenvalues < 0.0).sum()),
    }
    return np.asarray(result, dtype=np.float64), diagnostics


__all__ = [
    "BTTBOperator",
    "LanczosDecomposition",
    "build_bttb_operator",
    "lanczos_decomposition",
    "sqrt_action",
]
