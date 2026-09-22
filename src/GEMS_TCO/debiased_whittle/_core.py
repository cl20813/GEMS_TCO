"""Shared numerical core for configured Debiased Whittle estimators.

This module contains the shared tapering, Fourier, covariance, likelihood,
linear-algebra, and optimizer implementation.  Filter-only differences are
selected through an immutable ``SpatialFilterSpec``.

It is private by design.  Use :mod:`GEMS_TCO.debiased_whittle` from application
code.
"""

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .filters import SUMMED_FIRST_DIFFERENCES


@dataclass(frozen=True)
class DebiasedWhittleFitResult:
    """Best finite parameter state observed during optimization."""

    parameters: torch.Tensor
    loss: float
    steps: int
    converged: bool


def _spectral_likelihood_terms(
    expected_periodogram: torch.Tensor,
    sample_periodogram: torch.Tensor,
) -> torch.Tensor:
    """Return stable per-frequency Gaussian spectral likelihood terms.

    Non-positive-definite or non-finite frequency blocks are represented by
    ``inf``.  Callers may then exclude structurally unidentifiable frequencies
    without allowing those blocks to contaminate retained frequencies.
    """
    if (
        expected_periodogram.ndim < 2
        or expected_periodogram.shape[-1] != expected_periodogram.shape[-2]
    ):
        raise ValueError("expected_periodogram must end in square matrix dimensions.")
    if sample_periodogram.shape != expected_periodogram.shape:
        raise ValueError("sample_periodogram and expected_periodogram must have identical shapes.")

    device = expected_periodogram.device
    matrix_size = expected_periodogram.shape[-1]
    expected = expected_periodogram.to(dtype=torch.complex128)
    sample = sample_periodogram.to(device=device, dtype=torch.complex128)
    eye = torch.eye(matrix_size, dtype=torch.complex128, device=device)
    diagonal = torch.abs(expected.diagonal(dim1=-2, dim2=-1))
    finite_diagonal = diagonal[torch.isfinite(diagonal)]
    if finite_diagonal.numel() == 0:
        return torch.full(expected.shape[:-2], float("inf"), device=device, dtype=torch.float64)
    diagonal_scale = finite_diagonal.mean()
    diagonal_load = torch.clamp(diagonal_scale * 1e-8, min=1e-9)
    stable = expected + eye * diagonal_load

    cholesky_factor, cholesky_info = torch.linalg.cholesky_ex(stable, check_errors=False)
    try:
        log_abs_determinant = 2.0 * torch.log(cholesky_factor.diagonal(dim1=-2, dim2=-1).real).sum(
            dim=-1
        )
        solved = torch.cholesky_solve(sample, cholesky_factor)
        trace = torch.einsum("...ii->...", solved).real
    except torch.linalg.LinAlgError:
        return torch.full(expected.shape[:-2], float("inf"), device=device, dtype=torch.float64)

    terms = log_abs_determinant.real + trace
    expected_is_hermitian = torch.isclose(
        expected,
        expected.conj().transpose(-1, -2),
        rtol=1e-7,
        atol=1e-10,
    ).all(dim=(-2, -1))
    sample_is_hermitian = torch.isclose(
        sample,
        sample.conj().transpose(-1, -2),
        rtol=1e-7,
        atol=1e-10,
    ).all(dim=(-2, -1))
    valid = (
        (cholesky_info == 0) & expected_is_hermitian & sample_is_hermitian & torch.isfinite(terms)
    )
    return torch.where(valid, terms, torch.full_like(terms, float("inf")))


def _optimize_parameters(
    parameters: Sequence[torch.Tensor],
    optimizer,
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_steps: int,
    gradient_tolerance: float,
    loss_tolerance: float = 1e-12,
) -> DebiasedWhittleFitResult:
    """Run a closure-based optimizer and retain a loss-consistent best state."""
    parameters = tuple(parameters)
    if max_steps < 1:
        raise ValueError("max_steps must be at least one.")
    if gradient_tolerance < 0 or loss_tolerance < 0:
        raise ValueError("Optimization tolerances must be non-negative.")
    if not parameters:
        raise ValueError("At least one parameter tensor is required.")

    if not all(isinstance(parameter, torch.Tensor) for parameter in parameters):
        raise TypeError("parameters must contain only torch.Tensor objects.")
    if not all(parameter.is_floating_point() for parameter in parameters):
        raise TypeError("Parameter tensors must have a floating-point dtype.")
    trainable_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable_parameters:
        raise ValueError("At least one parameter tensor must have requires_grad=True.")
    parameter_devices = {parameter.device for parameter in parameters}
    if len(parameter_devices) != 1:
        raise ValueError("All parameter tensors must be on the same device.")
    if len({parameter.dtype for parameter in parameters}) != 1:
        raise ValueError("All parameter tensors must have the same dtype.")

    initial_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    best_parameters = initial_parameters.clone()
    best_loss = float("inf")
    previous_loss = float("inf")
    last_loss = float("inf")
    last_gradient = float("inf")
    converged = False

    def closure():
        nonlocal best_loss, best_parameters, last_loss, last_gradient
        optimizer.zero_grad()
        parameter_tensor = torch.cat([parameter.reshape(-1) for parameter in parameters])
        loss = objective(parameter_tensor)
        if loss.ndim != 0:
            raise ValueError("The optimization objective must return a scalar tensor.")
        if not torch.isfinite(loss):
            last_loss = float("inf")
            last_gradient = float("inf")
            return loss

        loss.backward()
        gradients = [parameter.grad for parameter in trainable_parameters]
        if any(gradient is None for gradient in gradients) or any(
            not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
        ):
            optimizer.zero_grad()
            last_loss = float("inf")
            last_gradient = float("inf")
            return loss

        last_loss = float(loss.detach())
        last_gradient = max(
            float(gradient.detach().abs().max()) for gradient in gradients if gradient is not None
        )
        if last_loss < best_loss:
            best_loss = last_loss
            best_parameters = torch.cat(
                [parameter.detach().reshape(-1) for parameter in parameters]
            ).clone()
        return loss

    steps_completed = 0
    for step in range(max_steps):
        optimizer.step(closure)
        # Closure-based optimizers may return the loss evaluated before their
        # final parameter update.  Evaluate once at the actual post-step state
        # so every recorded loss is paired with the state that produced it.
        closure()
        steps_completed = step + 1
        if not math.isfinite(last_loss):
            break
        if last_gradient <= gradient_tolerance:
            converged = True
            break
        if step > 0 and abs(last_loss - previous_loss) <= loss_tolerance:
            converged = True
            break
        previous_loss = last_loss

    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            length = parameter.numel()
            parameter.copy_(best_parameters[offset : offset + length].reshape_as(parameter))
            offset += length

    if not math.isfinite(best_loss):
        raise RuntimeError(
            "Debiased Whittle optimization produced no finite state with finite gradients; "
            "the initial parameters and every evaluated candidate were invalid"
        )

    return DebiasedWhittleFitResult(
        parameters=best_parameters,
        loss=best_loss,
        steps=steps_completed,
        converged=converged,
    )


class BaseDebiasedWhittlePreprocessor:
    """Apply one configured spatial filter to ordered temporal grid slices.

    Parameters
    ----------
    time_slices:
        Chronologically ordered tensors with columns
        ``(latitude, longitude, value, time)``.  Every non-empty tensor must
        contain one complete, duplicate-free rectangular spatial grid and a
        single time value.
    """

    filter_spec = SUMMED_FIRST_DIFFERENCES

    def __init__(self, time_slices):
        self.time_slices = tuple(time_slices)
        if not all(isinstance(tensor, torch.Tensor) for tensor in self.time_slices):
            raise TypeError("time_slices must contain only torch.Tensor objects.")
        if len({tensor.device for tensor in self.time_slices}) > 1:
            raise ValueError("All time slices must be on the same device.")
        self._validate_unit_time_axis()

    def _validate_unit_time_axis(self) -> None:
        """Require complete, chronological slices on the modeled unit-time grid.

        The expected-periodogram implementation uses integer index differences
        for temporal lags.  Accepting empty, repeated, or gapped slices would
        silently compress physical time and change the fitted temporal range
        and advection units.
        """

        if not self.time_slices:
            return
        time_values = []
        for time_index, grid in enumerate(self.time_slices):
            if grid.ndim != 2 or grid.shape[1] < 4:
                raise ValueError(
                    "Each grid must be a two-dimensional tensor with at least four columns."
                )
            if grid.shape[0] == 0:
                raise ValueError(
                    f"time-slice index {time_index} is empty; Debiased Whittle "
                    "preprocessing requires every modeled time slot"
                )
            times = grid[:, 3]
            if not torch.isfinite(times).all() or not torch.all(times == times[0]):
                raise ValueError(
                    f"time-slice index {time_index} must contain one finite time value"
                )
            time_values.append(times[0].to(dtype=torch.float64))

        if len(time_values) < 2:
            return
        time_axis = torch.stack(time_values)
        increments = torch.diff(time_axis)
        if not torch.allclose(
            increments,
            torch.ones_like(increments),
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError(
                "time_slices must be chronological and consecutive with unit spacing; "
                "the likelihood models temporal lags by slice index"
            )

    @staticmethod
    def _subset_grid(
        grid: torch.Tensor,
        latitude_start: float,
        latitude_end: float,
        longitude_start: float,
        longitude_end: float,
    ) -> torch.Tensor:
        """Return rows inside inclusive latitude/longitude bounds."""
        bounds = (latitude_start, latitude_end, longitude_start, longitude_end)
        if not all(math.isfinite(float(bound)) for bound in bounds):
            raise ValueError("Spatial subset bounds must be finite.")
        if latitude_start > latitude_end or longitude_start > longitude_end:
            raise ValueError("Spatial subset starts must not exceed their ends.")
        if grid.ndim != 2 or grid.shape[1] < 4:
            raise ValueError(
                "Each grid must be a two-dimensional tensor with at least four columns."
            )
        latitude_mask = (grid[:, 0] >= latitude_start) & (grid[:, 0] <= latitude_end)
        longitude_mask = (grid[:, 1] >= longitude_start) & (grid[:, 1] <= longitude_end)
        return grid[latitude_mask & longitude_mask].clone()

    @staticmethod
    def _validated_grid(
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate a complete grid and return data, coordinates, and row order."""
        if grid.ndim != 2 or grid.shape[1] < 4:
            raise ValueError(
                "Each grid must be a two-dimensional tensor with at least four columns."
            )
        if grid.shape[0] == 0:
            raise ValueError("A spatial filter cannot be applied to an empty grid.")

        grid = grid.to(dtype=torch.float64)
        if not torch.isfinite(grid[:, :2]).all():
            raise ValueError("Spatial coordinates must be finite.")
        if torch.isinf(grid[:, 2]).any():
            raise ValueError(
                "Grid values may be NaN for missing observations but cannot be infinite."
            )
        if not torch.isfinite(grid[:, 3]).all() or not torch.all(grid[:, 3] == grid[0, 3]):
            raise ValueError("Every row in one grid must have the same finite time value.")

        unique_lats = torch.unique(grid[:, 0])
        unique_lons = torch.unique(grid[:, 1])
        lat_count, lon_count = unique_lats.size(0), unique_lons.size(0)

        for axis_name, coordinates in (
            ("latitude", unique_lats),
            ("longitude", unique_lons),
        ):
            if coordinates.numel() > 2:
                increments = torch.diff(coordinates)
                if not torch.allclose(
                    increments,
                    increments[0].expand_as(increments),
                    rtol=1e-5,
                    atol=1e-10,
                ):
                    raise ValueError(f"The {axis_name} coordinates must form a regular grid.")

        latitude_index = torch.searchsorted(unique_lats, grid[:, 0].contiguous())
        longitude_index = torch.searchsorted(unique_lons, grid[:, 1].contiguous())
        linear_index = latitude_index * lon_count + longitude_index
        if grid.size(0) != lat_count * lon_count or torch.unique(linear_index).numel() != grid.size(
            0
        ):
            raise ValueError(
                "Each latitude/longitude pair must occur exactly once on a complete grid."
            )

        return grid, unique_lats, unique_lons, torch.argsort(linear_index)

    def apply_filter(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply the configured stencil and grid reduction to one time slice."""
        if grid.ndim != 2 or grid.shape[1] < 4:
            raise ValueError(
                "Each grid must be a two-dimensional tensor with at least four columns."
            )
        if grid.size(0) == 0:
            return grid.new_empty((0, 4), dtype=torch.float64)

        grid, unique_lats, unique_lons, row_order = self._validated_grid(grid)
        lat_count, lon_count = unique_lats.size(0), unique_lons.size(0)

        spec = self.filter_spec
        if spec.demean_input:
            values = grid[:, 2]
            valid_mask = torch.isfinite(values)
            output = grid.clone()
            if valid_mask.sum() > 0:
                output[valid_mask, 2] = values[valid_mask] - values[valid_mask].mean()
            return output

        shrink_lat, shrink_lon = spec.grid_reduction
        if lat_count < shrink_lat + 1 or lon_count < shrink_lon + 1:
            return grid.new_empty((0, 4), dtype=torch.float64)

        grid = grid[row_order]
        observed_values = grid[:, 2].reshape(1, 1, lat_count, lon_count)
        kernel_lat, kernel_lon = spec.kernel_shape
        diff_kernel = grid.new_zeros((1, 1, kernel_lat, kernel_lon), dtype=torch.float64)
        for (lat_offset, lon_offset), weight in spec.weights:
            diff_kernel[0, 0, lat_offset, lon_offset] = weight

        filtered_grid = F.conv2d(observed_values, diff_kernel, padding="valid")

        new_lats = unique_lats[: lat_count - shrink_lat]
        new_lons = unique_lons[: lon_count - shrink_lon]

        new_lat_grid, new_lon_grid = torch.meshgrid(new_lats, new_lons, indexing="ij")
        filtered_values = filtered_grid.flatten()
        time_value = grid[0, 3].repeat(filtered_values.size(0))

        return torch.stack(
            [new_lat_grid.flatten(), new_lon_grid.flatten(), filtered_values, time_value], dim=1
        )

    def generate_filtered_data(
        self,
        latitude_start: float,
        latitude_end: float,
        longitude_start: float,
        longitude_end: float,
    ) -> torch.Tensor:
        """Filter every time slice and concatenate them in chronological order."""
        filtered_slices = []
        for time_index, grid in enumerate(self.time_slices):
            subset = self._subset_grid(
                grid, latitude_start, latitude_end, longitude_start, longitude_end
            )
            if subset.size(0) == 0:
                raise ValueError(
                    f"Spatial bounds leave time-slice index {time_index} empty; "
                    "all modeled time slots must share the selected grid."
                )
            try:
                filtered = self.apply_filter(subset)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid spatial grid at time-slice index {time_index}: {exc}"
                ) from exc
            if filtered.size(0) > 0:
                filtered_slices.append(filtered)
            else:
                raise ValueError(
                    f"The spatial filter leaves time-slice index {time_index} empty; "
                    "increase the selected grid extent."
                )

        if filtered_slices:
            return torch.cat(filtered_slices, dim=0)
        if self.time_slices:
            return self.time_slices[0].new_empty((0, 4), dtype=torch.float64)
        return torch.empty((0, 4), dtype=torch.float64)


class BaseDebiasedWhittleLikelihood:
    """Common numerical DW likelihood configured by ``filter_spec``."""

    filter_spec = SUMMED_FIRST_DIFFERENCES

    @staticmethod
    def _regular_axis(values: torch.Tensor, axis_name: str) -> torch.Tensor:
        """Validate and return the sorted unique coordinates of one grid axis."""
        if not torch.isfinite(values).all():
            raise ValueError(f"{axis_name} coordinates must be finite.")
        coordinates = torch.unique(values)
        if coordinates.numel() > 2:
            increments = torch.diff(coordinates)
            if not torch.allclose(
                increments,
                increments[0].expand_as(increments),
                rtol=1e-5,
                atol=1e-10,
            ):
                raise ValueError(f"{axis_name} coordinates must form a regular grid.")
        return coordinates

    # =========================================================================
    # 1. Tapering & Data Functions
    # =========================================================================
    @staticmethod
    def hamming_taper(u, n1, n2):
        """Evaluate the historical periodic Hamming taper on grid indices.

        The denominator is the grid size (not ``n - 1``), so this is the
        periodic convention used by the established numerical results.
        """
        if n1 < 1 or n2 < 1:
            raise ValueError("Taper dimensions must be positive.")
        u1, u2 = u
        device = (
            u1.device
            if isinstance(u1, torch.Tensor)
            else (u2.device if isinstance(u2, torch.Tensor) else torch.device("cpu"))
        )
        u1_tensor = (
            u1.to(device)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=device, dtype=torch.float64)
        )
        u2_tensor = (
            u2.to(device)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=device, dtype=torch.float64)
        )
        hamming1 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u1_tensor / float(n1))
        hamming2 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u2_tensor / float(n2))
        return hamming1 * hamming2

    @staticmethod
    def taper_autocorrelation(taper_grid, n1, n2, device=None):
        """Compute the normalized taper autocorrelation with zero padding."""
        if n1 < 1 or n2 < 1:
            raise ValueError("Taper dimensions must be positive.")
        if taper_grid.shape != (n1, n2):
            raise ValueError(f"taper_grid must have shape {(n1, n2)}.")
        if not torch.isfinite(taper_grid).all():
            raise ValueError("taper_grid must contain only finite values.")
        target_device = taper_grid.device if device is None else torch.device(device)
        taper_grid = taper_grid.to(target_device)
        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            raise ValueError("The taper has near-zero squared norm.")
        N1, N2 = 2 * n1 - 1, 2 * n2 - 1
        taper_fft = torch.fft.fft2(taper_grid, s=(N1, N2))
        power_spectrum = torch.abs(taper_fft) ** 2
        autocorr_unnormalized = torch.fft.ifft2(power_spectrum).real
        autocorr_shifted = torch.fft.fftshift(autocorr_unnormalized)
        c_gn_grid = autocorr_shifted / H
        return c_gn_grid

    @staticmethod
    def _fill_grid_from_tensor(
        tensor, unique_lats, unique_lons, lat_col, lon_col, val_col, n1, n2, device
    ):
        data_grid = torch.zeros((n1, n2), dtype=torch.float64, device=device)
        obs_mask = torch.zeros((n1, n2), dtype=torch.bool, device=device)

        if tensor.ndim != 2 or tensor.shape[1] <= max(lat_col, lon_col, val_col):
            raise ValueError("A time slice does not contain the requested columns.")
        if tensor.numel() == 0:
            return data_grid, obs_mask

        coordinates = tensor[:, [lat_col, lon_col]]
        if not torch.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates must be finite.")
        lat_vals = coordinates[:, 0]
        lon_vals = coordinates[:, 1]

        unique_lats_dev = unique_lats.to(lat_vals.device)
        unique_lons_dev = unique_lons.to(lon_vals.device)
        lat_idx = torch.searchsorted(unique_lats_dev, lat_vals.contiguous())
        lon_idx = torch.searchsorted(unique_lons_dev, lon_vals.contiguous())

        in_bounds = (lat_idx >= 0) & (lat_idx < n1) & (lon_idx >= 0) & (lon_idx < n2)
        if not in_bounds.all():
            raise ValueError("A spatial coordinate lies outside the shared grid.")

        matched = (unique_lats_dev[lat_idx] == lat_vals) & (unique_lons_dev[lon_idx] == lon_vals)
        if not matched.all():
            raise ValueError("A spatial coordinate is not present in the shared grid.")

        linear_idx = lat_idx * n2 + lon_idx
        if torch.unique(linear_idx).numel() != linear_idx.numel():
            raise ValueError("Every time slice must contain at most one row per grid location.")

        values = tensor[:, val_col]
        if torch.isinf(values).any():
            raise ValueError("Observed values may be NaN for missing data but cannot be infinite.")
        finite_values = torch.isfinite(values)
        if not finite_values.any():
            return data_grid, obs_mask

        lat_idx = lat_idx[finite_values].to(device=device, dtype=torch.long)
        lon_idx = lon_idx[finite_values].to(device=device, dtype=torch.long)
        val_vals = values[finite_values].to(device=device, dtype=torch.float64)
        data_grid[lat_idx, lon_idx] = val_vals
        obs_mask[lat_idx, lon_idx] = True
        return data_grid, obs_mask

    @classmethod
    def tapered_fourier_transform(
        cls, tensor_list, tapering_func, lat_col, lon_col, val_col, device
    ):
        """Return the normalized tapered DFT for complete scalar grids."""
        p_time = len(tensor_list)
        if p_time == 0:
            raise ValueError("tensor_list must contain at least one time slice.")

        required_column = max(lat_col, lon_col, val_col)
        if any(
            tensor.ndim != 2 or tensor.shape[1] <= required_column or tensor.shape[0] == 0
            for tensor in tensor_list
        ):
            raise ValueError(
                "Every time slice must be a non-empty 2-D tensor with the requested columns."
            )
        unique_lats = cls._regular_axis(
            torch.cat([tensor[:, lat_col] for tensor in tensor_list]), "Latitude"
        )
        unique_lons = cls._regular_axis(
            torch.cat([tensor[:, lon_col] for tensor in tensor_list]), "Longitude"
        )
        n1, n2 = len(unique_lats), len(unique_lons)
        target_device = torch.device(device)
        u1_mesh, u2_mesh = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64, device=target_device),
            torch.arange(n2, dtype=torch.float64, device=target_device),
            indexing="ij",
        )
        taper_grid = tapering_func((u1_mesh, u2_mesh), n1, n2).to(
            device=target_device, dtype=torch.float64
        )
        if taper_grid.shape != (n1, n2) or not torch.isfinite(taper_grid).all():
            raise ValueError("tapering_func must return a finite (n1, n2) tensor.")

        fft_results = []
        for time_index, tensor in enumerate(tensor_list):
            data_grid, observation_mask = cls._fill_grid_from_tensor(
                tensor,
                unique_lats,
                unique_lons,
                lat_col,
                lon_col,
                val_col,
                n1,
                n2,
                target_device,
            )
            if not observation_mask.all():
                raise ValueError(
                    f"Scalar time slice {time_index} does not contain one finite observation "
                    "at every grid location; use masked_tapered_fourier_transform for missing data."
                )
            data_grid_tapered = data_grid * taper_grid
            fft_results.append(torch.fft.fft2(data_grid_tapered))

        J_vector_tensor = torch.stack(fft_results, dim=2).to(target_device)

        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            raise ValueError("The taper has near-zero squared norm.")
        norm_factor = (torch.sqrt(1.0 / H) / (2.0 * cmath.pi)).to(target_device)

        result = J_vector_tensor * norm_factor
        return result, n1, n2, p_time, taper_grid

    @classmethod
    def masked_tapered_fourier_transform(
        cls, tensor_list, tapering_func, lat_col, lon_col, val_col, device
    ):
        """
        Same as :meth:`tapered_fourier_transform`, allowing missing cells and
        also returning observation masks with shape ``(p_time, n1, n2)``.
        obs_masks[q, i, j] = True if cell (i,j) was observed at time q.
        Used for the multivariate-corrected c_{g,n}^{(qr)} computation.
        """
        p_time = len(tensor_list)
        if p_time == 0:
            raise ValueError("tensor_list must contain at least one time slice.")

        required_column = max(lat_col, lon_col, val_col)
        if any(
            tensor.ndim != 2 or tensor.shape[1] <= required_column or tensor.shape[0] == 0
            for tensor in tensor_list
        ):
            raise ValueError(
                "Every time slice must be a non-empty 2-D tensor with the requested columns."
            )
        unique_lats = cls._regular_axis(
            torch.cat([tensor[:, lat_col] for tensor in tensor_list]), "Latitude"
        )
        unique_lons = cls._regular_axis(
            torch.cat([tensor[:, lon_col] for tensor in tensor_list]), "Longitude"
        )
        n1, n2 = len(unique_lats), len(unique_lons)
        target_device = torch.device(device)
        u1_mesh, u2_mesh = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64, device=target_device),
            torch.arange(n2, dtype=torch.float64, device=target_device),
            indexing="ij",
        )
        taper_grid = tapering_func((u1_mesh, u2_mesh), n1, n2).to(
            device=target_device, dtype=torch.float64
        )
        if taper_grid.shape != (n1, n2) or not torch.isfinite(taper_grid).all():
            raise ValueError("tapering_func must return a finite (n1, n2) tensor.")

        fft_results = []
        obs_masks_list = []
        for tensor in tensor_list:
            data_grid, obs_mask = cls._fill_grid_from_tensor(
                tensor,
                unique_lats,
                unique_lons,
                lat_col,
                lon_col,
                val_col,
                n1,
                n2,
                target_device,
            )

            data_grid_tapered = data_grid * taper_grid
            fft_results.append(torch.fft.fft2(data_grid_tapered))
            obs_masks_list.append(obs_mask)

        obs_masks = torch.stack(obs_masks_list, dim=0)  # (p_time, n1, n2)

        # Per-variate normalization: J^{(q)} uses H_q = Σ_s(taper_s * obs_s^{(q)})²
        # This is consistent with c_{g,n}^{(qr)} / sqrt(H_q * H_r) in the multivariate
        # expected periodogram (Guillaumin et al. 2022, Sec. 4.3.4).
        normed = []
        for q_idx, (fft_q, obs_q) in enumerate(zip(fft_results, obs_masks_list)):
            H_q = (taper_grid * obs_q.to(taper_grid.dtype)).pow(2).sum()
            if H_q < 1e-12:
                raise ValueError(
                    f"Time slice {q_idx} has no observations with positive taper weight."
                )
            norm_q = (torch.sqrt(1.0 / H_q) / (2.0 * cmath.pi)).to(target_device)
            normed.append(fft_q * norm_q)
        result = torch.stack(normed, dim=2).to(target_device)
        return result, n1, n2, p_time, taper_grid, obs_masks

    @staticmethod
    def multivariate_taper_autocorrelation(taper_grid, obs_masks, n1, n2, device=None):
        """
        Computes c_{g,n}^{(qr)} for every pair (q, r) per Section 4.3.4 of
        Guillaumin et al. (2022, JRSS-B).

        g_s^{(q)} = taper_s * obs_s^{(q)}   (paper Eq. incorporating missing data)

        Args:
            taper_grid : (n1, n2) float64 taper weights
            obs_masks  : (p, n1, n2) bool — True = observed at time q
            n1, n2     : grid dimensions
            device     : torch device

        Returns:
            (p, p, 2*n1-1, 2*n2-1) float64 tensor — c_{g,n}^{(qr)} for each pair
        """
        if n1 < 1 or n2 < 1:
            raise ValueError("Taper dimensions must be positive.")
        if taper_grid.shape != (n1, n2):
            raise ValueError(f"taper_grid must have shape {(n1, n2)}.")
        if obs_masks.ndim != 3 or obs_masks.shape[1:] != (n1, n2):
            raise ValueError(f"obs_masks must have shape (p_time, {n1}, {n2}).")
        if obs_masks.dtype != torch.bool:
            raise TypeError("obs_masks must have boolean dtype.")
        if not torch.isfinite(taper_grid).all():
            raise ValueError("taper_grid must contain only finite values.")
        target_device = taper_grid.device if device is None else torch.device(device)
        taper_grid = taper_grid.to(target_device)
        obs_masks = obs_masks.to(target_device, dtype=torch.bool)
        p = obs_masks.shape[0]
        N1, N2 = 2 * n1 - 1, 2 * n2 - 1

        # g^{(q)} = taper * obs_mask^{(q)},  shape (p, n1, n2)
        g_all = taper_grid.unsqueeze(0) * obs_masks.to(taper_grid.dtype)
        H_all = (g_all**2).sum(dim=(1, 2))  # (p,)
        if torch.any(H_all < 1e-12):
            raise ValueError("Every component must have observations with positive taper weight.")

        # FFT of each g^{(q)} padded to (N1, N2)
        g_ffts = torch.fft.fft2(g_all, s=(N1, N2))  # (p, N1, N2) complex

        result = torch.zeros((p, p, N1, N2), device=target_device, dtype=taper_grid.dtype)
        for q in range(p):
            for r in range(p):
                # Cross-correlation via FFT: IFFT(G^{(q)} * conj(G^{(r)}))
                cross = torch.fft.ifft2(g_ffts[q] * g_ffts[r].conj()).real
                cross_shifted = torch.fft.fftshift(cross)
                denom = torch.sqrt(H_all[q] * H_all[r])
                result[q, r] = cross_shifted / denom

        return result  # (p, p, 2n1-1, 2n2-1)

    @staticmethod
    def sample_periodogram(fourier_transform):
        """Return ``J Jᴴ`` at each spatial frequency."""
        if fourier_transform.ndim != 3:
            raise ValueError("fourier_transform must have shape (n1, n2, p_time).")
        if not torch.isfinite(fourier_transform).all():
            raise ValueError("fourier_transform must contain only finite values.")

        J_col = fourier_transform.unsqueeze(-1)
        J_row_conj = fourier_transform.unsqueeze(-2).conj()
        return J_col @ J_row_conj

    # =========================================================================
    # 2. Covariance Functions (7-Parameter Version)
    # =========================================================================
    @staticmethod
    def spatiotemporal_covariance(u1, u2, t, params):
        """Evaluate the seven-parameter advected exponential covariance."""
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        device = params.device
        u1_dev = (
            u1.to(device)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=device, dtype=torch.float64)
        )
        u2_dev = (
            u2.to(device)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=device, dtype=torch.float64)
        )
        t_dev = (
            t.to(device)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=device, dtype=torch.float64)
        )

        if torch.isnan(params).any() or torch.isinf(params).any():
            out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
            return torch.full(out_shape, float("nan"), device=device, dtype=torch.float64)

        # --- A. Unpack and Recover Parameters ---
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])  # range_lon_inv
        phi3 = torch.exp(params[2])  # (range_lon / range_lat)^2
        phi4 = torch.exp(params[3])  # beta^2
        advec_lat = params[4]
        advec_lon = params[5]
        nugget = torch.exp(params[6])

        epsilon = 1e-12
        sigmasq = phi1 / phi2
        range_lon_inv = phi2
        range_lat_inv = torch.sqrt(phi3) * phi2
        beta_scaled_inv = torch.sqrt(phi4) * phi2  # This is beta * range_lon_inv

        # --- B. Calculate Anisotropic Advected Distance ---
        u1_adv = u1_dev - advec_lat * t_dev
        u2_adv = u2_dev - advec_lon * t_dev

        dist_sq = (
            (u1_adv * range_lat_inv).pow(2)
            + (u2_adv * range_lon_inv).pow(2)
            + (t_dev * beta_scaled_inv).pow(2)
        )

        # Keep the small stabilizer away from zero, but preserve the exact
        # zero-lag variance and its well-defined parameter derivatives.
        distance = torch.where(
            dist_sq == 0,
            torch.zeros_like(dist_sq),
            torch.sqrt(dist_sq + epsilon),
        )

        # --- C. Calculate Covariance (Matern 0.5 = Exponential) ---
        cov_smooth = sigmasq * torch.exp(-distance)

        # --- D. Add Nugget ---
        is_zero_lag = (u1_dev == 0) & (u2_dev == 0) & (t_dev == 0)
        final_cov = torch.where(is_zero_lag, cov_smooth + nugget, cov_smooth)

        return final_cov

    @classmethod
    def filtered_covariance(cls, u1, u2, t, params, delta1, delta2):
        """
        Calculates covariance Cov(Y(s,t_q), Y(s+u,t_r))
        where Y is the spatially differenced field.
        u1, u2 are PHYSICAL lags. t is the PHYSICAL time lag.
        """
        device = params.device
        u1_dev = (
            u1.to(device)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=device, dtype=torch.float64)
        )
        u2_dev = (
            u2.to(device)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=device, dtype=torch.float64)
        )
        t_dev = (
            t.to(device)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=device, dtype=torch.float64)
        )

        if cls.filter_spec.demean_input:
            # The identity filter is exactly the latent-field covariance.
            return cls.spatiotemporal_covariance(u1_dev, u2_dev, t_dev, params)

        out_shape = torch.broadcast_shapes(
            u1.shape if isinstance(u1, torch.Tensor) else (),
            u2.shape if isinstance(u2, torch.Tensor) else (),
            t.shape if isinstance(t, torch.Tensor) else (),
        )
        cov = torch.zeros(out_shape, device=device, dtype=torch.float64)

        # Deterministic tuple order keeps floating-point accumulation reproducible.
        weights = cls.filter_spec.weights
        for (a_idx, b_idx), w_ab in weights:
            offset_a1 = a_idx * delta1
            offset_a2 = b_idx * delta2
            for (c_idx, d_idx), w_cd in weights:
                offset_c1 = c_idx * delta1
                offset_c2 = d_idx * delta2

                lag_u1 = u1_dev + (offset_a1 - offset_c1)
                lag_u2 = u2_dev + (offset_a2 - offset_c2)

                term_cov = cls.spatiotemporal_covariance(lag_u1, lag_u2, t_dev, params)

                if torch.isnan(term_cov).any():
                    return torch.full_like(cov, float("nan"))
                cov += w_ab * w_cd * term_cov

        return cov

    @classmethod
    def _tapered_covariance(
        cls, u1, u2, t, params, n1, n2, taper_autocorr_grid, delta1, delta2, q_idx=None, r_idx=None
    ):
        """
        Computes c_Y(u) * c_gn(u).
        u1, u2 are GRID index lags (e.g., -n1..0..n1)
        t is the PHYSICAL time lag.

        If taper_autocorr_grid is 4-D (p, p, 2n1-1, 2n2-1) — multivariate per-pair
        correction per Guillaumin et al. (2022) Section 4.3.4 — pass q_idx and r_idx
        to select the corresponding c_{g,n}^{(qr)} slice.
        Otherwise falls back to the 2-D single-taper behaviour.
        """
        device = params.device
        u1_dev = (
            u1.to(device)
            if isinstance(u1, torch.Tensor)
            else torch.tensor(u1, device=device, dtype=torch.float64)
        )
        u2_dev = (
            u2.to(device)
            if isinstance(u2, torch.Tensor)
            else torch.tensor(u2, device=device, dtype=torch.float64)
        )
        t_dev = (
            t.to(device)
            if isinstance(t, torch.Tensor)
            else torch.tensor(t, device=device, dtype=torch.float64)
        )

        # --- Convert GRID lags to PHYSICAL lags ---
        lag_u1 = u1_dev * delta1
        lag_u2 = u2_dev * delta2

        cov_X_value = cls.filtered_covariance(lag_u1, lag_u2, t_dev, params, delta1, delta2)

        # --- Get Taper Autocorrelation Value c_gn(u) from grid ---
        u1_idx = u1_dev.long()
        u2_idx = u2_dev.long()

        idx1 = n1 - 1 + u1_idx
        idx2 = n2 - 1 + u2_idx
        in_support = (idx1 >= 0) & (idx1 < 2 * n1 - 1) & (idx2 >= 0) & (idx2 < 2 * n2 - 1)
        safe_idx1 = torch.clamp(idx1, 0, 2 * n1 - 2)
        safe_idx2 = torch.clamp(idx2, 0, 2 * n2 - 2)

        if taper_autocorr_grid.ndim == 4:
            if q_idx is None or r_idx is None:
                raise ValueError("Pair-specific taper autocorrelation requires q_idx and r_idx.")
            taper_autocorr_value = taper_autocorr_grid[q_idx, r_idx, safe_idx1, safe_idx2]
        elif taper_autocorr_grid.ndim == 2:
            taper_autocorr_value = taper_autocorr_grid[safe_idx1, safe_idx2]
        else:
            raise ValueError("taper_autocorr_grid must have two or four dimensions.")
        taper_autocorr_value = torch.where(
            in_support,
            taper_autocorr_value,
            torch.zeros_like(taper_autocorr_value),
        )

        if torch.isnan(cov_X_value).any() or torch.isnan(taper_autocorr_value).any():
            out_shape = torch.broadcast_shapes(cov_X_value.shape, taper_autocorr_value.shape)
            return torch.full(out_shape, float("nan"), device=device, dtype=torch.float64)

        return cov_X_value * taper_autocorr_value

    @classmethod
    def expected_periodogram(cls, params, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        Calculates the expected periodogram I(omega_s) (a pxp matrix in time)
        using the taper autocorrelation c_gn(u) and the four-term spatial
        aliasing sum.

        Temporal positions are interpreted as ordered, equally spaced indices;
        the covariance time lag is ``q - r`` in those index units.
        """
        if min(n1, n2, p_time) < 1:
            raise ValueError("n1, n2, and p_time must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        valid_taper_shapes = {
            (2 * n1 - 1, 2 * n2 - 1),
            (p_time, p_time, 2 * n1 - 1, 2 * n2 - 1),
        }
        if tuple(taper_autocorr_grid.shape) not in valid_taper_shapes:
            raise ValueError("taper_autocorr_grid has an incompatible shape.")
        device = params.device
        params_tensor = params
        taper_autocorr_grid = taper_autocorr_grid.to(device=device, dtype=torch.float64)

        u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
        u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
        u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing="ij")

        t_lags = torch.arange(p_time, dtype=torch.float64, device=device)

        rows = []
        has_nan = False
        for q in range(p_time):
            cols = []
            for r in range(p_time):
                t_diff = t_lags[q] - t_lags[r]

                # For 4-D taper_autocorr_grid pass per-pair indices (multivariate correction)
                _q = q if taper_autocorr_grid.ndim == 4 else None
                _r = r if taper_autocorr_grid.ndim == 4 else None
                term1 = cls._tapered_covariance(
                    u1_mesh,
                    u2_mesh,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    _q,
                    _r,
                )
                term2 = cls._tapered_covariance(
                    u1_mesh - n1,
                    u2_mesh,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    _q,
                    _r,
                )
                term3 = cls._tapered_covariance(
                    u1_mesh,
                    u2_mesh - n2,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    _q,
                    _r,
                )
                term4 = cls._tapered_covariance(
                    u1_mesh - n1,
                    u2_mesh - n2,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    _q,
                    _r,
                )

                tilde_cn_grid_qr = term1 + term2 + term3 + term4

                if not torch.isfinite(tilde_cn_grid_qr).all():
                    has_nan = True
                    cols.append(torch.zeros(n1, n2, dtype=torch.complex128, device=device))
                else:
                    cols.append(tilde_cn_grid_qr.to(torch.complex128))
            rows.append(torch.stack(cols, dim=-1))  # (n1, n2, p_time)
        tilde_cn_tensor = torch.stack(rows, dim=-2)  # (n1, n2, p_time, p_time)

        if has_nan:
            nan_shape = (n1, n2, p_time, p_time)
            return torch.full(nan_shape, float("nan"), dtype=torch.complex128, device=device)

        fft_result = torch.fft.fft2(tilde_cn_tensor, dim=(0, 1))
        normalization_factor = 1.0 / (4.0 * cmath.pi**2)
        result_raw = fft_result * normalization_factor
        return (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

    @classmethod
    def _retained_frequency_sum(cls, likelihood_terms, n1, n2):
        """Apply the configured frequency mask and return ``(sum, count)``.

        The mask is explicit in :class:`SpatialFilterSpec`; it is not inferred
        from the stencil.
        """
        if n1 < 1 or n2 < 1:
            raise ValueError("Spatial frequency dimensions must be positive.")
        if likelihood_terms.shape != (n1, n2):
            raise ValueError(f"likelihood_terms must have shape {(n1, n2)}.")

        device = likelihood_terms.device
        total_sum = torch.sum(likelihood_terms)
        exclusion = cls.filter_spec.excluded_frequencies

        retained = torch.ones((n1, n2), dtype=torch.bool, device=device)
        if exclusion == "latitude_axis":
            retained[0, :] = False
        elif exclusion == "longitude_axis":
            retained[:, 0] = False
        elif exclusion == "both_axes":
            retained[0, :] = False
            retained[:, 0] = False
        else:
            retained[0, 0] = False
        count = int(retained.sum().item())
        if count == 0:
            return likelihood_terms.new_zeros(()), 0

        # Preserve the established finite-input accumulation order.  When an
        # excluded frequency is non-finite, sum only retained entries so that
        # an irrelevant DC/axis value cannot poison the objective.
        if not torch.isfinite(likelihood_terms).all():
            return likelihood_terms[retained].sum(), count

        if exclusion == "latitude_axis":
            return total_sum - likelihood_terms[0, :].sum(), count

        if exclusion == "longitude_axis":
            return total_sum - likelihood_terms[:, 0].sum(), count

        if exclusion == "both_axes":
            row0_sum = likelihood_terms[0, :].sum()
            col0_sum = likelihood_terms[:, 0].sum()
            dc_term = likelihood_terms[0, 0]
            return total_sum - row0_sum - col0_sum + dc_term, count

        # ``identity`` and ``summed_first_differences`` retain every spatial
        # frequency except the DC point.
        return total_sum - likelihood_terms[0, 0], count

    @classmethod
    def _likelihood_terms(
        cls, params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2
    ):
        """Return per-frequency log-determinant plus trace terms."""
        if n1 < 1 or n2 < 1 or p_time < 1:
            raise ValueError("n1, n2, and p_time must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        expected_sample_shape = (n1, n2, p_time, p_time)
        if I_sample.shape != expected_sample_shape:
            raise ValueError(f"I_sample must have shape {expected_sample_shape}.")
        valid_taper_shapes = {
            (2 * n1 - 1, 2 * n2 - 1),
            (p_time, p_time, 2 * n1 - 1, 2 * n2 - 1),
        }
        if tuple(taper_autocorr_grid.shape) not in valid_taper_shapes:
            raise ValueError(
                "taper_autocorr_grid must be spatial or pair-specific with the expected lag shape."
            )

        device = I_sample.device
        params_tensor = params.to(device)
        taper = taper_autocorr_grid.to(device=device, dtype=torch.float64)
        if not torch.isfinite(params_tensor).all() or not torch.isfinite(taper).all():
            return None

        I_expected = cls.expected_periodogram(params_tensor, n1, n2, p_time, taper, delta1, delta2)
        return _spectral_likelihood_terms(I_expected, I_sample)

    @classmethod
    def negative_log_likelihood(
        cls, params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2
    ):
        """Return the average tapered Debiased Whittle objective."""
        terms = cls._likelihood_terms(
            params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2
        )
        if terms is None:
            return torch.tensor(float("inf"), device=I_sample.device, dtype=torch.float64)

        sum_loss, num_terms = cls._retained_frequency_sum(terms, n1, n2)
        if num_terms == 0:
            return torch.tensor(float("inf"), device=I_sample.device, dtype=torch.float64)
        average_loss = sum_loss / num_terms
        if not torch.isfinite(average_loss):
            return torch.tensor(float("inf"), device=I_sample.device, dtype=torch.float64)
        return average_loss

    @classmethod
    def fit(
        cls,
        parameters,
        optimizer,
        sample_periodogram,
        n1,
        n2,
        n_time,
        taper_autocorrelation,
        delta1=0.044,
        delta2=0.063,
        max_steps=5,
        gradient_tolerance=1e-5,
        loss_tolerance=1e-12,
    ) -> DebiasedWhittleFitResult:
        """Fit the seven covariance parameters with a closure-based optimizer."""
        parameters = tuple(parameters)
        if not parameters:
            raise ValueError("At least one parameter tensor is required.")
        parameter_device = parameters[0].device
        sample_periodogram = sample_periodogram.to(parameter_device)
        taper_autocorrelation = taper_autocorrelation.to(parameter_device)

        def objective(parameter_tensor):
            return cls.negative_log_likelihood(
                parameter_tensor,
                sample_periodogram,
                n1,
                n2,
                n_time,
                taper_autocorrelation,
                delta1,
                delta2,
            )

        return _optimize_parameters(
            parameters,
            optimizer,
            objective,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            loss_tolerance=loss_tolerance,
        )
