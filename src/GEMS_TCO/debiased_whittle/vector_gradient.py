"""
Gradient-filter Debiased Whittle likelihood.

This module keeps the two first differences as a vector-valued spatial filter:

    D_lat X(i, j) = X(i + 1, j) - X(i, j)
    D_lon X(i, j) = X(i, j + 1) - X(i, j)

on the common anchor grid (n_lat - 1) x (n_lon - 1).  The Whittle likelihood
then treats the data as a 2 * T dimensional multivariate process, preserving
the cross-covariance between D_lat and D_lon instead of collapsing them into
the scalar summed-first-differences filter.

Because both components are deterministic filters of one scalar field, their
spectral blocks can be rank deficient.  The likelihood uses the package's
documented diagonal loading; inferential calibration must account for that
regularization.
"""

import cmath
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ._core import (
    BaseDebiasedWhittleLikelihood,
    BaseDebiasedWhittlePreprocessor,
    DebiasedWhittleFitResult,
    _optimize_parameters,
    _spectral_likelihood_terms,
)
from .filters import LATITUDE_DIFFERENCE, LONGITUDE_DIFFERENCE

LAT_COMPONENT = 0
LON_COMPONENT = 1


class VectorGradientPreprocessor(BaseDebiasedWhittlePreprocessor):
    """Preprocess regular-grid hourly maps into vector gradient components."""

    def apply_gradient_filter(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (D_lat, D_lon) tensors on the same anchor grid.

        Each output has columns [lat, lon, filtered_value, time].
        Both components use anchors (lat[:-1], lon[:-1]) so their DFTs share the
        same spatial grid and can enter one multivariate periodogram.
        """
        if grid.ndim != 2 or grid.shape[1] < 4:
            raise ValueError(
                "Each grid must be a two-dimensional tensor with at least four columns."
            )
        if grid.size(0) == 0:
            empty = grid.new_empty((0, 4), dtype=torch.float64)
            return empty, empty

        grid, unique_lats, unique_lons, row_order = self._validated_grid(grid)
        n_lat, n_lon = unique_lats.numel(), unique_lons.numel()
        if n_lat < 2 or n_lon < 2:
            empty = grid.new_empty((0, 4), dtype=torch.float64)
            return empty, empty

        grid = grid[row_order]
        value_grid = grid[:, 2].reshape(1, 1, n_lat, n_lon)
        kernel_lat = grid.new_tensor([[[[-1.0], [1.0]]]], dtype=torch.float64)
        kernel_lon = grid.new_tensor([[[[-1.0, 1.0]]]], dtype=torch.float64)

        d_lat_full = F.conv2d(value_grid, kernel_lat, padding="valid").squeeze(0).squeeze(0)
        d_lon_full = F.conv2d(value_grid, kernel_lon, padding="valid").squeeze(0).squeeze(0)

        d_lat = d_lat_full[:, :-1]
        d_lon = d_lon_full[:-1, :]

        anchor_lats = unique_lats[:-1]
        anchor_lons = unique_lons[:-1]
        lat_grid, lon_grid = torch.meshgrid(anchor_lats, anchor_lons, indexing="ij")
        time_value = grid[0, 3].repeat(d_lat.numel())

        lat_tensor = torch.stack(
            [lat_grid.flatten(), lon_grid.flatten(), d_lat.flatten(), time_value],
            dim=1,
        )
        lon_tensor = torch.stack(
            [lat_grid.flatten(), lon_grid.flatten(), d_lon.flatten(), time_value],
            dim=1,
        )
        return lat_tensor, lon_tensor

    def generate_gradient_time_slices(
        self, lat_s: float, lat_e: float, lon_s: float, lon_e: float
    ) -> List[torch.Tensor]:
        """
        Returns tensors in time-major component order:

            [D_lat(t0), D_lon(t0), D_lat(t1), D_lon(t1), ...].
        """
        slices: List[torch.Tensor] = []
        for time_index, grid in enumerate(self.time_slices):
            subset = self._subset_grid(grid, lat_s, lat_e, lon_s, lon_e)
            if subset.size(0) == 0:
                raise ValueError(
                    f"Spatial bounds leave time-slice index {time_index} empty; "
                    "all modeled time slots must share the selected grid."
                )
            try:
                lat_diff, lon_diff = self.apply_gradient_filter(subset)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid spatial grid at time-slice index {time_index}: {exc}"
                ) from exc
            if lat_diff.size(0) > 0 and lon_diff.size(0) > 0:
                slices.extend([lat_diff, lon_diff])
            else:
                raise ValueError(
                    f"The gradient filter leaves time-slice index {time_index} empty; "
                    "select at least two latitude and two longitude coordinates."
                )
        return slices

    def generate_filtered_data(
        self, lat_s: float, lat_e: float, lon_s: float, lon_e: float
    ) -> torch.Tensor:
        """Return all gradient rows with an explicit component column."""
        rows = []
        for tensor in self.generate_gradient_time_slices(lat_s, lat_e, lon_s, lon_e):
            rows.append(tensor)
        if not rows:
            if self.time_slices:
                return self.time_slices[0].new_empty((0, 5), dtype=torch.float64)
            return torch.empty((0, 5), dtype=torch.float64)

        tagged = []
        for idx, tensor in enumerate(rows):
            component = torch.full(
                (tensor.shape[0], 1), idx % 2, dtype=tensor.dtype, device=tensor.device
            )
            tagged.append(torch.cat([tensor, component], dim=1))
        return torch.cat(tagged, dim=0)


class VectorGradientDebiasedWhittleLikelihood(BaseDebiasedWhittleLikelihood):
    """Multivariate DW likelihood for latitude/longitude first differences."""

    @staticmethod
    def _component_metadata(
        matrix_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if matrix_size < 2 or matrix_size % 2 != 0:
            raise ValueError(
                "matrix_size must be a positive multiple of the two gradient components."
            )
        idx = torch.arange(matrix_size, dtype=torch.long, device=device)
        time_idx = idx // 2
        component_idx = idx % 2
        return time_idx.to(torch.float64), component_idx

    @staticmethod
    def _component_weights(component: int):
        if int(component) == LAT_COMPONENT:
            return LATITUDE_DIFFERENCE.weights
        if int(component) == LON_COMPONENT:
            return LONGITUDE_DIFFERENCE.weights
        raise ValueError(f"Unknown gradient component: {component}")

    @staticmethod
    def gradient_covariance(u1, u2, t, params, delta1, delta2, component_q: int, component_r: int):
        """Covariance between one gradient component at q and one at r."""
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

        out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
        cov = torch.zeros(out_shape, device=device, dtype=torch.float64)
        weights_q = VectorGradientDebiasedWhittleLikelihood._component_weights(component_q)
        weights_r = VectorGradientDebiasedWhittleLikelihood._component_weights(component_r)

        for (a_lat, a_lon), w_a in weights_q:
            for (b_lat, b_lon), w_b in weights_r:
                lag_u1 = u1_dev + (a_lat - b_lat) * delta1
                lag_u2 = u2_dev + (a_lon - b_lon) * delta2
                cov = (
                    cov
                    + w_a
                    * w_b
                    * VectorGradientDebiasedWhittleLikelihood.spatiotemporal_covariance(
                        lag_u1, lag_u2, t_dev, params
                    )
                )
        return cov

    @staticmethod
    def _tapered_gradient_covariance(
        u1,
        u2,
        t,
        params,
        n1,
        n2,
        taper_autocorr_grid,
        delta1,
        delta2,
        q_idx,
        r_idx,
        component_q: int,
        component_r: int,
    ):
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

        lag_u1 = u1_dev * delta1
        lag_u2 = u2_dev * delta2
        cov_value = VectorGradientDebiasedWhittleLikelihood.gradient_covariance(
            lag_u1, lag_u2, t_dev, params, delta1, delta2, component_q, component_r
        )

        idx1 = n1 - 1 + u1_dev.long()
        idx2 = n2 - 1 + u2_dev.long()
        in_support = (idx1 >= 0) & (idx1 < 2 * n1 - 1) & (idx2 >= 0) & (idx2 < 2 * n2 - 1)
        safe_idx1 = torch.clamp(idx1, 0, 2 * n1 - 2)
        safe_idx2 = torch.clamp(idx2, 0, 2 * n2 - 2)
        if taper_autocorr_grid.ndim == 4:
            taper_value = taper_autocorr_grid[q_idx, r_idx, safe_idx1, safe_idx2]
        elif taper_autocorr_grid.ndim == 2:
            taper_value = taper_autocorr_grid[safe_idx1, safe_idx2]
        else:
            raise ValueError("taper_autocorr_grid must have two or four dimensions.")
        taper_value = torch.where(in_support, taper_value, torch.zeros_like(taper_value))
        return cov_value * taper_value

    @staticmethod
    def expected_periodogram(params, n1, n2, matrix_size, taper_autocorr_grid, delta1, delta2):
        """Return the expected matrix periodogram of both gradient components."""

        if min(n1, n2, matrix_size) < 1:
            raise ValueError("n1, n2, and matrix_size must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        valid_taper_shapes = {
            (2 * n1 - 1, 2 * n2 - 1),
            (matrix_size, matrix_size, 2 * n1 - 1, 2 * n2 - 1),
        }
        if tuple(taper_autocorr_grid.shape) not in valid_taper_shapes:
            raise ValueError("taper_autocorr_grid has an incompatible shape.")
        device = params.device
        params_tensor = params
        taper_autocorr_grid = taper_autocorr_grid.to(device=device, dtype=torch.float64)

        time_idx, component_idx = VectorGradientDebiasedWhittleLikelihood._component_metadata(
            matrix_size, device=device
        )
        u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
        u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
        u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing="ij")

        rows = []
        for q in range(matrix_size):
            cols = []
            for r in range(matrix_size):
                t_diff = time_idx[q] - time_idx[r]
                cq = int(component_idx[q].item())
                cr = int(component_idx[r].item())
                term1 = VectorGradientDebiasedWhittleLikelihood._tapered_gradient_covariance(
                    u1_mesh,
                    u2_mesh,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    q,
                    r,
                    cq,
                    cr,
                )
                term2 = VectorGradientDebiasedWhittleLikelihood._tapered_gradient_covariance(
                    u1_mesh - n1,
                    u2_mesh,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    q,
                    r,
                    cq,
                    cr,
                )
                term3 = VectorGradientDebiasedWhittleLikelihood._tapered_gradient_covariance(
                    u1_mesh,
                    u2_mesh - n2,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    q,
                    r,
                    cq,
                    cr,
                )
                term4 = VectorGradientDebiasedWhittleLikelihood._tapered_gradient_covariance(
                    u1_mesh - n1,
                    u2_mesh - n2,
                    t_diff,
                    params_tensor,
                    n1,
                    n2,
                    taper_autocorr_grid,
                    delta1,
                    delta2,
                    q,
                    r,
                    cq,
                    cr,
                )
                cols.append((term1 + term2 + term3 + term4).to(torch.complex128))
            rows.append(torch.stack(cols, dim=-1))

        tilde_cn_tensor = torch.stack(rows, dim=-2)
        fft_result = torch.fft.fft2(tilde_cn_tensor, dim=(0, 1))
        result_raw = fft_result * (1.0 / (4.0 * cmath.pi**2))
        return (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

    @staticmethod
    def negative_log_likelihood(
        params, I_sample, n1, n2, matrix_size, taper_autocorr_grid, delta1, delta2
    ):
        """Return the average vector-gradient Debiased Whittle objective."""

        if min(n1, n2, matrix_size) < 1:
            raise ValueError("n1, n2, and matrix_size must be positive.")
        if not isinstance(params, torch.Tensor) or params.ndim != 1 or params.numel() != 7:
            raise ValueError("params must be a one-dimensional tensor with seven elements.")
        if I_sample.shape != (n1, n2, matrix_size, matrix_size):
            raise ValueError("I_sample has an incompatible shape.")
        VectorGradientDebiasedWhittleLikelihood._component_metadata(matrix_size, I_sample.device)
        valid_taper_shapes = {
            (2 * n1 - 1, 2 * n2 - 1),
            (matrix_size, matrix_size, 2 * n1 - 1, 2 * n2 - 1),
        }
        if tuple(taper_autocorr_grid.shape) not in valid_taper_shapes:
            raise ValueError("taper_autocorr_grid has an incompatible shape.")

        device = I_sample.device
        params_tensor = params.to(device)
        sample = I_sample.to(device=device, dtype=torch.complex128)
        taper = taper_autocorr_grid.to(device=device, dtype=torch.float64)
        if not all(torch.isfinite(tensor).all() for tensor in (params_tensor, taper)):
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)

        I_expected = VectorGradientDebiasedWhittleLikelihood.expected_periodogram(
            params_tensor, n1, n2, matrix_size, taper, delta1, delta2
        )
        terms = _spectral_likelihood_terms(I_expected, sample)

        retained = torch.ones((n1, n2), dtype=torch.bool, device=device)
        retained[0, 0] = False
        retained_count = int(retained.sum().item())
        if retained_count == 0:
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)
        loss = terms[retained].sum() / retained_count
        if not torch.isfinite(loss):
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)
        return loss

    @classmethod
    def fit(
        cls,
        parameters,
        optimizer,
        sample_periodogram,
        n1,
        n2,
        matrix_size,
        taper_autocorrelation,
        delta1=0.044,
        delta2=0.063,
        max_steps=5,
        gradient_tolerance=1e-5,
        loss_tolerance=1e-12,
    ) -> DebiasedWhittleFitResult:
        """Fit the vector-gradient covariance parameters."""
        parameters = tuple(parameters)
        if not parameters:
            raise ValueError("At least one parameter tensor is required.")
        device = parameters[0].device
        sample_periodogram = sample_periodogram.to(device)
        taper_autocorrelation = taper_autocorrelation.to(device)

        def objective(parameter_tensor):
            return cls.negative_log_likelihood(
                parameter_tensor,
                sample_periodogram,
                n1,
                n2,
                matrix_size,
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


__all__ = [
    "LAT_COMPONENT",
    "LON_COMPONENT",
    "VectorGradientPreprocessor",
    "VectorGradientDebiasedWhittleLikelihood",
]
