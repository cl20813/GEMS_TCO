"""Shared machinery for block-target pure-spatial Vecchia models.

Estimated covariance parameters:
  log(signal_variance), log(range_lat), log(range_lon), log(nugget)

The likelihood treats time slots as independent spatial replicates that share
the same covariance parameters.  There is no advection and no temporal range.
Regression coefficients are profiled by GLS, as in the spatio-temporal Vecchia
kernels used elsewhere in the project.

Every conditional covariance receives a fixed ``1e-6`` diagonal numerical
stabilizer.  This jitter is not an estimated nugget and is therefore not
included in reported covariance parameters.

This module contains only the covariance/GLS/optimization machinery used by
the supported grouped block implementation, plus its selectable mean design.
Point-target nearest-neighbor and column-scan engines are legacy research code
and intentionally are not part of the publication API.

  - _PureSpaceVecchiaBase: covariance, GLS profiling, optimizer helpers.
  - _MeanDesignMixin: intercept + centered covariates for the mean function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

_INVALID_NLL = 1.0e10
_NUMERICAL_JITTER = 1.0e-6


@dataclass(frozen=True)
class SpatialLBFGSFitResult:
    """Validated best state returned by the pure-spatial L-BFGS fitter."""

    raw_parameters: tuple[float, ...]
    final_nll: float
    interpretable_parameters: Dict[str, float]
    steps_completed: int
    best_step: int
    converged: bool
    valid: bool
    max_abs_gradient: float
    message: str


def _stable_sqrt_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    """Take a square root without perturbing exact zero distances."""
    positive = squared_distance > 0
    safe = torch.sqrt(squared_distance.clamp_min(torch.finfo(squared_distance.dtype).eps))
    return torch.where(positive, safe, torch.zeros_like(squared_distance))


class _PureSpaceVecchiaBase:
    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        target_chunk_size: int = 4096,
    ):
        if not math.isfinite(float(smooth)) or float(smooth) <= 0.0:
            raise ValueError(f"smooth must be finite and positive, got {smooth}")
        if not input_map:
            raise ValueError("input_map must contain at least one spatial replicate")
        if int(target_chunk_size) <= 0:
            raise ValueError("target_chunk_size must be positive")
        self.smooth = float(smooth)
        self.input_map = input_map
        first_val = next(iter(input_map.values()))
        self.device = (
            first_val.device if isinstance(first_val, torch.Tensor) else torch.device("cpu")
        )
        self.target_chunk_size = int(target_chunk_size)

        self.n_features = 9
        self.lat_mean_val = 0.0
        self.is_precomputed = False

    def _validate_covariance_parameters(self, params: torch.Tensor) -> None:
        expected = getattr(self, "_n_covariance_parameters", None)
        if expected is None:
            raise RuntimeError("covariance subclass did not declare its parameter count")
        if not isinstance(params, torch.Tensor) or params.ndim != 1:
            raise ValueError("covariance parameters must be a one-dimensional torch tensor")
        if params.numel() != int(expected):
            raise ValueError(f"expected {expected} covariance parameters, got {params.numel()}")
        if params.device != self.device:
            raise ValueError(f"covariance parameters must be on {self.device}, got {params.device}")
        if params.dtype != torch.float64:
            raise ValueError(f"covariance parameters must use torch.float64, got {params.dtype}")
        if not torch.isfinite(params).all():
            raise ValueError("covariance parameters must be finite")

    def _transformed_covariance_parameters(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return transformed scalars and reject underflow/overflow.

        The first four values returned by every covariance family are signal
        variance, latitude range, longitude range, and nugget.  Additional
        values are positive family-specific parameters.  A finite log-scale
        value alone is insufficient: exponentiation can still underflow to
        zero or overflow to infinity.
        """
        transformed = self._raw_params(params)
        if not isinstance(transformed, (tuple, list)) or len(transformed) < 4:
            raise RuntimeError("_raw_params must return signal variance, two ranges, and nugget")
        values = tuple(transformed)
        if any(not isinstance(value, torch.Tensor) or value.numel() != 1 for value in values):
            raise RuntimeError("_raw_params must return scalar torch tensors")
        stacked = torch.stack([value.reshape(()) for value in values])
        positive_indices = [0, 1, 2, *range(4, len(values))]
        valid = torch.isfinite(stacked).all()
        valid = valid & (stacked[positive_indices] > 0.0).all()
        valid = valid & (stacked[3] >= 0.0)
        if not bool(valid.detach().cpu().item()):
            raise ValueError(
                "raw covariance parameters must transform to finite natural "
                "parameters with positive variance, ranges, and shape parameters "
                "and a non-negative nugget"
            )
        return values

    def _natural_parameters_are_valid(self, params: torch.Tensor) -> bool:
        try:
            self._transformed_covariance_parameters(params)
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            return False
        return True

    def _natural_values_from_raw_sequence(self, raw: List[float]) -> tuple[float, ...]:
        """Convert a raw result vector to validated natural-scale floats."""
        params = torch.as_tensor(raw, dtype=torch.float64, device=self.device)
        self._validate_covariance_parameters(params)
        transformed = self._transformed_covariance_parameters(params)
        return tuple(float(value.detach().cpu().item()) for value in transformed)

    def _raw_params(self, params: torch.Tensor):
        raise NotImplementedError("covariance subclasses must implement _raw_params")

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        raise NotImplementedError("covariance subclasses must implement _cov_from_deltas")

    def _cov_full(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Return covariance matrices for batches of pure-spatial observations."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[..., 0], diff[..., 1], params)
        nugget = self._raw_params(params)[3]
        n_points = coords.shape[1]
        eye = torch.eye(n_points, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        return cov + eye * (nugget + _NUMERICAL_JITTER)

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        dums = flat[:, 4:11].to(torch.float64)
        X = torch.cat([ones, lat, dums], dim=1)
        if not torch.isfinite(X).all():
            raise ValueError("mean-design covariates must be finite for observed rows")
        return X.reshape(*orig_shape, self.n_features)

    def _make_full_data(self, max_m: int):
        all_data_list = []
        for key, value in self.input_map.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise TypeError(
                    f"input_map[{key!r}] must be a NumPy array or torch tensor, "
                    f"got {type(value).__name__}"
                )
            tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            if tensor.ndim != 2:
                raise ValueError(f"input_map[{key!r}] must be two-dimensional, got {tensor.shape}")
            all_data_list.append(tensor)

        mean_design = getattr(self, "mean_design", "base")
        required_cols = 11 if mean_design in {"base", "latlon_hour", "hour_spatial"} else 4
        column_counts = [int(data.shape[1]) for data in all_data_list]
        if any(count < required_cols for count in column_counts):
            raise ValueError(
                f"mean_design={mean_design!r} requires at least {required_cols} columns, "
                f"got {column_counts}"
            )
        if len(set(column_counts)) != 1:
            raise ValueError(
                f"All spatial replicates must have equal column counts, got {column_counts}"
            )

        all_data_list = [d.to(self.device, dtype=torch.float64) for d in all_data_list]
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if any(length <= 0 for length in day_lengths):
            raise ValueError(f"Every spatial replicate must be non-empty, got {day_lengths}")
        if len(set(day_lengths)) != 1:
            raise ValueError(
                f"Pure-space kernels require equal grid length per time, got {day_lengths}"
            )

        supplied_grid = getattr(self, "grid_coords", None)
        if supplied_grid is None:
            reference_coords = all_data_list[0][:, :2]
            if not torch.isfinite(reference_coords).all():
                raise ValueError(
                    "The first replicate must have finite coordinates at every grid row "
                    "when grid_coords is not supplied"
                )
        else:
            reference_coords = torch.as_tensor(
                supplied_grid, dtype=torch.float64, device=self.device
            )

        for replicate_index, data in enumerate(all_data_list):
            coords = data[:, :2]
            observed = torch.isfinite(data[:, 2])
            coords_finite = torch.isfinite(coords).all(dim=1)
            if torch.any(observed & ~coords_finite):
                raise ValueError(
                    f"finite responses require finite coordinates in replicate {replicate_index}"
                )
            coordinate_match = torch.isclose(coords, reference_coords, rtol=0.0, atol=1e-10).all(
                dim=1
            )
            if torch.any(observed & ~coordinate_match):
                raise ValueError(
                    "All observed spatial rows must use the same coordinates (or match "
                    "grid_coords) in the same row order"
                )

        real_data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = real_data.shape

        y = real_data[:, 2]
        coord_ok = torch.isfinite(real_data[:, 0]) & torch.isfinite(real_data[:, 1])
        obs_ok = torch.isfinite(y) & coord_ok
        if not torch.any(obs_ok):
            raise ValueError(
                "input_map contains no observations with finite coordinates and response"
            )
        valid_lats = real_data[obs_ok, 0]
        self.lat_mean_val = (
            float(valid_lats.mean().item())
            if valid_lats.numel()
            else float(torch.nanmean(real_data[:, 0]).item())
        )

        max_m = max(0, int(max_m))
        if max_m > 0:
            dummy_block = torch.zeros((max_m, num_cols), device=self.device, dtype=torch.float64)
            for k in range(max_m):
                dummy_block[k, 0] = (k + 1) * 1e8
                dummy_block[k, 1] = (k + 1) * 1e8
                dummy_block[k, 3] = (k + 1) * 1e8
            full_data = torch.cat([real_data, dummy_block], dim=0).contiguous()
        else:
            full_data = real_data

        cumulative_len = np.cumsum([0] + day_lengths)
        valid_obs_np = obs_ok.detach().cpu().numpy()
        if max_m > 0:
            valid_obs_np = np.append(valid_obs_np, np.ones(max_m, dtype=bool))
        return all_data_list, full_data, n_real, num_cols, day_lengths, cumulative_len, valid_obs_np

    def _check_precomputed(self):
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")

    def _accumulate_gls_stats(
        self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False
    ):
        raise NotImplementedError("geometry subclasses must implement _accumulate_gls_stats")

    def profiled_negative_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        """Return the average profiled Gaussian negative log-likelihood."""
        self._validate_covariance_parameters(params)
        if not self._natural_parameters_are_valid(params):
            return params.sum() * 0.0 + _INVALID_NLL
        stats = self._accumulate_gls_stats(params, include_y_quad=True, catch_cholesky=True)
        if stats is None:
            # params.sum() * 0 preserves grad_fn so loss.backward() works
            return params.sum() * 0.0 + _INVALID_NLL
        XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N = stats
        try:
            beta = torch.linalg.solve(XT_Sinv_X, XT_Sinv_y)
        except torch.linalg.LinAlgError:
            return params.sum() * 0.0 + _INVALID_NLL
        if not torch.isfinite(beta).all():
            return params.sum() * 0.0 + _INVALID_NLL
        quad = (
            yT_Sinv_y - 2.0 * (beta.T @ XT_Sinv_y).squeeze() + (beta.T @ XT_Sinv_X @ beta).squeeze()
        )
        loss = 0.5 * (log_det + quad) / total_N + 0.5 * math.log(2.0 * math.pi)
        if not torch.isfinite(loss):
            return params.sum() * 0.0 + _INVALID_NLL
        return loss

    @staticmethod
    def _is_valid_objective(loss: torch.Tensor) -> bool:
        """Whether ``loss`` is finite and not the invalid-likelihood sentinel."""
        return bool(torch.isfinite(loss).item() and float(loss.detach().item()) < _INVALID_NLL)

    def estimate_gls_coefficients(self, params_list: List[torch.Tensor]) -> torch.Tensor:
        """Estimate profiled GLS mean coefficients at fixed covariance parameters."""
        params = torch.stack([p.reshape(()) for p in params_list])
        self._validate_covariance_parameters(params)
        self._transformed_covariance_parameters(params)
        XT_Sinv_X, XT_Sinv_y, _, _, _ = self._accumulate_gls_stats(
            params, include_y_quad=False, catch_cholesky=False
        )
        return torch.linalg.solve(XT_Sinv_X, XT_Sinv_y)

    def make_lbfgs_optimizer(
        self,
        param_groups,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=10,
    ):
        """Construct the LBFGS optimizer used by :meth:`fit_lbfgs`."""
        return torch.optim.LBFGS(
            param_groups,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        raise NotImplementedError("covariance subclasses must implement _convert_params")

    def fit_lbfgs(
        self,
        params_list: List[torch.Tensor],
        optimizer: torch.optim.LBFGS,
        max_steps: int = 50,
        grad_tol: float = 1e-5,
        verbose: bool = False,
    ) -> SpatialLBFGSFitResult:
        """Fit covariance parameters and restore the best valid accepted state.

        The initial and every post-step objective are evaluated explicitly.
        Invalid sentinel values are never treated as convergence candidates.
        If the initial state is invalid, optimization fails immediately because
        the flat penalty supplies no direction toward the valid domain.
        """
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if not math.isfinite(float(grad_tol)) or float(grad_tol) < 0:
            raise ValueError("grad_tol must be finite and non-negative")
        expected = getattr(self, "_n_covariance_parameters", None)
        if expected is None:
            raise RuntimeError("covariance subclass did not declare its parameter count")
        if len(params_list) != int(expected):
            raise ValueError(
                f"expected {expected} scalar covariance parameters, got {len(params_list)}"
            )
        if any(not isinstance(parameter, torch.Tensor) for parameter in params_list):
            raise TypeError("each covariance parameter must be a torch.Tensor")
        if any(parameter.numel() != 1 for parameter in params_list):
            raise ValueError("each covariance parameter must be scalar")
        if not isinstance(optimizer, torch.optim.LBFGS):
            raise TypeError("optimizer must be a torch.optim.LBFGS instance")
        if not any(parameter.requires_grad for parameter in params_list):
            raise ValueError("at least one covariance parameter must require gradients")
        self._validate_covariance_parameters(
            torch.stack([parameter.reshape(()) for parameter in params_list])
        )
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        if verbose:
            print("--- Starting Pure-Space Vecchia L-BFGS ---")

        def closure():
            optimizer.zero_grad()
            for parameter in params_list:
                parameter.grad = None
            params = torch.stack([p.reshape(()) for p in params_list])
            loss = self.profiled_negative_log_likelihood(params)
            loss.backward()
            return loss

        initial_loss = closure()
        if not self._is_valid_objective(initial_loss):
            raise RuntimeError(
                "initial covariance parameters produce an invalid likelihood; "
                "provide a valid starting point"
            )
        initial_gradients = [
            float(parameter.grad.detach().item())
            for parameter in params_list
            if parameter.grad is not None
        ]
        if not initial_gradients or not all(math.isfinite(value) for value in initial_gradients):
            raise RuntimeError(
                "initial covariance parameters produce non-finite likelihood gradients"
            )

        best_loss = float(initial_loss.detach().item())
        best_values = [parameter.detach().clone() for parameter in params_list]
        best_step = 0
        steps_completed = 0
        initial_max_gradient = max(abs(value) for value in initial_gradients)
        converged = initial_max_gradient < grad_tol
        termination_reason = "converged_at_initial_state" if converged else None

        for i in range(0 if converged else max_steps):
            try:
                optimizer.step(closure)
                steps_completed = i + 1
                post_step_loss = closure()
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                termination_reason = f"optimizer_failed: {exc}"
                break

            post_step_valid = self._is_valid_objective(post_step_loss)
            grads = [
                abs(float(parameter.grad.detach().item()))
                for parameter in params_list
                if parameter.grad is not None
            ]
            gradients_finite = all(math.isfinite(value) for value in grads)
            max_grad = max(grads) if grads else 0.0
            post_step_value = float(post_step_loss.detach().item())
            if post_step_valid and gradients_finite and post_step_value < best_loss:
                best_loss = post_step_value
                best_values = [parameter.detach().clone() for parameter in params_list]
                best_step = steps_completed
            if verbose:
                print(
                    f"--- Step {i + 1}/{max_steps} / "
                    f"Loss: {post_step_value:.6f} / Valid: {post_step_valid} / "
                    f"Max Grad: {max_grad:.2e} ---"
                )
            if not post_step_valid:
                termination_reason = "invalid_post_step_best_valid_state_restored"
                break
            if not gradients_finite or not grads:
                termination_reason = "non_finite_post_step_gradient_best_valid_state_restored"
                break
            if max_grad < grad_tol:
                converged = True
                termination_reason = "converged"
                if verbose:
                    print(f"Converged: max_grad {max_grad:.2e} < {grad_tol:.2e}")
                break
        if termination_reason is None:
            termination_reason = "maximum_steps_reached_best_valid_state_restored"

        with torch.no_grad():
            for parameter, best_value in zip(params_list, best_values):
                parameter.copy_(best_value)
        final_loss_tensor = closure()
        if not self._is_valid_objective(final_loss_tensor):
            raise RuntimeError("failed to restore a finite valid optimization state")
        final_gradients = [
            abs(float(parameter.grad.detach().item()))
            for parameter in params_list
            if parameter.grad is not None
        ]
        gradients_finite = all(math.isfinite(value) for value in final_gradients)
        max_gradient = max(final_gradients) if final_gradients else 0.0
        converged = bool(converged and gradients_finite and max_gradient < grad_tol)
        raw = tuple(float(parameter.detach().cpu().item()) for parameter in params_list)
        final_loss = float(final_loss_tensor.detach().cpu().item())
        message = termination_reason
        if message.startswith("converged") and not converged:
            message = "best_valid_state_restored_without_gradient_convergence"
        interpretable = self._convert_params(list(raw))
        if verbose:
            print("Final Pure-Space Params:", interpretable)
        return SpatialLBFGSFitResult(
            raw_parameters=raw,
            final_nll=final_loss,
            interpretable_parameters=interpretable,
            steps_completed=steps_completed,
            best_step=best_step,
            converged=converged,
            valid=True,
            max_abs_gradient=float(max_gradient),
            message=message,
        )


def _n_features_for_mean_design(mean_design: str) -> int:
    if mean_design == "lat":
        return 2
    if mean_design == "base":
        return 9
    if mean_design == "latlon":
        return 3
    if mean_design == "latlon_hour":
        return 10
    if mean_design == "hour_spatial":
        return 24
    raise ValueError(f"Unknown mean_design={mean_design!r}")


class _MeanDesignMixin:
    """Selectable centered mean designs for pure-spatial replicates.

    ``latlon`` is exactly ``[1, latitude, longitude]`` and matches the direct
    full-likelihood helper. ``latlon_hour`` additionally includes the seven
    supplied hour indicators. ``base`` is ``[1, latitude, hour indicators]``;
    ``hour_spatial`` allows an hour-specific intercept and spatial slopes.
    """

    def _init_mean_design(self, mean_design: str):
        if mean_design not in (
            "lat",
            "base",
            "latlon",
            "latlon_hour",
            "hour_spatial",
        ):
            raise ValueError(
                "mean_design must be one of: lat, base, latlon, latlon_hour, " "hour_spatial"
            )
        self.mean_design = str(mean_design)
        self.n_features = _n_features_for_mean_design(self.mean_design)
        self.lon_mean_val = 0.0

    def _make_full_data(self, max_m: int):
        out = super()._make_full_data(max_m)
        _, full_data, n_real, _, _, _, _ = out
        real_data = full_data[:n_real]
        y = real_data[:, 2]
        coord_ok = torch.isfinite(real_data[:, 0]) & torch.isfinite(real_data[:, 1])
        obs_ok = torch.isfinite(y) & coord_ok
        valid_lons = real_data[obs_ok, 1]
        self.lon_mean_val = (
            float(valid_lons.mean().item())
            if valid_lons.numel()
            else float(torch.nanmean(real_data[:, 1]).item())
        )
        return out

    def _hour_dummies(self, flat: torch.Tensor) -> torch.Tensor:
        dums = flat[:, 4:11].to(torch.float64)
        if dums.shape[1] < 7:
            pad = torch.zeros(
                (dums.shape[0], 7 - dums.shape[1]), device=self.device, dtype=torch.float64
            )
            dums = torch.cat([dums, pad], dim=1)
        return dums

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        lon = (flat[:, 1:2] - self.lon_mean_val).to(torch.float64)
        dums = self._hour_dummies(flat)

        if self.mean_design == "lat":
            X = torch.cat([ones, lat], dim=1)
        elif self.mean_design == "base":
            X = torch.cat([ones, lat, dums], dim=1)
        elif self.mean_design == "latlon":
            X = torch.cat([ones, lat, lon], dim=1)
        elif self.mean_design == "latlon_hour":
            X = torch.cat([ones, lat, lon, dums], dim=1)
        else:
            first_hour = (1.0 - dums.sum(dim=1, keepdim=True)).clamp(min=0.0, max=1.0)
            hour_onehot = torch.cat([first_hour, dums], dim=1)
            X = torch.cat([hour_onehot, hour_onehot * lat, hour_onehot * lon], dim=1)

        if not torch.isfinite(X).all():
            raise ValueError("mean-design covariates must be finite for observed rows")
        return X.reshape(*orig_shape, self.n_features)


__all__ = ["SpatialLBFGSFitResult"]
