"""Shared numerical base for the supported grouped-batch Vecchia models.

This module deliberately contains only operations used by block-target
Vecchia fits.  The retired point-target conditioning-set builder and its
Y-refresh/per-unit helpers live in the research archive rather than in the
installed package.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _integer_option(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Validate an integer configuration option without silently truncating it."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _stable_sqrt_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    """Take a square root without perturbing exact zero distances."""
    positive = squared_distance > 0
    safe = torch.sqrt(squared_distance.clamp_min(torch.finfo(squared_distance.dtype).eps))
    return torch.where(positive, safe, torch.zeros_like(squared_distance))


@dataclass(frozen=True)
class LBFGSFitResult:
    """Final state returned by :meth:`GroupedVecchiaBase.fit_lbfgs`."""

    raw_parameters: tuple[float, ...]
    final_nll: float
    interpretable_parameters: Dict[str, float]
    steps_completed: int
    converged: bool
    max_abs_gradient: float
    objective_evaluations: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class _LBFGSEvaluation:
    """Loss and gradients at one exact covariance-parameter vector."""

    parameters: torch.Tensor
    loss: float
    valid: bool
    gradients: tuple[torch.Tensor | None, ...]


class GroupedVecchiaBase:
    """Likelihood and optimization machinery for grouped Vecchia subclasses."""

    covariance_parameter_count = 7

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, Any],
        second_lag_stride: int = 2,
    ) -> None:
        if not isinstance(input_map, Mapping) or not input_map:
            raise ValueError("input_map must be a non-empty mapping of time slices")
        second_lag_stride = _integer_option("second_lag_stride", second_lag_stride, minimum=2)

        self.smooth = float(smooth)
        self.input_map = input_map
        self.second_lag_stride = second_lag_stride

        first_value = next(iter(input_map.values()))
        self.device = (
            first_value.device if isinstance(first_value, torch.Tensor) else torch.device("cpu")
        )

        # Mean design: intercept, centered latitude, and seven hour indicators.
        self.n_features = 9
        self._active_feature_indices = torch.arange(9, device=self.device)
        self.is_precomputed = False
        self.lat_mean_val = 0.0

    @staticmethod
    def pairwise_anisotropic_distance(
        dist_params: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return all advection-adjusted distances between two point sets.

        ``x`` and ``y`` must be two-dimensional arrays whose columns 0, 1, and
        3 are latitude, longitude, and time.  ``dist_params`` contains the
        positive latitude/time anisotropy weights followed by the latitude and
        longitude advection coefficients.
        """
        if not isinstance(dist_params, torch.Tensor):
            raise TypeError("dist_params must be a torch.Tensor")
        dist_params = dist_params.reshape(-1)
        if dist_params.numel() != 4:
            raise ValueError("dist_params must be a four-element torch.Tensor")
        if not dist_params.is_floating_point():
            raise TypeError("dist_params must use a floating-point dtype")
        if not torch.isfinite(dist_params).all() or (dist_params[:2] <= 0).any():
            raise ValueError("anisotropy weights must be positive and all parameters finite")
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch.Tensor point arrays")
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] < 4 or y.shape[1] < 4:
            raise ValueError("x and y must be two-dimensional with at least four columns")
        if x.device != y.device or dist_params.device != x.device:
            raise ValueError("dist_params, x, and y must be on the same device")
        if x.dtype != y.dtype or not x.is_floating_point():
            raise TypeError("x and y must use the same floating-point dtype")
        columns = [0, 1, 3]
        if not torch.isfinite(x[:, columns]).all() or not torch.isfinite(y[:, columns]).all():
            raise ValueError("point latitude, longitude, and time must be finite")
        phi3, phi4, advec_lat, advec_lon = dist_params

        u_vec = torch.stack(
            [
                x[:, 0] - advec_lat * x[:, 3],
                x[:, 1] - advec_lon * x[:, 3],
                x[:, 3],
            ],
            dim=1,
        )
        v_vec = torch.stack(
            [
                y[:, 0] - advec_lat * y[:, 3],
                y[:, 1] - advec_lon * y[:, 3],
                y[:, 3],
            ],
            dim=1,
        )

        one = torch.ones(1, device=x.device, dtype=phi3.dtype)
        weights = torch.stack([phi3.view(1), one, phi4.view(1)]).view(-1)
        u_sq = (u_vec.pow(2) * weights).sum(dim=1, keepdim=True)
        v_sq = (v_vec.pow(2) * weights).sum(dim=1, keepdim=True)
        uv = (u_vec * weights) @ v_vec.T
        return _stable_sqrt_distance(u_sq - 2 * uv + v_sq.T)

    @staticmethod
    def _validate_batched_coordinates(
        x_batch: torch.Tensor,
        expected_device: torch.device,
    ) -> None:
        """Validate a public batched-coordinate argument without evaluating it."""
        if not isinstance(x_batch, torch.Tensor):
            raise TypeError("x_batch must be a torch.Tensor")
        if x_batch.ndim != 3 or x_batch.shape[2] != 3:
            raise ValueError("x_batch must have shape (batch, points, 3)")
        if x_batch.device != expected_device:
            raise ValueError("dist_params and x_batch must be on the same device")
        if not x_batch.is_floating_point():
            raise TypeError("x_batch must use a floating-point dtype")
        if not torch.isfinite(x_batch).all():
            raise ValueError("x_batch latitude, longitude, and time must be finite")

    @staticmethod
    def batched_anisotropic_distance(
        dist_params: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return within-batch advection-adjusted anisotropic distances.

        ``x_batch`` must have shape ``(batch, points, 3)`` with latitude,
        longitude, and time in its final dimension.  ``dist_params`` follows
        :meth:`pairwise_anisotropic_distance`.
        """
        if not isinstance(dist_params, torch.Tensor):
            raise TypeError("dist_params must be a torch.Tensor")
        dist_params = dist_params.reshape(-1)
        if dist_params.numel() != 4:
            raise ValueError("dist_params must be a four-element torch.Tensor")
        if not dist_params.is_floating_point():
            raise TypeError("dist_params must use a floating-point dtype")
        if not torch.isfinite(dist_params).all() or (dist_params[:2] <= 0).any():
            raise ValueError("anisotropy weights must be positive and all parameters finite")
        GroupedVecchiaBase._validate_batched_coordinates(x_batch, dist_params.device)
        return GroupedVecchiaBase._batched_anisotropic_distance_unchecked(
            dist_params,
            x_batch,
        )

    @staticmethod
    def _batched_anisotropic_distance_unchecked(
        dist_params: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate batched distances after validation at the public boundary."""
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat = x_batch[:, :, 0] - advec_lat * x_batch[:, :, 2]
        x_lon = x_batch[:, :, 1] - advec_lon * x_batch[:, :, 2]
        x_time = x_batch[:, :, 2]

        d_lat = x_lat.unsqueeze(2) - x_lat.unsqueeze(1)
        d_lon = x_lon.unsqueeze(2) - x_lon.unsqueeze(1)
        d_time = x_time.unsqueeze(2) - x_time.unsqueeze(1)
        return _stable_sqrt_distance(d_lat.pow(2) * phi3 + d_lon.pow(2) + d_time.pow(2) * phi4)

    def precompute_conditioning_sets(self):
        """Build grouped conditioning tensors in a concrete geometry class."""
        raise NotImplementedError

    def _accumulate_gls_stats(
        self,
        params: torch.Tensor,
        include_y_quad: bool = True,
        catch_cholesky: bool = False,
    ):
        """Accumulate GLS sufficient statistics in a grouped subclass."""
        raise NotImplementedError

    def _check_precomputed(self) -> None:
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first!")

    def _validated_params(self, params: torch.Tensor) -> torch.Tensor:
        """Return a flat covariance-parameter vector with a clear API error."""
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a torch.Tensor")
        params = params.reshape(-1)
        if params.numel() != self.covariance_parameter_count:
            raise ValueError(
                f"expected {self.covariance_parameter_count} covariance parameters, "
                f"got {params.numel()}"
            )
        if params.device != self.device:
            raise ValueError(f"params are on {params.device}, but model data are on {self.device}")
        if not params.is_floating_point():
            raise TypeError("params must use a floating-point dtype")
        if not torch.isfinite(params).all():
            raise ValueError("params must contain only finite values")
        return params

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return torch.exp(params[6])

    @staticmethod
    def _same_ordered_points(x: torch.Tensor, y: torch.Tensor) -> bool:
        """Whether arrays describe the same ordered space-time point set.

        Point identity is inferred from latitude, longitude, and time (columns
        0, 1, and 3); responses and mean-design columns do not define a point.
        This package assumes there is at most one observation at an exact
        space-time coordinate.
        """
        if x is y:
            return True
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            return False
        if x.shape[1] < 4 or y.shape[1] < 4 or x.device != y.device:
            return False
        columns = [0, 1, 3]
        return torch.equal(x[:, columns], y[:, columns])

    def _log_cholesky_failure(self, params: torch.Tensor, label: str) -> None:
        with torch.no_grad():
            phi2 = torch.exp(params[1])
            range_lon = 1.0 / phi2
            range_lat = range_lon / torch.exp(params[2]).sqrt()
            range_time = range_lon / torch.exp(params[3]).sqrt()
            nugget = self._nugget_from_params(params)
            signal_variance = torch.exp(params[0]) / phi2
            logger.warning(
                f"[Cholesky FAIL | {label}] "
                f"signal_variance={signal_variance.item():.4f}  "
                f"range_lon={range_lon.item():.4f}  "
                f"range_lat={range_lat.item():.4f}  "
                f"range_t={range_time.item():.4f}  "
                f"nugget={nugget.item():.4e}"
            )

    def _optimization_banner(self) -> str:
        return f"--- Starting Batched L-BFGS Optimization ({self.device}) ---"

    @staticmethod
    def _solve_gls_normal_equations(
        xt_sinv_x: torch.Tensor,
        xt_sinv_y: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the exact full-rank GLS normal equations by Cholesky."""
        try:
            factor = torch.linalg.cholesky(xt_sinv_x)
            return torch.cholesky_solve(xt_sinv_y, factor)
        except torch.linalg.LinAlgError as error:
            raise RuntimeError(
                "the active mean design is rank deficient under the Vecchia "
                "precision; remove redundant mean columns before fitting"
            ) from error

    def profiled_negative_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        """Return the constant-free profiled Vecchia NLL per target.

        The identifiable linear mean coefficients are profiled out by exact
        generalized least squares.  Structurally zero mean columns are removed
        during precomputation; any remaining rank deficiency fails explicitly
        rather than introducing a hidden ridge penalty.  The returned scalar is

        ``0.5 * (conditional_log_determinant + profiled_quadratic) / N``.

        Here ``N`` is the number of target observations across all grouped
        conditionals.  The additive Gaussian constant ``0.5 * log(2*pi)`` is
        omitted, so this value is suitable for comparing fits evaluated on the
        same targets but is not the full normalized Gaussian NLL.
        """
        result, _ = self._profiled_nll_with_validity(params)
        return result

    def _profiled_nll_with_validity(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Evaluate the profiled objective and flag covariance failures.

        The public objective retains a differentiable finite penalty so it can
        be used by line-search optimizers.  The separate validity flag prevents
        that flat penalty from being mistaken for a valid optimum or a
        zero-gradient convergence event by :meth:`fit_lbfgs`.
        """
        params = self._validated_params(params)
        stats = self._accumulate_gls_stats(
            params,
            include_y_quad=True,
            catch_cholesky=True,
        )
        if stats is None:
            # Preserve grad_fn so callers can safely invoke backward().
            return params.sum() * 0.0 + 1e10, False

        xt_sinv_x, xt_sinv_y, yt_sinv_y, log_det, total_n = stats
        if total_n <= 0:
            raise RuntimeError("Vecchia likelihood has no valid target observations")
        beta = self._solve_gls_normal_equations(xt_sinv_x, xt_sinv_y)

        if not torch.isfinite(beta).all():
            raise RuntimeError("GLS coefficient solve produced non-finite values")

        quadratic = yt_sinv_y - 2 * (beta.T @ xt_sinv_y) + (beta.T @ xt_sinv_x @ beta)
        result = 0.5 * (log_det + quadratic.squeeze()) / total_n
        if not torch.isfinite(result):
            self._log_cholesky_failure(params, "profiled objective")
            return params.sum() * 0.0 + 1e10, False
        return result, True

    def estimate_gls_coefficients(self, params: torch.Tensor) -> torch.Tensor:
        """Estimate the nine mean coefficients by exact GLS.

        Coefficients for structurally absent design columns are returned as
        zero; all active coefficients are solved without regularization.
        """
        params = self._validated_params(params)
        xt_sinv_x, xt_sinv_y, _, _, _ = self._accumulate_gls_stats(
            params,
            include_y_quad=False,
            catch_cholesky=False,
        )
        active_beta = self._solve_gls_normal_equations(xt_sinv_x, xt_sinv_y)
        beta = torch.zeros(
            (self.n_features, 1),
            device=active_beta.device,
            dtype=active_beta.dtype,
        )
        beta[self._active_feature_indices] = active_beta
        return beta

    @staticmethod
    def make_lbfgs_optimizer(
        param_groups,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval=None,
        tolerance_grad: float = 1e-5,
        tolerance_change: float = 1e-9,
        history_size: int = 10,
    ) -> torch.optim.LBFGS:
        """Construct the strong-Wolfe L-BFGS optimizer used by :meth:`fit_lbfgs`."""

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

    def interpretable_parameters(self, raw: List[float]) -> Dict[str, float]:
        """Convert raw optimization parameters to covariance-scale values."""
        raw_array = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw_array.size != self.covariance_parameter_count:
            raise ValueError(
                f"expected {self.covariance_parameter_count} covariance parameters, "
                f"got {raw_array.size}"
            )
        if not np.isfinite(raw_array).all():
            raise ValueError("raw covariance parameters must be finite")
        with np.errstate(over="ignore", under="ignore"):
            phi1, phi2, phi3, phi4 = np.exp(raw_array[0:4])
            nugget = float(np.exp(raw_array[6])) if self.covariance_parameter_count == 7 else 0.0
        if not np.isfinite([phi1, phi2, phi3, phi4, nugget]).all():
            raise ValueError("raw covariance parameters overflow after exponentiation")
        if min(phi1, phi2, phi3, phi4) <= 0:
            raise ValueError("raw covariance parameters underflow after exponentiation")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            signal_variance = phi1 / phi2
            range_lon = 1.0 / phi2
            range_lat = 1.0 / (phi2 * np.sqrt(phi3))
            range_time = 1.0 / (phi2 * np.sqrt(phi4))
        if not np.isfinite([signal_variance, range_lon, range_lat, range_time]).all():
            raise ValueError("interpretable covariance parameters are outside numeric range")
        return {
            "signal_variance": float(signal_variance),
            "range_lon": float(range_lon),
            "range_lat": float(range_lat),
            "range_time": float(range_time),
            "advec_lat": float(raw_array[4]),
            "advec_lon": float(raw_array[5]),
            "nugget": nugget,
        }

    def fit_lbfgs(
        self,
        params_list: List[torch.Tensor],
        optimizer: torch.optim.LBFGS,
        max_steps: int = 50,
        grad_tol: float = 1e-5,
    ) -> LBFGSFitResult:
        """Fit covariance parameters with an existing PyTorch L-BFGS optimizer.

        The best accepted state (including the initial state) is restored
        before the structured result is returned. Exact parameter, loss, and
        gradient snapshots from the PyTorch closure are reused when L-BFGS or
        the post-step check requests the same state again. This avoids a full
        duplicate Vecchia pass without approximating the objective.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not np.isfinite(grad_tol) or grad_tol < 0:
            raise ValueError("grad_tol must be finite and non-negative")
        if len(params_list) != self.covariance_parameter_count:
            raise ValueError(
                f"expected {self.covariance_parameter_count} scalar parameters, "
                f"got {len(params_list)}"
            )
        if any(not isinstance(parameter, torch.Tensor) for parameter in params_list):
            raise TypeError("each covariance parameter must be a torch.Tensor")
        if any(parameter.numel() != 1 for parameter in params_list):
            raise ValueError("each optimized covariance parameter must be scalar")
        if not isinstance(optimizer, torch.optim.LBFGS):
            raise TypeError("optimizer must be a torch.optim.LBFGS instance")
        if not any(parameter.requires_grad for parameter in params_list):
            raise ValueError("at least one covariance parameter must require gradients")
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        logger.info(self._optimization_banner())

        last_evaluation_valid = False
        current_evaluation: _LBFGSEvaluation | None = None
        evaluation_cache: dict[tuple[str, tuple[int, ...], bytes], _LBFGSEvaluation] = {}
        objective_evaluations = 0
        cache_hits = 0

        def parameter_snapshot() -> torch.Tensor:
            return torch.stack([parameter.detach().reshape(()) for parameter in params_list])

        def parameter_key(parameters: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
            """Return a bit-exact, fit-local key for one parameter state."""
            byte_view = parameters.detach().contiguous().cpu().view(torch.uint8)
            return str(parameters.dtype), tuple(parameters.shape), byte_view.numpy().tobytes()

        def restore_gradients(evaluation: _LBFGSEvaluation) -> None:
            for parameter, gradient in zip(params_list, evaluation.gradients):
                parameter.grad = None if gradient is None else gradient.clone()

        def closure():
            nonlocal cache_hits, current_evaluation, last_evaluation_valid
            nonlocal objective_evaluations
            optimizer.zero_grad()
            current_parameters = parameter_snapshot()
            key = parameter_key(current_parameters)
            cached_evaluation = evaluation_cache.get(key)
            if cached_evaluation is not None:
                current_evaluation = cached_evaluation
                restore_gradients(current_evaluation)
                last_evaluation_valid = current_evaluation.valid
                cache_hits += 1
                return current_parameters.new_tensor(current_evaluation.loss)

            loss, last_evaluation_valid = self._profiled_nll_with_validity(
                torch.stack([parameter.reshape(()) for parameter in params_list])
            )
            loss.backward()
            current_evaluation = _LBFGSEvaluation(
                parameters=current_parameters,
                loss=float(loss.detach().item()),
                valid=last_evaluation_valid,
                gradients=tuple(
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in params_list
                ),
            )
            evaluation_cache[key] = current_evaluation
            objective_evaluations += 1
            return loss

        closure()
        if current_evaluation is None:  # pragma: no cover - closure always populates it.
            raise RuntimeError("initial Vecchia objective evaluation produced no result")
        initial_value = current_evaluation.loss
        if not last_evaluation_valid or not np.isfinite(initial_value):
            raise RuntimeError(
                "initial covariance parameters do not produce a valid "
                "positive-definite Vecchia objective"
            )
        best_loss = initial_value
        best_evaluation = current_evaluation
        steps_completed = 0
        converged = False
        max_gradient = float("inf")
        for iteration in range(max_steps):
            optimizer.step(closure)
            steps_completed = iteration + 1
            closure()
            if current_evaluation is None:  # pragma: no cover - closure always populates it.
                raise RuntimeError("post-step Vecchia objective evaluation produced no result")
            post_step_evaluation = current_evaluation
            post_step_valid = post_step_evaluation.valid

            with torch.no_grad():
                gradients = [
                    abs(gradient.item())
                    for gradient in post_step_evaluation.gradients
                    if gradient is not None
                ]
                max_gradient = max(gradients) if gradients else 0.0
                post_step_value = post_step_evaluation.loss
                if post_step_valid and np.isfinite(post_step_value) and post_step_value < best_loss:
                    best_loss = post_step_value
                    best_evaluation = post_step_evaluation
                logger.info(
                    "L-BFGS step %d/%d: nll=%.8g, valid=%s, max_abs_gradient=%.4g",
                    iteration + 1,
                    max_steps,
                    post_step_value,
                    post_step_valid,
                    max_gradient,
                )

            if post_step_valid and max_gradient < grad_tol:
                converged = True
                break

        with torch.no_grad():
            for parameter, best_value in zip(params_list, best_evaluation.parameters):
                parameter.copy_(best_value.reshape_as(parameter))
        closure()
        if current_evaluation is None:  # pragma: no cover - closure always populates it.
            raise RuntimeError("final Vecchia objective evaluation produced no result")
        if not current_evaluation.valid:
            raise RuntimeError("failed to restore a valid Vecchia optimization state")
        final_loss = current_evaluation.loss
        final_gradients = [
            abs(gradient.item())
            for gradient in current_evaluation.gradients
            if gradient is not None
        ]
        max_gradient = max(final_gradients) if final_gradients else 0.0
        converged = max_gradient < grad_tol
        raw = tuple(float(parameter.item()) for parameter in params_list)
        converted = self.interpretable_parameters(list(raw))
        logger.info("Final interpretable parameters: %s", converted)
        logger.info(
            "Vecchia objective evaluations: %d computed, %d exact-state cache hits",
            objective_evaluations,
            cache_hits,
        )
        return LBFGSFitResult(
            raw_parameters=raw,
            final_nll=final_loss,
            interpretable_parameters=converted,
            steps_completed=steps_completed,
            converged=converged,
            max_abs_gradient=float(max_gradient),
            objective_evaluations=objective_evaluations,
            cache_hits=cache_hits,
        )
