"""
Space-time corridor Vecchia wrappers with generalized Cauchy correlations.

The conditioning geometry, mean model, optimizer, and anisotropic/advection
distance are inherited from the existing 4x4 lag-643 real-data corridor model.
Only the correlation shape is changed to

    corr(r) = (1 + r^alpha)^(-beta / alpha),

where r is the same scaled ST distance used by the Matérn wrappers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    RealDataCorridorWidth4x4Lag643VecchiaFit,
)
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (
    RealDataCorridorWidth4x4Lag432VecchiaFit,
)


class _STGeneralizedCauchyMixin:
    def _init_st_cauchy(self, gc_alpha: float, gc_beta: float):
        self.gc_alpha = float(gc_alpha)
        self.gc_beta = float(gc_beta)
        if self.gc_alpha <= 0:
            raise ValueError(f"gc_alpha must be positive, got {gc_alpha}")
        if self.gc_beta <= 0:
            raise ValueError(f"gc_beta must be positive, got {gc_beta}")

    def _cauchy_corr(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        alpha = scaled_distance.new_tensor(self.gc_alpha)
        beta = scaled_distance.new_tensor(self.gc_beta)
        scaled = scaled_distance.clamp_min(1e-10)
        return torch.pow(1.0 + torch.pow(scaled, alpha), -beta / alpha)

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return torch.exp(params[6])

    def matern_cov_batched(self, params: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        distance = self.batched_manual_dist(dist_params, x_batch)
        cov = (phi1 / phi2) * self._cauchy_corr(distance * phi2)

        _, n_points, _ = x_batch.shape
        eye = torch.eye(n_points, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye.expand_as(cov) * (nugget + 1e-6)

    def matern_cov_aniso_STABLE_log_reparam(
        self,
        params: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        cov = sigmasq * self._cauchy_corr(distance * phi2)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        phi3 = float(np.exp(raw[2]))
        phi4 = float(np.exp(raw[3]))
        range_lon = 1.0 / phi2
        return {
            "sigma_sq": phi1 / phi2,
            "range_lon": range_lon,
            "range_lat": range_lon / np.sqrt(phi3),
            "range_time": range_lon / np.sqrt(phi4),
            "advec_lat": float(raw[4]),
            "advec_lon": float(raw[5]),
            "nugget": float(np.exp(raw[6])) if len(raw) > 6 else 0.0,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": float(self.gc_beta),
        }

    def fit_vecc_lbfgs(
        self,
        params_list: List[torch.Tensor],
        optimizer: torch.optim.LBFGS,
        max_steps: int = 50,
        grad_tol: float = 1e-5,
    ):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print(
            "--- Starting ST generalized Cauchy L-BFGS "
            f"(alpha={self.gc_alpha:g}, beta={self.gc_beta:g}) ---"
        )

        def closure():
            optimizer.zero_grad()
            params = torch.stack([p.reshape(()) for p in params_list])
            loss = self.vecchia_batched_likelihood(params)
            loss.backward()
            return loss

        loss = None
        last_iter = 0
        for i in range(max_steps):
            last_iter = i
            loss = optimizer.step(closure)
            with torch.no_grad():
                grads = [abs(float(p.grad.detach().item())) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0
                print(
                    f"--- Step {i + 1}/{max_steps} / "
                    f"Loss: {float(loss.detach().item()):.6f} / Max Grad: {max_grad:.2e} ---"
                )
            if max_grad < grad_tol:
                print(f"Converged: max_grad {max_grad:.2e} < {grad_tol:.2e}")
                break

        raw = [float(p.detach().cpu().item()) for p in params_list]
        final_loss = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float("nan")
        print("Final ST Cauchy Params:", self._convert_params(raw))
        return raw + [final_loss], last_iter


class _STNoNuggetGeneralizedCauchyMixin(_STGeneralizedCauchyMixin):
    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_tensor(0.0)


class RealDataCorridorWidth4x4Lag643GeneralizedCauchyFit(
    _STGeneralizedCauchyMixin,
    RealDataCorridorWidth4x4Lag643VecchiaFit,
):
    """4x4 lag-643 corridor model with fixed generalized Cauchy alpha/beta."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(4, 4),
        n_neighbor_blocks_t: int = 6,
        lag1_local_blocks: int = 4,
        lag2_local_blocks: int = 3,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.126,
        lag2_lon_offset=None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        **_ignored,
    ):
        if tuple(block_shape) != (4, 4):
            raise ValueError(f"corridor lag643 requires block_shape=(4, 4), got {block_shape}")
        if int(n_neighbor_blocks_t) != 6 or int(lag1_local_blocks) != 4 or int(lag2_local_blocks) != 3:
            raise ValueError(
                "corridor lag643 requires n_neighbor_blocks_t=6, "
                "lag1_local_blocks=4, lag2_local_blocks=3"
            )
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=float(abs(lag1_lon_offset)),
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit(
    _STNoNuggetGeneralizedCauchyMixin,
    RealDataCorridorWidth4x4Lag643VecchiaFit,
):
    """4x4 lag-643 corridor generalized Cauchy model with nugget fixed at 0."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(4, 4),
        n_neighbor_blocks_t: int = 6,
        lag1_local_blocks: int = 4,
        lag2_local_blocks: int = 3,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.126,
        lag2_lon_offset=None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        **_ignored,
    ):
        if tuple(block_shape) != (4, 4):
            raise ValueError(f"corridor lag643 requires block_shape=(4, 4), got {block_shape}")
        if int(n_neighbor_blocks_t) != 6 or int(lag1_local_blocks) != 4 or int(lag2_local_blocks) != 3:
            raise ValueError(
                "corridor lag643 requires n_neighbor_blocks_t=6, "
                "lag1_local_blocks=4, lag2_local_blocks=3"
            )
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=float(abs(lag1_lon_offset)),
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class RealDataCorridorWidth4x4Lag432GeneralizedCauchyFit(
    _STGeneralizedCauchyMixin,
    RealDataCorridorWidth4x4Lag432VecchiaFit,
):
    """4x4 lag-432 corridor model with fixed generalized Cauchy alpha/beta."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(4, 4),
        n_neighbor_blocks_t: int = 4,
        lag1_local_blocks: int = 3,
        lag2_local_blocks: int = 2,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.126,
        lag2_lon_offset=None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        **_ignored,
    ):
        if tuple(block_shape) != (4, 4):
            raise ValueError(f"corridor lag432 requires block_shape=(4, 4), got {block_shape}")
        if int(n_neighbor_blocks_t) != 4 or int(lag1_local_blocks) != 3 or int(lag2_local_blocks) != 2:
            raise ValueError(
                "corridor lag432 requires n_neighbor_blocks_t=4, "
                "lag1_local_blocks=3, lag2_local_blocks=2"
            )
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=float(abs(lag1_lon_offset)),
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class RealDataCorridorWidth4x4Lag432NoNuggetGeneralizedCauchyFit(
    _STNoNuggetGeneralizedCauchyMixin,
    RealDataCorridorWidth4x4Lag432VecchiaFit,
):
    """4x4 lag-432 corridor generalized Cauchy model with nugget fixed at 0."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(4, 4),
        n_neighbor_blocks_t: int = 4,
        lag1_local_blocks: int = 3,
        lag2_local_blocks: int = 2,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.126,
        lag2_lon_offset=None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        **_ignored,
    ):
        if tuple(block_shape) != (4, 4):
            raise ValueError(f"corridor lag432 requires block_shape=(4, 4), got {block_shape}")
        if int(n_neighbor_blocks_t) != 4 or int(lag1_local_blocks) != 3 or int(lag2_local_blocks) != 2:
            raise ValueError(
                "corridor lag432 requires n_neighbor_blocks_t=4, "
                "lag1_local_blocks=3, lag2_local_blocks=2"
            )
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=float(abs(lag1_lon_offset)),
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


__all__ = [
    "RealDataCorridorWidth4x4Lag643GeneralizedCauchyFit",
    "RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit",
    "RealDataCorridorWidth4x4Lag432GeneralizedCauchyFit",
    "RealDataCorridorWidth4x4Lag432NoNuggetGeneralizedCauchyFit",
]
