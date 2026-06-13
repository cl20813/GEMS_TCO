"""
kernels_space_aniso_cauchy_cluster_060326.py

Pure-space anisotropic generalized Cauchy cluster Vecchia model.

The likelihood geometry is inherited from the existing pure-space cluster
Vecchia implementation:

  - fixed regular-grid clusters, usually 4x4 points;
  - max-min ordering on cluster centroids;
  - each target is a whole cluster block;
  - condition on previous same-time nearest cluster blocks.

Only the covariance and parameter interpretation differ.  The no-nugget model
uses the log-parameter vector

    params[0] = log phi1       phi1 = sigma^2 / range_lon
    params[1] = log phi2       phi2 = 1 / range_lon
    params[2] = log phi3       phi3 = (range_lon / range_lat)^2
    params[3] = log gc_beta

with fixed gc_alpha.  The covariance is

    C(h) = sigma^2 * (1 + d(h)^gc_alpha)^(-gc_beta / gc_alpha)

where

    d(h)^2 = (delta_lon / range_lon)^2 + (delta_lat / range_lat)^2.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from GEMS_TCO.kernels_space_base_engine_052126 import _MeanDesignMixin
from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceVecchiaFit


class _AnisoGeneralizedCauchyNoNuggetMixin:
    """Generalized Cauchy covariance with phi reparameterization."""

    def _finish_init_cauchy(self, args, kwargs, mean_design: str, gc_alpha: float):
        if "smooth" not in kwargs and not args:
            kwargs = dict(kwargs)
            kwargs["smooth"] = 0.5
        elif "smooth" in kwargs:
            kwargs = dict(kwargs)
            kwargs["smooth"] = 0.5
        else:
            args = (0.5,) + tuple(args)[1:]

        super().__init__(*args, **kwargs)
        self.gc_alpha = float(gc_alpha)
        if self.gc_alpha <= 0:
            raise ValueError(f"gc_alpha must be positive, got {gc_alpha}")
        self._init_mean_design(mean_design)

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        gc_beta = torch.exp(params[3])
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, _, _, _, _, phi2, phi3, gc_beta = self._raw_params(params)
        dist = torch.sqrt(d_lat.new_tensor(1e-8) + d_lat.pow(2) * phi3 + d_lon.pow(2))
        scaled = dist * phi2
        alpha = scaled.new_tensor(self.gc_alpha)
        corr = torch.pow(1.0 + torch.pow(scaled, alpha), -gc_beta / alpha)
        return sigmasq * corr

    def _cov_full(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[..., 0], diff[..., 1], params)
        n = coords.shape[1]
        eye = torch.eye(n, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye * 1e-6

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        phi3 = float(np.exp(raw[2]))
        gc_beta = float(np.exp(raw[3]))
        return {
            "sigmasq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "nugget": 0.0,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": gc_beta,
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
            "--- Starting Pure-Space Anisotropic Generalized Cauchy "
            f"L-BFGS (alpha={self.gc_alpha}) ---"
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
        print("Final Pure-Space Cauchy Params:", self._convert_params(raw))
        return raw + [final_loss], last_iter


class ClusterSpaceAnisoCauchyNoNuggetTrendVecchiaFit(
    _AnisoGeneralizedCauchyNoNuggetMixin,
    _MeanDesignMixin,
    ClusterSpaceVecchiaFit,
):
    """Cluster pure-space anisotropic generalized Cauchy model, nugget fixed 0."""

    def __init__(self, *args, mean_design: str = "latlon", gc_alpha: float = 0.6, **kwargs):
        self._finish_init_cauchy(args, kwargs, mean_design=mean_design, gc_alpha=gc_alpha)


class _AnisoGeneralizedCauchyFixedBetaNoNuggetMixin(_AnisoGeneralizedCauchyNoNuggetMixin):
    """Generalized Cauchy covariance with fixed beta and nugget fixed at 0."""

    def _finish_init_cauchy_fixed_beta(
        self,
        args,
        kwargs,
        mean_design: str,
        gc_alpha: float,
        gc_beta: float,
    ):
        self.gc_beta_fixed = float(gc_beta)
        if self.gc_beta_fixed <= 0:
            raise ValueError(f"gc_beta must be positive, got {gc_beta}")
        self._finish_init_cauchy(args, kwargs, mean_design=mean_design, gc_alpha=gc_alpha)

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        gc_beta = params.new_tensor(self.gc_beta_fixed)
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        phi3 = float(np.exp(raw[2]))
        return {
            "sigmasq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "nugget": 0.0,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": float(self.gc_beta_fixed),
        }

    def fit_vecc_lbfgs(
        self,
        params_list: List[torch.Tensor],
        optimizer: torch.optim.LBFGS,
        max_steps: int = 50,
        grad_tol: float = 1e-5,
    ):
        print(
            "--- Fixed beta Cauchy model: "
            f"alpha={self.gc_alpha}, beta={self.gc_beta_fixed} ---"
        )
        return super().fit_vecc_lbfgs(
            params_list,
            optimizer,
            max_steps=max_steps,
            grad_tol=grad_tol,
        )


class ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit(
    _AnisoGeneralizedCauchyFixedBetaNoNuggetMixin,
    _MeanDesignMixin,
    ClusterSpaceVecchiaFit,
):
    """Cluster pure-space anisotropic generalized Cauchy model with fixed beta."""

    def __init__(
        self,
        *args,
        mean_design: str = "latlon",
        gc_alpha: float = 0.6,
        gc_beta: float = 1.0,
        **kwargs,
    ):
        self._finish_init_cauchy_fixed_beta(
            args,
            kwargs,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
            gc_beta=gc_beta,
        )


class _AnisoGeneralizedCauchyFixedBetaNuggetMixin(_AnisoGeneralizedCauchyFixedBetaNoNuggetMixin):
    """Generalized Cauchy covariance with fixed beta and estimated nugget."""

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        nugget = torch.exp(params[3])
        gc_beta = params.new_tensor(self.gc_beta_fixed)
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _cov_full(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[..., 0], diff[..., 1], params)
        _, _, _, nugget, _, _, _, _ = self._raw_params(params)
        n = coords.shape[1]
        eye = torch.eye(n, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye * (nugget + 1e-6)

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        phi3 = float(np.exp(raw[2]))
        nugget = float(np.exp(raw[3]))
        return {
            "sigmasq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "nugget": nugget,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": float(self.gc_beta_fixed),
        }


class ClusterSpaceAnisoCauchyFixedBetaTrendVecchiaFit(
    _AnisoGeneralizedCauchyFixedBetaNuggetMixin,
    _MeanDesignMixin,
    ClusterSpaceVecchiaFit,
):
    """Cluster pure-space anisotropic generalized Cauchy model with estimated nugget."""

    def __init__(
        self,
        *args,
        mean_design: str = "latlon",
        gc_alpha: float = 0.6,
        gc_beta: float = 1.0,
        **kwargs,
    ):
        self._finish_init_cauchy_fixed_beta(
            args,
            kwargs,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
            gc_beta=gc_beta,
        )


def cauchy_phi_init_from_natural(
    sigmasq: float,
    range_lat: float,
    range_lon: float,
    gc_beta: float,
) -> Dict[str, float]:
    """Return physical phi initialization for the notebook config."""
    sigmasq = max(float(sigmasq), 1e-12)
    range_lat = max(float(range_lat), 1e-12)
    range_lon = max(float(range_lon), 1e-12)
    phi2 = 1.0 / range_lon
    phi3 = (range_lon / range_lat) ** 2
    phi1 = sigmasq * phi2
    return {
        "phi1": phi1,
        "phi2": phi2,
        "phi3": phi3,
        "gc_beta": max(float(gc_beta), 1e-12),
    }


__all__ = [
    "ClusterSpaceAnisoCauchyNoNuggetTrendVecchiaFit",
    "ClusterSpaceAnisoCauchyFixedBetaTrendVecchiaFit",
    "ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit",
    "cauchy_phi_init_from_natural",
]
