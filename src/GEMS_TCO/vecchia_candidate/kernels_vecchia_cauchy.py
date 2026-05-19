"""
kernels_vecchia_cauchy.py
=========================
Generalized Cauchy covariance variant of the Vecchia approximation.

Model
-----
    C(d) = σ² / (1 + d / α)^β
         = (φ1/φ2) / (1 + d · φ2)^β

where d is the same anisotropic, advection-corrected distance as in the Matérn kernel
and the log-parameter vector is identical:

    params[0] = log φ1      φ1 = σ² · φ2
    params[1] = log φ2      φ2 = 1 / range_lon  (α = range_lon)
    params[2] = log φ3      φ3 = (range_lon / range_lat)²
    params[3] = log φ4      φ4 = (range_lon / range_time)²
    params[4] = advec_lat
    params[5] = advec_lon
    params[6] = log nugget

β  (gc_beta, default 1.0) controls the polynomial tail:
    β = 1   standard Cauchy  C(d) = σ²/(1 + d/α)
    β > 1   faster decay
    β < 1   heavier tail / stronger long-range dependence

Comparison with Matérn (ν=0.5):
    Matérn  : C(d) = σ² · exp(−d · φ2)            exponential decay
    Cauchy  : C(d) = σ² · (1 + d · φ2)^(−β)       polynomial decay

Both agree to first order near d=0:
    exp(−d·φ2) ≈ 1 − d·φ2
    (1 + d·φ2)^{−1} ≈ 1 − d·φ2

but differ globally: Cauchy retains meaningful long-range correlations even
for small α, making temporal heads more informative.

Reference
---------
Gneiting & Schlather (2004). Stochastic Models That Separate Fractal Dimension
and the Hurst Effect. SIAM Review, 46(2), 269-282.
"""

import numpy as np
import torch
from typing import Dict, Any, List

from GEMS_TCO.kernels_vecchia import VecchiaBatched


# ── Covariance override ────────────────────────────────────────────────────────

class VecchiaBatchedCauchy(VecchiaBatched):
    """Vecchia approximation with Generalized Cauchy covariance.

    Drop-in replacement for VecchiaBatched: identical conditioning-set logic,
    identical parameter vector — only the kernel formula changes.

    Parameters
    ----------
    gc_beta : float
        Polynomial decay exponent β  (default 1.0).
    All other kwargs are forwarded to VecchiaBatched.
    """

    def __init__(self, gc_beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gc_beta = gc_beta

    # ── Heads block (exact GP, N×N covariance matrix) ─────────────────────────

    def matern_cov_aniso_STABLE_log_reparam(
        self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Cauchy covariance for the Heads (exact GP) block."""
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget    = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq   = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance    = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        cov         = sigmasq * torch.pow(1.0 + distance * phi2, -self.gc_beta)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov

    # ── Tails block (batched B×N×N covariance matrices) ───────────────────────

    def matern_cov_batched(
        self, params: torch.Tensor, x_batch: torch.Tensor
    ) -> torch.Tensor:
        """Cauchy covariance for the Tails (batched conditioning) blocks."""
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget      = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])

        dist = self.batched_manual_dist(dist_params, x_batch)
        cov  = (phi1 / phi2) * torch.pow(1.0 + dist * phi2, -self.gc_beta)

        B, N, _ = x_batch.shape
        eye = (torch.eye(N, device=self.device, dtype=torch.float64)
               .unsqueeze(0).expand(B, N, N))
        return cov + eye * (nugget + 1e-6)


# ── Fitting class ──────────────────────────────────────────────────────────────

class fit_cauchy_vecchia_lbfgs(VecchiaBatchedCauchy):
    """L-BFGS fitting wrapper for the Generalized Cauchy Vecchia model.

    Usage is identical to kernels_vecchia.fit_vecchia_lbfgs:

        model = fit_cauchy_vecchia_lbfgs(
            smooth=0.5, gc_beta=1.0,
            input_map=..., nns_map=..., mm_cond_number=100,
            nheads=300, limit_A=8, limit_B=8, limit_C=8, daily_stride=8,
        )
        model.precompute_conditioning_sets()
        opt = model.set_optimizer(params_list, lr=1.0, max_iter=100, history_size=100)
        out, steps = model.fit_vecc_lbfgs(params_list, opt, max_steps=3)
    """

    def __init__(self, smooth: float, input_map: Dict[str, Any],
                 nns_map: Dict[str, Any], mm_cond_number: int, nheads: int,
                 gc_beta: float = 1.0,
                 limit_A: int = 8, limit_B: int = 8, limit_C: int = 8,
                 daily_stride: int = 8):
        super().__init__(
            gc_beta=gc_beta,
            smooth=smooth, input_map=input_map, nns_map=nns_map,
            mm_cond_number=mm_cond_number, nheads=nheads,
            limit_A=limit_A, limit_B=limit_B, limit_C=limit_C,
            daily_stride=daily_stride,
        )

    def set_optimizer(self, param_groups, lr=1.0, max_iter=20, max_eval=None,
                      tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10):
        return torch.optim.LBFGS(
            param_groups, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn="strong_wolfe"
        )

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
        return {
            "sigma_sq":   phi1 / phi2,
            "range_lon":  1.0 / phi2,
            "range_lat":  1.0 / (phi2 * np.sqrt(phi3)),
            "range_time": 1.0 / (phi2 * np.sqrt(phi4)),
            "advec_lat":  raw[4],
            "advec_lon":  raw[5],
            "nugget":     np.exp(raw[6]),
            "gc_beta":    self.gc_beta,   # echo β in output for clarity
        }

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor],
                       optimizer: torch.optim.LBFGS,
                       max_steps: int = 5, grad_tol: float = 1e-5):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print(f"--- Starting Cauchy Vecchia L-BFGS  (β={self.gc_beta}) ---")

        def closure():
            optimizer.zero_grad()
            loss = self.vecchia_batched_likelihood(torch.stack(params_list))
            loss.backward()
            return loss

        loss = None
        for i in range(max_steps):
            loss = optimizer.step(closure)

            with torch.no_grad():
                grads = [abs(p.grad.item()) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                for j, p in enumerate(params_list):
                    g = p.grad.item() if p.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={p.item():.4f}, Grad={g}')
                print(f'  Max Abs Grad: {max_grad:.6e}')
                print("-" * 30)

            if max_grad < grad_tol:
                print(f"\nConverged at step {i+1}")
                break

        raw = [p.item() for p in params_list]
        final_loss = loss.item() if loss is not None else float('inf')
        print("Final Interpretable Params:", self._convert_params(raw))

        return raw + [final_loss], i
