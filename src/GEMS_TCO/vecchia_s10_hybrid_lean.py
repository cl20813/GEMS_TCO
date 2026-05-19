"""
vecchia_s10_hybrid_lean.py

HybridVecchia with Matérn smooth=1.0 (ν=1), evaluated via natural cubic spline.

Matérn ν=1 correlation (no PyTorch-native closed form, hence the spline):
    f(r) = √2·r · K₁(√2·r),   f(0) = 1

Full covariance:
    cov(x,y) = (phi1/phi2) · f(phi2 · d(x,y))

where d is the anisotropic space-time distance built from phi3/phi4/advec params.

Spline strategy:
    We precompute scipy.interpolate.CubicSpline on a fine grid [0, r_max] and
    store its piecewise coefficients as frozen torch tensors.  Evaluation at
    query distances is written entirely in PyTorch (searchsorted + polynomial),
    so gradients flow through the spline w.r.t. the input distances.
    No external spline package is required beyond scipy (already a dependency).

Lean conditioning structure (Hybrid_Lean_L08F04_C4F03, offset=0.063):
  Set A (t):   20 spatial NN
  Set B (t-1): 1 anchor + 8 local NN + 4 fresh upstream-shifted NN
  Set C (t-2): 1 anchor + 4 local NN + 3 fresh 2×upstream-shifted NN
"""

import numpy as np
import torch
from scipy.special import kv as _scipy_kv
from scipy.interpolate import CubicSpline as _CubicSpline

from GEMS_TCO import kernels_vecchia
from GEMS_TCO.vecchia_candidate.kernels_vecchia_hybrid import HybridVecchiaFit


class HybridVecchiaS10Lean(HybridVecchiaFit):
    """Hybrid Vecchia, Matérn ν=1 (smooth=1.0) via cubic spline, Lean conditioning.

    Lean defaults (Hybrid_Lean_L08F04_C4F03):
        limit_A=20, limit_B_local=8, lag1_fresh=4,
        limit_C_local=4, lag2_fresh=3, lag1_lon_offset=0.063

    The Matérn ν=1 covariance has no simple closed form, so we precompute a
    natural cubic spline of the normalized correlation f(r) = √2·r·K₁(√2·r)
    and store piecewise coefficients as frozen torch tensors for differentiable
    evaluation via PyTorch autograd.
    """

    def __init__(
        self,
        input_map: dict,
        nns_map,
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B_local: int = 8,
        limit_C_local: int = 4,
        daily_stride: int = 2,
        spatial_coords=None,
        lag1_lon_offset: float = 0.063,
        lag1_fresh_count: int = 4,
        lag2_fresh_count: int = 3,
        spline_n_points: int = 500,
        spline_rmax: float = 15.0,
    ):
        # HybridVecchiaFit rejects smooth not in {0.5, 1.5}, so we call the
        # grandparent (fit_vecchia_lbfgs) directly and set attrs manually.
        kernels_vecchia.fit_vecchia_lbfgs.__init__(
            self, 1.0, input_map, nns_map, mm_cond_number, nheads,
            limit_A=limit_A,
            limit_B=limit_B_local,
            limit_C=limit_C_local,
            daily_stride=daily_stride,
        )
        self.smooth = 1.0
        self.spatial_coords = spatial_coords
        self.lag1_lon_offset = float(abs(lag1_lon_offset))
        self.lag1_fresh_count = int(lag1_fresh_count)
        self.lag2_fresh_count = int(lag2_fresh_count)

        self._build_matern1_spline(n_points=spline_n_points, r_max=spline_rmax)

    # ------------------------------------------------------------------
    # Spline construction (once at init, no gradient needed here)
    # ------------------------------------------------------------------

    def _build_matern1_spline(self, n_points: int = 500, r_max: float = 15.0):
        """Precompute natural cubic spline of Matérn ν=1 correlation function.

        f(r) = √2·r · K₁(√2·r)  for r>0,  f(0) = 1  (unit sigmasq, unit range).

        Coefficients stored as frozen torch tensors; evaluation is differentiable.
        scipy.interpolate.CubicSpline with bc_type='natural' gives the natural
        cubic spline (zero second derivative at endpoints).
        """
        r_arr = np.linspace(0.0, r_max, n_points)
        f_arr = np.empty(n_points, dtype=np.float64)
        f_arr[0] = 1.0
        sqrt2_r = np.sqrt(2.0) * r_arr[1:]
        f_arr[1:] = sqrt2_r * _scipy_kv(1, sqrt2_r)

        cs = _CubicSpline(r_arr, f_arr, bc_type='natural')

        # cs.c has shape (4, n-1):
        #   S_i(x) = cs.c[3,i] + cs.c[2,i]*dx + cs.c[1,i]*dx^2 + cs.c[0,i]*dx^3
        # Store as (n-1,) tensors for fast indexing in _spline_eval.
        dev = self.device
        self._sp_knots = torch.tensor(r_arr, dtype=torch.float64, device=dev)
        self._sp_a = torch.tensor(cs.c[3], dtype=torch.float64, device=dev)  # constant
        self._sp_b = torch.tensor(cs.c[2], dtype=torch.float64, device=dev)  # linear
        self._sp_c = torch.tensor(cs.c[1], dtype=torch.float64, device=dev)  # quadratic
        self._sp_d = torch.tensor(cs.c[0], dtype=torch.float64, device=dev)  # cubic
        self._spline_rmax = float(r_max)
        print(
            f"[HybridVecchiaS10Lean] Matérn ν=1 spline built: "
            f"{n_points} knots, r_max={r_max:.1f}, "
            f"f(0)={f_arr[0]:.4f}, f(1)={float(cs(1.0)):.4f}, device={dev}"
        )

    def _spline_eval(self, r: torch.Tensor) -> torch.Tensor:
        """Differentiable Matérn ν=1 correlation at distance r (arbitrary shape).

        Uses piecewise cubic polynomial evaluation in PyTorch so that gradients
        flow through the spline w.r.t. r (and thus w.r.t. phi2/phi3/phi4/advec).
        """
        r_c = r.clamp(0.0, self._spline_rmax)
        orig = r_c.shape
        r_flat = r_c.reshape(-1)

        # Find interval index: knots[idx] <= r_flat < knots[idx+1]
        idx = torch.searchsorted(self._sp_knots, r_flat, right=True) - 1
        idx = idx.clamp(0, len(self._sp_knots) - 2)

        dx = r_flat - self._sp_knots[idx]   # offset within interval (differentiable w.r.t. r_flat)

        # Horner evaluation: a + dx*(b + dx*(c + dx*d))
        vals = self._sp_a[idx] + dx * (self._sp_b[idx] + dx * (self._sp_c[idx] + dx * self._sp_d[idx]))
        return vals.reshape(orig)

    # ------------------------------------------------------------------
    # Covariance overrides (smooth=1.0 via spline)
    # ------------------------------------------------------------------

    def matern_cov_batched(self, params, x_batch):
        """Batched covariance for tail points — Matérn ν=1 via spline."""
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        d = self.batched_manual_dist(dist_params, x_batch)  # (B, N, N)
        cov = (phi1 / phi2) * self._spline_eval(d * phi2)

        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def matern_cov_aniso_STABLE_log_reparam(self, params, x, y):
        """Dense covariance for head points — Matérn ν=1 via spline."""
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)  # (N, M)
        cov = sigmasq * self._spline_eval(distance * phi2)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov
