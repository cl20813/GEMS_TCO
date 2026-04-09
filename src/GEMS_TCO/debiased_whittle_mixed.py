"""
debiased_whittle_mixed.py

Mixed-frequency Debiased Whittle (DW) estimator.

Motivation
----------
Spatial differencing filters suppress low-frequency power:
  2D filter  |H(ω)|² = 4sin²(ω1/2)·4sin²(ω2/2)  → 0 as ω→0
  lat-1D     |H(ω)|² = 2(1-cos ω1)               → 0 as ω1→0

Advection parameters are identified through temporal cross-spectrum phase shifts
  Δφ(ω,τ) = ω1·v_lat·τ + ω2·v_lon·τ
which are most detectable where spatial power is largest — the low frequencies.
Differencing destroys exactly the information that identifies advection best.

Strategy
--------
Split the total Whittle objective by spatial frequency:

  L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω; θ)   +   Σ_{ω∈Ω_H} ℓ_diff(ω; θ)

where:
  Ω_L = { (k1,k2) : k1 ≤ K1, k2 ≤ K2 } \\ {(0,0)}  — raw periodogram
  Ω_H = complement on the diff grid             — 2D-filter periodogram

Statistical validity (composite likelihood, Varin et al. 2011 Stat. Sinica):
  Each piece has E[∂ℓ/∂θ]=0 at the true θ  →  combined score is unbiased
  →  consistent estimator.
  Asymptotic variance via Godambe / sandwich matrix (not standard Fisher info).

Frequency partition (α = fraction of each dimension treated as low-freq):
  K1 = floor(n1 · α),  K2 = floor(n2 · α)
  Recommended α ∈ {0.15, 0.20, 0.25} for this domain.

Covariance functions
--------------------
  raw  piece: Cov(X, X)(u,v,τ) = C_X(u·δ1, v·δ2, τ)            [identity H]
  diff piece: Cov(Z, Z)(u,v,τ) = ΣΣ h_ab·h_cd·C_X(u+(a-c)δ1, v+(b-d)δ2, τ)
                                   h: {(0,0):-1, (1,0):+1, (0,1):+1, (1,1):-1}
"""

# Configuration
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"

import sys
import cmath
import numpy as np

sys.path.append(gems_tco_path)

from pathlib import Path
import torch
import torch.fft
import torch.nn.functional as F
from torch.nn import Parameter

from GEMS_TCO import configuration as config


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class full_vecc_dw_likelihoods:
    def __init__(self, daily_aggregated_tensors, daily_hourly_maps,
                 day_idx, params_list, lat_range, lon_range):
        self.day_idx                   = day_idx
        self.lat_range                 = lat_range
        self.lon_range                 = lon_range
        self.daily_aggregated_tensors  = daily_aggregated_tensors
        self.daily_hourly_maps         = daily_hourly_maps
        self.daily_aggregated_tensor   = daily_aggregated_tensors[day_idx]
        self.daily_hourly_map          = daily_hourly_maps[day_idx]
        self.params_list = [
            torch.tensor([val], dtype=torch.float64, requires_grad=True, device='cpu')
            for val in params_list
        ]
        self.params_tensor = torch.cat(self.params_list)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — RAW  (no spatial filter, per-slice demean)
# ─────────────────────────────────────────────────────────────────────────────

class debiased_whittle_preprocess_raw(full_vecc_dw_likelihoods):
    """
    Returns full nlat × nlon grid after spatial demeaning.
    No filter applied → identity transfer H(ω) = 1.
    """

    def __init__(self, daily_aggregated_tensors, daily_hourly_maps,
                 day_idx, params_list, lat_range, lon_range):
        super().__init__(daily_aggregated_tensors, daily_hourly_maps,
                         day_idx, params_list, lat_range, lon_range)

    def _subset(self, t, lat_s, lat_e, lon_s, lon_e):
        lm = (t[:, 0] >= lat_s) & (t[:, 0] <= lat_e)
        nm = (t[:, 1] >= lon_s) & (t[:, 1] <= lon_e)
        return t[lm & nm].clone()

    def _passthrough(self, df: torch.Tensor) -> torch.Tensor:
        if df.size(0) == 0:
            return torch.empty(0, 4, dtype=torch.float64)
        if df.dtype != torch.float64:
            df = df.to(torch.float64)
        ulat = torch.unique(df[:, 0]); ulon = torch.unique(df[:, 1])
        nl, nc = ulat.size(0), ulon.size(0)
        if df.size(0) != nl * nc:
            raise ValueError("Incomplete grid in raw passthrough.")
        # demean observed cells
        vals = df[:, 2]; valid = ~torch.isnan(vals)
        out = df.clone()
        if valid.sum() > 0:
            out[valid, 2] = vals[valid] - vals[valid].mean()
        return out

    def generate_spatially_filtered_days(self, lat_s, lat_e, lon_s, lon_e):
        chunks = []
        for _, tensor in self.daily_hourly_map.items():
            sub = self._subset(tensor, lat_s, lat_e, lon_s, lon_e)
            if sub.size(0) > 0:
                try:
                    chunks.append(self._passthrough(sub))
                except ValueError as e:
                    print(f"[raw preprocess] skip: {e}")
        if chunks:
            return torch.cat(chunks, dim=0)
        return torch.empty(0, 4, dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — DIFF  (2D separable filter [-1,1;1,-1])
# ─────────────────────────────────────────────────────────────────────────────

class debiased_whittle_preprocess_diff(full_vecc_dw_likelihoods):
    """
    Applies Z(i,j) = -X(i,j) + X(i,j+1) + X(i+1,j) - X(i+1,j+1).
    Output grid: (nlat-1) × (nlon-1).
    Transfer function: |H(ω)|² = 4sin²(ω1/2)·4sin²(ω2/2).
    """

    def __init__(self, daily_aggregated_tensors, daily_hourly_maps,
                 day_idx, params_list, lat_range, lon_range):
        super().__init__(daily_aggregated_tensors, daily_hourly_maps,
                         day_idx, params_list, lat_range, lon_range)

    def _subset(self, t, lat_s, lat_e, lon_s, lon_e):
        lm = (t[:, 0] >= lat_s) & (t[:, 0] <= lat_e)
        nm = (t[:, 1] >= lon_s) & (t[:, 1] <= lon_e)
        return t[lm & nm].clone()

    def _apply_diff_2d(self, df: torch.Tensor) -> torch.Tensor:
        if df.size(0) == 0:
            return torch.empty(0, 4, dtype=torch.float64)
        if df.dtype != torch.float64:
            df = df.to(torch.float64)
        ulat = torch.unique(df[:, 0]); ulon = torch.unique(df[:, 1])
        nl, nc = ulat.size(0), ulon.size(0)
        if df.size(0) != nl * nc:
            raise ValueError("Incomplete grid in diff preprocessing.")
        if nl < 2 or nc < 2:
            return torch.empty(0, 4)
        sidx = (df[:, 0] * 1e6 + df[:, 1]).argsort()
        df = df[sidx]
        ozone = df[:, 2].reshape(1, 1, nl, nc)
        kernel = torch.tensor([[[[-1., 1.], [1., -1.]]]], dtype=torch.float64, device=df.device)
        fgrid = F.conv2d(ozone, kernel, padding='valid').squeeze()
        if fgrid.dim() == 1:
            fgrid = fgrid.unsqueeze(0)
        nlats = ulat[:-1]; nlons = ulon[:-1]
        g_lat, g_lon = torch.meshgrid(nlats, nlons, indexing='ij')
        tval = df[0, 3].repeat(fgrid.flatten().size(0))
        return torch.stack([g_lat.flatten(), g_lon.flatten(), fgrid.flatten(), tval], dim=1)

    def generate_spatially_filtered_days(self, lat_s, lat_e, lon_s, lon_e):
        chunks = []
        for _, tensor in self.daily_hourly_map.items():
            sub = self._subset(tensor, lat_s, lat_e, lon_s, lon_e)
            if sub.size(0) > 0:
                try:
                    d = self._apply_diff_2d(sub)
                    if d.size(0) > 0:
                        chunks.append(d)
                except ValueError as e:
                    print(f"[diff preprocess] skip: {e}")
        if chunks:
            return torch.cat(chunks, dim=0)
        return torch.empty(0, 4, dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-frequency Debiased Whittle Likelihood
# ─────────────────────────────────────────────────────────────────────────────

class debiased_whittle_likelihood:

    def __init__(self):
        pass

    # =========================================================================
    # 1.  Tapering & DFT  (shared, identical to other DW modules)
    # =========================================================================

    @staticmethod
    def cgn_hamming(u, n1, n2):
        u1, u2 = u
        dev = u1.device if isinstance(u1, torch.Tensor) else (
              u2.device if isinstance(u2, torch.Tensor) else torch.device('cpu'))
        u1t = u1.to(dev) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=dev, dtype=torch.float64)
        u2t = u2.to(dev) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=dev, dtype=torch.float64)
        n1e = float(n1) if n1 > 0 else 1.0
        n2e = float(n2) if n2 > 0 else 1.0
        h1 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u1t / n1e)
        h2 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u2t / n2e)
        return h1 * h2

    @staticmethod
    def generate_Jvector_tapered_mv(tensor_list, tapering_func,
                                    lat_col, lon_col, val_col, device):
        """Tapered DFT with per-variate normalization and obs_masks."""
        p = len(tensor_list)
        _empty = lambda: (torch.empty(0,0,0,device=device,dtype=torch.complex128),0,0,0,None,None)
        if p == 0: return _empty()
        valid = [t for t in tensor_list if t.numel()>0 and t.shape[1]>max(lat_col,lon_col,val_col)]
        if not valid: return _empty()
        try:
            all_lats = torch.cat([t[:,lat_col] for t in valid])
            all_lons = torch.cat([t[:,lon_col] for t in valid])
        except IndexError: return _empty()
        all_lats = all_lats[~torch.isnan(all_lats)&~torch.isinf(all_lats)]
        all_lons = all_lons[~torch.isnan(all_lons)&~torch.isinf(all_lons)]
        if all_lats.numel()==0 or all_lons.numel()==0: return _empty()
        ulat, ulon = torch.unique(all_lats), torch.unique(all_lons)
        n1, n2 = len(ulat), len(ulon)
        lat_map = {v.item(): i for i,v in enumerate(ulat)}
        lon_map = {v.item(): i for i,v in enumerate(ulon)}
        u1m, u2m = torch.meshgrid(torch.arange(n1,dtype=torch.float64),
                                   torch.arange(n2,dtype=torch.float64), indexing='ij')
        taper = tapering_func((u1m, u2m), n1, n2).to(device)
        ffts, masks = [], []
        for tensor in tensor_list:
            dg = torch.zeros((n1,n2), dtype=torch.float64, device=device)
            om = torch.zeros((n1,n2), dtype=torch.bool,    device=device)
            if tensor.numel()>0 and tensor.shape[1]>max(lat_col,lon_col,val_col):
                for row in tensor:
                    li, lj = row[lat_col].item(), row[lon_col].item()
                    if not (np.isnan(li) or np.isnan(lj)):
                        i = lat_map.get(li); j = lon_map.get(lj)
                        if i is not None and j is not None:
                            v = row[val_col].item() if isinstance(row[val_col],torch.Tensor) else row[val_col]
                            if not (np.isnan(v) or np.isinf(v)):
                                dg[i,j] = v; om[i,j] = True
            dgt = dg * taper
            if torch.isnan(dgt).any() or torch.isinf(dgt).any():
                dgt = torch.nan_to_num(dgt, nan=0., posinf=0., neginf=0.)
            ffts.append(torch.fft.fft2(dgt)); masks.append(om)
        if not ffts: return torch.empty(0,0,0,device=device,dtype=torch.complex128), n1,n2,0,taper,None
        obs_masks = torch.stack(masks, dim=0)
        normed = []
        for fq, om in zip(ffts, masks):
            Hq = (taper * om.to(taper.dtype)).pow(2).sum()
            nq = (torch.sqrt(1./Hq)/(2.*cmath.pi)).to(device) if Hq>=1e-12 \
                 else torch.tensor(0., device=device, dtype=torch.float64)
            normed.append(fq * nq)
        return torch.stack(normed, dim=2).to(device), n1, n2, p, taper, obs_masks

    @staticmethod
    def calculate_taper_autocorrelation_multivariate(taper, obs_masks, n1, n2, device):
        taper = taper.to(device); obs_masks = obs_masks.to(device)
        p = obs_masks.shape[0]; N1, N2 = 2*n1-1, 2*n2-1
        g_all = taper.unsqueeze(0) * obs_masks.to(taper.dtype)
        H_all = (g_all**2).sum(dim=(1,2))
        gf = torch.fft.fft2(g_all, s=(N1,N2))
        res = torch.zeros((p,p,N1,N2), device=device, dtype=taper.dtype)
        for q in range(p):
            for r in range(p):
                cross = torch.fft.ifft2(gf[q]*gf[r].conj()).real
                res[q,r] = torch.fft.fftshift(cross) / (torch.sqrt(H_all[q]*H_all[r])+1e-12)
        return res

    @staticmethod
    def calculate_sample_periodogram_vectorized(J):
        if torch.isnan(J).any() or torch.isinf(J).any():
            n1,n2,p = J.shape
            return torch.full((n1,n2,p,p), float('nan'), dtype=torch.complex128, device=J.device)
        return J.unsqueeze(-1) @ J.unsqueeze(-2).conj()

    # =========================================================================
    # 2a.  Covariance kernel — shared
    # =========================================================================

    @staticmethod
    def cov_x_kernel(u1, u2, t, params):
        """C_X(u1,u2,t) — 7-param spatio-temporal exponential kernel."""
        dev = params.device
        u1 = u1.to(dev) if isinstance(u1,torch.Tensor) else torch.tensor(u1,device=dev,dtype=torch.float64)
        u2 = u2.to(dev) if isinstance(u2,torch.Tensor) else torch.tensor(u2,device=dev,dtype=torch.float64)
        t  = t.to(dev)  if isinstance(t, torch.Tensor) else torch.tensor(t, device=dev,dtype=torch.float64)
        if torch.isnan(params).any() or torch.isinf(params).any():
            return torch.full(torch.broadcast_shapes(u1.shape,u2.shape,t.shape),
                              float('nan'), device=dev, dtype=torch.float64)
        phi1 = torch.exp(params[0]); phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2]); phi4 = torch.exp(params[3])
        vl = params[4]; vn = params[5]; nug = torch.exp(params[6])
        eps = 1e-12
        sig  = phi1 / (phi2 + eps)
        rl   = torch.sqrt(phi3+eps) * phi2
        rn   = phi2
        beta = torch.sqrt(phi4+eps) * phi2
        u1a  = u1 - vl * t;  u2a = u2 - vn * t
        d    = torch.sqrt((u1a*rl)**2 + (u2a*rn)**2 + (t*beta)**2 + eps)
        cov  = sig * torch.exp(-d)
        zero = (torch.abs(u1)<1e-9)&(torch.abs(u2)<1e-9)&(torch.abs(t)<1e-9)
        return torch.where(zero, cov+nug, cov)

    # =========================================================================
    # 2b.  Covariance — RAW (identity filter)
    # =========================================================================

    @staticmethod
    def cov_raw(u1, u2, t, params, delta1, delta2):
        """
        Cov_raw(u,v,τ) = C_X(u,v,τ).
        delta1/delta2 unused (kept for uniform API with cov_diff_2d).
        """
        dev = params.device
        u1 = u1.to(dev) if isinstance(u1,torch.Tensor) else torch.tensor(u1,device=dev,dtype=torch.float64)
        u2 = u2.to(dev) if isinstance(u2,torch.Tensor) else torch.tensor(u2,device=dev,dtype=torch.float64)
        t  = t.to(dev)  if isinstance(t, torch.Tensor) else torch.tensor(t, device=dev,dtype=torch.float64)
        return debiased_whittle_likelihood.cov_x_kernel(u1, u2, t, params)

    # =========================================================================
    # 2c.  Covariance — DIFF (2D separable filter)
    # =========================================================================

    @staticmethod
    def cov_diff_2d(u1, u2, t, params, delta1, delta2):
        """
        Cov_Z(u,v,τ) = ΣΣ h_ab·h_cd·C_X(u+(a-c)δ1, v+(b-d)δ2, τ)
        h: {(0,0):-1, (1,0):+1, (0,1):+1, (1,1):-1}

        Expanded = 4C_X(u,v,τ)
                 - 2[C_X(u-δ1,v,τ)+C_X(u+δ1,v,τ)]
                 - 2[C_X(u,v-δ2,τ)+C_X(u,v+δ2,τ)]
                 + C_X(u-δ1,v-δ2,τ)+C_X(u+δ1,v-δ2,τ)
                 + C_X(u-δ1,v+δ2,τ)+C_X(u+δ1,v+δ2,τ)
        """
        weights = {(0,0):-1., (1,0):1., (0,1):1., (1,1):-1.}
        dev = params.device
        out = torch.broadcast_shapes(
            u1.shape if isinstance(u1,torch.Tensor) else (),
            u2.shape if isinstance(u2,torch.Tensor) else (),
            t.shape  if isinstance(t, torch.Tensor) else ())
        cov  = torch.zeros(out, device=dev, dtype=torch.float64)
        u1d  = u1.to(dev) if isinstance(u1,torch.Tensor) else torch.tensor(u1,device=dev,dtype=torch.float64)
        u2d  = u2.to(dev) if isinstance(u2,torch.Tensor) else torch.tensor(u2,device=dev,dtype=torch.float64)
        td   = t.to(dev)  if isinstance(t, torch.Tensor) else torch.tensor(t, device=dev,dtype=torch.float64)
        for (a,b), wab in weights.items():
            for (c,d), wcd in weights.items():
                term = debiased_whittle_likelihood.cov_x_kernel(
                    u1d + (a-c)*delta1, u2d + (b-d)*delta2, td, params)
                if torch.isnan(term).any():
                    return torch.full_like(cov, float('nan'))
                cov += wab * wcd * term
        return cov

    # =========================================================================
    # 3.  cn_bar helpers — one per covariance type
    # =========================================================================

    @staticmethod
    def _cn_bar(cov_fn, u1, u2, t, params, n1, n2, taper_auto,
                delta1, delta2, q=None, r=None):
        """c_Y(u)·c_gn(u) using the supplied covariance function."""
        dev = params.device
        u1d = u1.to(dev) if isinstance(u1,torch.Tensor) else torch.tensor(u1,device=dev,dtype=torch.float64)
        u2d = u2.to(dev) if isinstance(u2,torch.Tensor) else torch.tensor(u2,device=dev,dtype=torch.float64)
        td  = t.to(dev)  if isinstance(t, torch.Tensor) else torch.tensor(t, device=dev,dtype=torch.float64)
        cov_val = cov_fn(u1d*delta1, u2d*delta2, td, params, delta1, delta2)
        idx1 = torch.clamp(n1-1+u1d.long(), 0, 2*n1-2)
        idx2 = torch.clamp(n2-1+u2d.long(), 0, 2*n2-2)
        if taper_auto.ndim == 4 and q is not None:
            tv = taper_auto[q, r, idx1, idx2]
        else:
            tv = taper_auto[idx1, idx2]
        if torch.isnan(cov_val).any() or torch.isnan(tv).any():
            return torch.full(torch.broadcast_shapes(cov_val.shape, tv.shape),
                              float('nan'), device=dev, dtype=torch.float64)
        return cov_val * tv

    # =========================================================================
    # 4.  Expected periodograms — one per covariance type
    # =========================================================================

    @staticmethod
    def _expected_periodogram(cov_fn, params, n1, n2, p_time, taper_auto, delta1, delta2):
        """
        E[I(ω)] = (1/(4π²)) FFT_2D[ Ã_n(u) ]   (aliasing sum, Lemma 2)
        Generic: controlled by cov_fn (cov_raw or cov_diff_2d).
        """
        dev = params.device if isinstance(params, torch.Tensor) else params[0].device
        pt  = torch.cat([p.to(dev) for p in params]) if isinstance(params,list) else params.to(dev)

        u1m, u2m = torch.meshgrid(torch.arange(n1,dtype=torch.float64,device=dev),
                                   torch.arange(n2,dtype=torch.float64,device=dev), indexing='ij')
        tl = torch.arange(p_time, dtype=torch.float64, device=dev)
        cn = torch.zeros((n1,n2,p_time,p_time), dtype=torch.complex128, device=dev)

        for q in range(p_time):
            for r in range(p_time):
                td = tl[q] - tl[r]
                _q = q if taper_auto.ndim==4 else None
                _r = r if taper_auto.ndim==4 else None

                def cb(du1, du2):
                    return debiased_whittle_likelihood._cn_bar(
                        cov_fn, du1, du2, td, pt, n1, n2, taper_auto, delta1, delta2, _q, _r)

                grid = cb(u1m, u2m) + cb(u1m-n1, u2m) + cb(u1m, u2m-n2) + cb(u1m-n1, u2m-n2)
                cn[:,:,q,r] = float('nan') if torch.isnan(grid).any() else grid.to(torch.complex128)

        if torch.isnan(cn).any():
            return torch.full((n1,n2,p_time,p_time), float('nan'), dtype=torch.complex128, device=dev)
        result_raw = torch.fft.fft2(cn, dim=(0,1)) * (1./(4.*cmath.pi**2))
        return (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

    @staticmethod
    def expected_periodogram_raw(params, n1, n2, p_time, taper_auto, delta1, delta2):
        """E[I_raw(ω)] using C_X directly (identity filter)."""
        return debiased_whittle_likelihood._expected_periodogram(
            debiased_whittle_likelihood.cov_raw,
            params, n1, n2, p_time, taper_auto, delta1, delta2)

    @staticmethod
    def expected_periodogram_diff(params, n1d, n2d, p_time, taper_auto_diff, delta1, delta2):
        """E[I_diff(ω)] using 2D-filter covariance cross-terms."""
        return debiased_whittle_likelihood._expected_periodogram(
            debiased_whittle_likelihood.cov_diff_2d,
            params, n1d, n2d, p_time, taper_auto_diff, delta1, delta2)

    # =========================================================================
    # 5.  Per-frequency likelihood terms   (helper, used in mixed loss)
    # =========================================================================

    @staticmethod
    def _likelihood_terms(I_exp, I_samp, p_time, device):
        """
        Returns (n1,n2) tensor:  log|f(ω)| + tr(f(ω)^{-1} I(ω))
        per spatial frequency.
        """
        eye = torch.eye(p_time, dtype=torch.complex128, device=device)
        dv  = torch.abs(I_exp.diagonal(dim1=-2, dim2=-1))
        dl  = max(dv.mean().item() if dv.numel()>0 else 1., 1e-9) * 1e-8
        dl  = max(dl, 1e-9)
        Is  = I_exp + eye * dl

        sign, logdet = torch.linalg.slogdet(Is)
        logdet = torch.where(sign.real > 1e-9, logdet.real,
                             torch.full(logdet.shape, 1e10, device=logdet.device, dtype=torch.float64))

        try:
            trace = torch.einsum('...ii->...', torch.linalg.solve(Is, I_samp)).real
        except torch.linalg.LinAlgError:
            return torch.full(I_exp.shape[:2], float('inf'), device=device, dtype=torch.float64)

        terms = logdet + trace
        terms = torch.nan_to_num(terms, nan=float('inf'), posinf=float('inf'))
        return terms  # (n1, n2) real

    # =========================================================================
    # 6.  Mixed-frequency Whittle loss
    # =========================================================================

    @staticmethod
    def whittle_likelihood_loss_mixed(params,
                                       I_samp_raw,  I_samp_diff,
                                       n1,  n2,
                                       n1d, n2d,
                                       p_time,
                                       taper_auto_raw,  taper_auto_diff,
                                       K1, K2,
                                       delta1=0.044, delta2=0.063):
        """
        L_mixed(θ) = Σ_{ω∈Ω_L} ℓ_raw(ω)  +  Σ_{ω∈Ω_H} ℓ_diff(ω)

        Ω_L = {k1≤K1, k2≤K2} \\ {(0,0)}    on raw  (n1 × n2)  grid
        Ω_H = {k1>K1 OR k2>K2}              on diff (n1d × n2d) grid

        Both pieces together cover the full spatial frequency range once.
        Loss is averaged over |Ω_L| + |Ω_H| terms.
        """
        dev = I_samp_raw.device
        pt  = params.to(dev)
        if torch.isnan(pt).any() or torch.isinf(pt).any():
            return torch.tensor(float('nan'), device=dev, dtype=torch.float64)

        # ── Expected periodograms ─────────────────────────────────────────────
        Ie_raw  = debiased_whittle_likelihood.expected_periodogram_raw(
            pt, n1, n2, p_time, taper_auto_raw.to(dev), delta1, delta2)
        Ie_diff = debiased_whittle_likelihood.expected_periodogram_diff(
            pt, n1d, n2d, p_time, taper_auto_diff.to(dev), delta1, delta2)

        if torch.isnan(Ie_raw).any() or torch.isnan(Ie_diff).any():
            return torch.tensor(float('nan'), device=dev, dtype=torch.float64)

        # ── Per-frequency likelihood terms ────────────────────────────────────
        terms_raw  = debiased_whittle_likelihood._likelihood_terms(
            Ie_raw,  I_samp_raw.to(dev),  p_time, dev)   # (n1,  n2)
        terms_diff = debiased_whittle_likelihood._likelihood_terms(
            Ie_diff, I_samp_diff.to(dev), p_time, dev)   # (n1d, n2d)

        # ── Frequency masks ───────────────────────────────────────────────────
        # Low-freq: from raw grid
        #   (a) rectangle {k1≤K1, k2≤K2}
        #   (b) entire k1=0 row  — 2D diff filter H(0,ω2)=(e^0-1)(·)=0,
        #       so these frequencies have zero expected power under diff for ALL θ.
        #       Must be handled by raw, where f_raw(0,ω2) = f_X(0,ω2) > 0.
        #   (c) entire k2=0 col  — same reason: H(ω1,0)=(·)(e^0-1)=0.
        #   DC (0,0) excluded from likelihood.
        low_mask = torch.zeros(n1, n2, dtype=torch.bool, device=dev)
        low_mask[:K1+1, :K2+1] = True   # rectangle
        low_mask[0, :]          = True   # k1=0 row  → raw (H_diff=0 there)
        low_mask[:, 0]          = True   # k2=0 col  → raw (H_diff=0 there)
        low_mask[0, 0]          = False  # DC always excluded

        # High-freq: from diff grid — interior only (k1>0 AND k2>0)
        #   |H(ω1,ω2)|² = 4sin²(ω1/2)·4sin²(ω2/2) > 0  for k1>0, k2>0.
        high_mask = torch.ones(n1d, n2d, dtype=torch.bool, device=dev)
        high_mask[:K1+1, :K2+1] = False  # exclude low-freq rectangle
        high_mask[0, :]          = False  # k1=0 row: H_diff=0, zero expected power
        high_mask[:, 0]          = False  # k2=0 col: H_diff=0, zero expected power

        # ── Sums ─────────────────────────────────────────────────────────────
        sum_low  = terms_raw[low_mask].sum()
        sum_high = terms_diff[high_mask].sum()
        n_total  = float(low_mask.sum() + high_mask.sum())

        if n_total < 1:
            return torch.tensor(float('inf'), device=dev, dtype=torch.float64)

        loss = (sum_low + sum_high) / n_total

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), device=dev, dtype=torch.float64)
        return loss

    # =========================================================================
    # 7.  Helpers
    # =========================================================================

    @staticmethod
    def get_printable_params(p_list):
        try:
            p = torch.cat([x.detach().clone().cpu() for x in p_list])
            phi2 = torch.exp(p[1]); phi3 = torch.exp(p[2]); phi4 = torch.exp(p[3])
            eps  = 1e-12
            sig  = torch.exp(p[0]) / (phi2+eps)
            rl   = 1./(torch.sqrt(phi3+eps)*phi2+eps)
            rn   = 1./(phi2+eps)
            rt   = rn / torch.sqrt(phi4+eps)
            nug  = torch.exp(p[6])
            return (f"sigmasq:{sig.item():.4f} rl:{rl.item():.4f} rn:{rn.item():.4f} "
                    f"rt:{rt.item():.4f} vl:{p[4].item():.4f} vn:{p[5].item():.4f} "
                    f"nug:{nug.item():.4f}")
        except Exception as e:
            return f"[err:{e}]"

    @staticmethod
    def get_raw_log_params(p_list):
        try:
            p = torch.cat([x.detach().clone().cpu() for x in p_list])
            return (f"lp1:{p[0].item():.3f} lp2:{p[1].item():.3f} lp3:{p[2].item():.3f} "
                    f"lp4:{p[3].item():.3f} vl:{p[4].item():.3f} vn:{p[5].item():.3f} "
                    f"ln:{p[6].item():.3f}")
        except Exception: return "[err]"

    # =========================================================================
    # 8.  L-BFGS optimizer loop — mixed loss
    # =========================================================================

    @staticmethod
    def run_lbfgs_mixed(params_list, optimizer,
                         I_samp_raw,  I_samp_diff,
                         n1,  n2,
                         n1d, n2d,
                         p_time,
                         taper_auto_raw,  taper_auto_diff,
                         K1, K2,
                         max_steps=5, device='cpu', grad_tol=1e-5):
        """
        L-BFGS loop minimising whittle_likelihood_loss_mixed.
        Returns (natural_str, phi_str, raw_str, best_loss, steps_completed).
        """
        best_params = [p.detach().clone() for p in params_list]
        best_loss   = float('inf')
        prev_loss   = float('inf')
        loss_tol    = 1e-12
        DELTA_LAT   = 0.044
        DELTA_LON   = 0.063

        Ir_dev  = I_samp_raw.to(device)
        Id_dev  = I_samp_diff.to(device)
        tar_dev = taper_auto_raw.to(device)
        tad_dev = taper_auto_diff.to(device)

        def closure():
            optimizer.zero_grad()
            pt = torch.cat(params_list)
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_mixed(
                pt, Ir_dev, Id_dev, n1, n2, n1d, n2d, p_time,
                tar_dev, tad_dev, K1, K2, DELTA_LAT, DELTA_LON)
            if torch.isnan(loss) or torch.isinf(loss):
                return loss
            loss.backward()
            if any(p.grad is not None and
                   (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                   for p in params_list):
                optimizer.zero_grad()
            return loss

        for i in range(max_steps):
            loss = optimizer.step(closure)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i+1}: loss NaN/Inf, stopping."); break

            cur = loss.item()
            if cur < best_loss:
                if not any(torch.isnan(p.data).any() or torch.isinf(p.data).any()
                           for p in params_list):
                    best_loss   = cur
                    best_params = [p.detach().clone() for p in params_list]

            mg = max((p.grad.abs().max().item()
                      for p in params_list if p.grad is not None), default=0.)
            print(f'--- Step {i+1}/{max_steps}  loss:{cur:.6f}  max|grad|:{mg:.2e}')
            print(f'    {debiased_whittle_likelihood.get_raw_log_params(params_list)}')

            if i > 2:
                if mg < grad_tol:
                    print(f"Converged (grad<{grad_tol}) step {i+1}"); break
                if abs(cur - prev_loss) < loss_tol:
                    print(f"Converged (Δloss<{loss_tol}) step {i+1}"); break
            prev_loss = cur

        print("--- Training Complete ---")
        fl = round(best_loss, 3) if best_loss != float('inf') else float('inf')
        return (debiased_whittle_likelihood.get_printable_params(best_params),
                debiased_whittle_likelihood.get_printable_params(best_params),
                debiased_whittle_likelihood.get_raw_log_params(best_params),
                fl, i+1)
