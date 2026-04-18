# Configuration
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"

# --- Standard Libraries ---
import sys
import os
import json
import time
import copy
import cmath
import pickle
import logging
import argparse

sys.path.append(gems_tco_path)

# --- Third-Party Libraries ---
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
from json import JSONEncoder

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import typer

import torch
import torch.optim as optim
import torch.fft
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from GEMS_TCO import configuration as config


# ─────────────────────────────────────────────────────────────────────────────
# full_vecc_dw_likelihoods  (unchanged from debiased_whittle_lat1d)
# ─────────────────────────────────────────────────────────────────────────────

class full_vecc_dw_likelihoods:
    def __init__(self, daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range):
        self.day_idx = day_idx
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.daily_aggregated_tensors = daily_aggregated_tensors
        self.daily_hourly_maps = daily_hourly_maps
        self.daily_aggregated_tensor = daily_aggregated_tensors[day_idx]
        self.daily_hourly_map = daily_hourly_maps[day_idx]
        self.params_list = [
            torch.tensor([val], dtype=torch.float64, requires_grad=True, device='cpu') for val in params_list
        ]
        self.params_tensor = torch.cat(self.params_list)


# ─────────────────────────────────────────────────────────────────────────────
# debiased_whittle_preprocess  — RAW (no spatial filter)
# ─────────────────────────────────────────────────────────────────────────────

class debiased_whittle_preprocess(full_vecc_dw_likelihoods):
    def __init__(self, daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range):
        super().__init__(daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range)

    def subset_tensor(self, df_tensor: torch.Tensor,
                      lat_s: float, lat_e: float,
                      lon_s: float, lon_e: float) -> torch.Tensor:
        lat_mask = (df_tensor[:, 0] >= lat_s) & (df_tensor[:, 0] <= lat_e)
        lon_mask = (df_tensor[:, 1] >= lon_s) & (df_tensor[:, 1] <= lon_e)
        return df_tensor[lat_mask & lon_mask].clone()

    def apply_raw_passthrough(self, df_tensor: torch.Tensor) -> torch.Tensor:
        """
        RAW baseline: no spatial filter.  Identity transfer function H(ω) = 1.

        Operations (per time slice):
          1. Validate that the tensor forms a complete lat × lon grid.
          2. Subtract the spatial mean of *observed* (non-NaN) cells from all
             observed cells.  This removes the DC component and makes the
             estimator consistent with the zero-mean stationary GP assumption
             without discarding any frequency-domain information.
          3. Return the full nlat × nlon grid tensor unchanged in shape.

        Why demean here instead of relying on the DC-term exclusion in the
        likelihood?  Both safeguards are kept (belt-and-suspenders), but
        demeaning first ensures that I(ω=0) does not receive spurious mass
        from a non-zero empirical mean, which can otherwise affect nearby
        frequencies through spectral leakage even after tapering.
        """
        if df_tensor.size(0) == 0:
            return torch.empty(0, 4, dtype=torch.float64)

        if df_tensor.dtype != torch.float64:
            df_tensor = df_tensor.to(torch.float64)

        # ── 1. Grid validation ────────────────────────────────────────────────
        unique_lats = torch.unique(df_tensor[:, 0])
        unique_lons = torch.unique(df_tensor[:, 1])
        lat_count, lon_count = unique_lats.size(0), unique_lons.size(0)

        if df_tensor.size(0) != lat_count * lon_count:
            raise ValueError(
                "Tensor size does not match grid dimensions. Must be a complete grid."
            )

        # ── 2. Per-time-slice spatial demeaning ───────────────────────────────
        vals = df_tensor[:, 2]
        valid_mask = ~torch.isnan(vals)
        df_out = df_tensor.clone()
        if valid_mask.sum() > 0:
            spatial_mean = vals[valid_mask].mean()
            df_out[valid_mask, 2] = vals[valid_mask] - spatial_mean

        # ── 3. Return full nlat × nlon grid (no size reduction) ───────────────
        return df_out

    def generate_spatially_filtered_days(self,
                                          lat_s: float, lat_e: float,
                                          lon_s: float, lon_e: float):
        """
        Subset each time slice to the region of interest and apply the raw
        passthrough (demean only, no spatial filter).
        Output grid: nlat × nlon  (unchanged, cf. (nlat-1)×nlon for lat1d).
        """
        tensors_to_aggregate = []

        for key, tensor in self.daily_hourly_map.items():
            subsetted = self.subset_tensor(tensor, lat_s, lat_e, lon_s, lon_e)
            if subsetted.size(0) > 0:
                try:
                    passthrough = self.apply_raw_passthrough(subsetted)
                    if passthrough.size(0) > 0:
                        tensors_to_aggregate.append(passthrough)
                except ValueError as e:
                    print(f"Skipping data chunk on day {self.day_idx+1} due to error: {e}")

        if tensors_to_aggregate:
            subsetted_aggregated_day = torch.cat(tensors_to_aggregate, dim=0)
        else:
            subsetted_aggregated_day = torch.empty(0, 4, dtype=torch.float64)
        return subsetted_aggregated_day


# ─────────────────────────────────────────────────────────────────────────────
# debiased_whittle_likelihood  — RAW covariance (identity filter)
# ─────────────────────────────────────────────────────────────────────────────

class debiased_whittle_likelihood:

    def __init__(self):
        pass

    # =========================================================================
    # 1. Tapering & Data Functions   (identical to debiased_whittle_lat1d)
    # =========================================================================

    @staticmethod
    def cgn_hamming(u, n1, n2):
        u1, u2 = u
        device = u1.device if isinstance(u1, torch.Tensor) else (
            u2.device if isinstance(u2, torch.Tensor) else torch.device('cpu'))
        u1_tensor = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_tensor = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        n1_eff = float(n1) if n1 > 0 else 1.0
        n2_eff = float(n2) if n2 > 0 else 1.0
        hamming1 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u1_tensor / n1_eff)
        hamming2 = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * u2_tensor / n2_eff)
        return hamming1 * hamming2

    @staticmethod
    def calculate_taper_autocorrelation_fft(taper_grid, n1, n2, device):
        taper_grid = taper_grid.to(device)
        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            print("Warning: Sum of squared taper weights (H) is near zero.")
            return torch.zeros((2*n1-1, 2*n2-1), device=device, dtype=taper_grid.dtype)
        N1, N2 = 2 * n1 - 1, 2 * n2 - 1
        taper_fft = torch.fft.fft2(taper_grid, s=(N1, N2))
        power_spectrum = torch.abs(taper_fft)**2
        autocorr_unnormalized = torch.fft.ifft2(power_spectrum).real
        autocorr_shifted = torch.fft.fftshift(autocorr_unnormalized)
        c_gn_grid = autocorr_shifted / (H + 1e-12)
        return c_gn_grid

    @staticmethod
    def generate_Jvector_tapered_mv(tensor_list, tapering_func, lat_col, lon_col, val_col, device):
        """
        Tapered DFT with per-variate normalization and obs_masks.
        (Identical to debiased_whittle_lat1d — no filter applied here.)
        """
        p_time = len(tensor_list)
        if p_time == 0:
            return torch.empty(0,0,0,device=device,dtype=torch.complex128), 0, 0, 0, None, None

        valid_tensors = [t for t in tensor_list if t.numel() > 0 and t.shape[1] > max(lat_col, lon_col, val_col)]
        if not valid_tensors:
            return torch.empty(0,0,0,device=device,dtype=torch.complex128), 0, 0, 0, None, None

        try:
            all_lats_cpu = torch.cat([t[:, lat_col] for t in valid_tensors])
            all_lons_cpu = torch.cat([t[:, lon_col] for t in valid_tensors])
        except IndexError:
            return torch.empty(0,0,0,device=device,dtype=torch.complex128), 0, 0, 0, None, None

        all_lats_cpu = all_lats_cpu[~torch.isnan(all_lats_cpu) & ~torch.isinf(all_lats_cpu)]
        all_lons_cpu = all_lons_cpu[~torch.isnan(all_lons_cpu) & ~torch.isinf(all_lons_cpu)]
        if all_lats_cpu.numel() == 0 or all_lons_cpu.numel() == 0:
            return torch.empty(0,0,0,device=device,dtype=torch.complex128), 0, 0, 0, None, None

        unique_lats_cpu, unique_lons_cpu = torch.unique(all_lats_cpu), torch.unique(all_lons_cpu)
        n1, n2 = len(unique_lats_cpu), len(unique_lons_cpu)
        lat_map = {lat.item(): i for i, lat in enumerate(unique_lats_cpu)}
        lon_map = {lon.item(): i for i, lon in enumerate(unique_lons_cpu)}

        u1_mesh_cpu, u2_mesh_cpu = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64),
            torch.arange(n2, dtype=torch.float64), indexing='ij')
        taper_grid = tapering_func((u1_mesh_cpu, u2_mesh_cpu), n1, n2).to(device)

        fft_results = []
        obs_masks_list = []
        for tensor in tensor_list:
            data_grid = torch.zeros((n1, n2), dtype=torch.float64, device=device)
            obs_mask  = torch.zeros((n1, n2), dtype=torch.bool,    device=device)
            if tensor.numel() > 0 and tensor.shape[1] > max(lat_col, lon_col, val_col):
                for row in tensor:
                    lat_item, lon_item = row[lat_col].item(), row[lon_col].item()
                    if not (np.isnan(lat_item) or np.isnan(lon_item)):
                        i = lat_map.get(lat_item)
                        j = lon_map.get(lon_item)
                        if i is not None and j is not None:
                            val = row[val_col]
                            val_num = val.item() if isinstance(val, torch.Tensor) else val
                            if not np.isnan(val_num) and not np.isinf(val_num):
                                data_grid[i, j] = val_num
                                obs_mask[i, j]  = True

            data_grid_tapered = data_grid * taper_grid
            if torch.isnan(data_grid_tapered).any() or torch.isinf(data_grid_tapered).any():
                data_grid_tapered = torch.nan_to_num(data_grid_tapered, nan=0.0, posinf=0.0, neginf=0.0)
            fft_results.append(torch.fft.fft2(data_grid_tapered))
            obs_masks_list.append(obs_mask)

        if not fft_results:
            return torch.empty(0,0,0,device=device,dtype=torch.complex128), n1, n2, 0, taper_grid, None

        obs_masks = torch.stack(obs_masks_list, dim=0)

        normed = []
        for q_idx, (fft_q, obs_q) in enumerate(zip(fft_results, obs_masks_list)):
            H_q = (taper_grid * obs_q.to(taper_grid.dtype)).pow(2).sum()
            norm_q = (torch.sqrt(1.0 / H_q) / (2.0 * cmath.pi)).to(device) if H_q >= 1e-12 \
                     else torch.tensor(0.0, device=device, dtype=torch.float64)
            normed.append(fft_q * norm_q)
        result = torch.stack(normed, dim=2).to(device)
        return result, n1, n2, p_time, taper_grid, obs_masks

    @staticmethod
    def calculate_taper_autocorrelation_multivariate(taper_grid, obs_masks, n1, n2, device):
        """
        c_{g,n}^{(qr)} for every pair (q,r).
        (Identical to debiased_whittle_lat1d — taper structure unchanged.)
        """
        taper_grid = taper_grid.to(device)
        obs_masks  = obs_masks.to(device)
        p   = obs_masks.shape[0]
        N1, N2 = 2 * n1 - 1, 2 * n2 - 1

        g_all = taper_grid.unsqueeze(0) * obs_masks.to(taper_grid.dtype)
        H_all = (g_all ** 2).sum(dim=(1, 2))

        g_ffts = torch.fft.fft2(g_all, s=(N1, N2))

        result = torch.zeros((p, p, N1, N2), device=device, dtype=taper_grid.dtype)
        for q in range(p):
            for r in range(p):
                cross = torch.fft.ifft2(g_ffts[q] * g_ffts[r].conj()).real
                cross_shifted = torch.fft.fftshift(cross)
                denom = torch.sqrt(H_all[q] * H_all[r]) + 1e-12
                result[q, r] = cross_shifted / denom

        return result

    @staticmethod
    def calculate_sample_periodogram_vectorized(J_vector_tensor):
        if torch.isnan(J_vector_tensor).any() or torch.isinf(J_vector_tensor).any():
            print("Warning: NaN/Inf detected in J_vector_tensor input.")
            n1, n2, p = J_vector_tensor.shape
            return torch.full((n1, n2, p, p), float('nan'), dtype=torch.complex128, device=J_vector_tensor.device)
        J_col = J_vector_tensor.unsqueeze(-1)
        J_row_conj = J_vector_tensor.unsqueeze(-2).conj()
        return J_col @ J_row_conj

    # =========================================================================
    # 2. Covariance Functions — RAW (identity spatial filter)
    # =========================================================================

    @staticmethod
    def cov_x_spatiotemporal_model_kernel(u1, u2, t, params):
        """
        Autocovariance of X: C_X(u1, u2, t) with 7-parameter spatio-temporal kernel.
        u1, u2: physical lags (already scaled by deltas).
        t: physical time lag.
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev  = t.to(device)  if isinstance(t,  torch.Tensor) else torch.tensor(t,  device=device, dtype=torch.float64)

        if torch.isnan(params).any() or torch.isinf(params).any():
            out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
            return torch.full(out_shape, float('nan'), device=device, dtype=torch.float64)

        phi1      = torch.exp(params[0])
        phi2      = torch.exp(params[1])   # range_lon_inv
        phi3      = torch.exp(params[2])   # (range_lon / range_lat)^2
        phi4      = torch.exp(params[3])   # beta^2
        advec_lat = params[4]
        advec_lon = params[5]
        nugget    = torch.exp(params[6])

        epsilon = 1e-12
        sigmasq          = phi1 / (phi2 + epsilon)
        range_lon_inv    = phi2
        range_lat_inv    = torch.sqrt(phi3 + epsilon) * phi2
        beta_scaled_inv  = torch.sqrt(phi4 + epsilon) * phi2

        u1_adv = u1_dev - advec_lat * t_dev
        u2_adv = u2_dev - advec_lon * t_dev

        dist_sq  = (u1_adv * range_lat_inv).pow(2) + \
                   (u2_adv * range_lon_inv).pow(2) + \
                   (t_dev  * beta_scaled_inv).pow(2)
        distance = torch.sqrt(dist_sq + epsilon)

        cov_smooth = sigmasq * torch.exp(-distance)

        is_zero_lag = (torch.abs(u1_dev) < 1e-9) & \
                      (torch.abs(u2_dev) < 1e-9) & \
                      (torch.abs(t_dev)  < 1e-9)
        final_cov = torch.where(is_zero_lag, cov_smooth + nugget, cov_smooth)

        if torch.isnan(final_cov).any():
            print("Warning: NaN detected in cov_x_spatiotemporal_model_kernel output.")
        return final_cov

    @staticmethod
    def cov_spatial_difference(u1, u2, t, params, delta1, delta2):
        """
        RAW covariance: Cov(X(s,t_q), X(s+u,t_r)) = C_X(u1, u2, t).

        No spatial filter is applied (identity H(ω) = 1), so the covariance of
        the observed field equals the underlying GP covariance directly.

        delta1, delta2 are kept for API compatibility with the lat1d / 2d_conv
        modules but are intentionally *not* used here — there are no filter
        cross-terms to add (cf. lat1d: 2C_X - C_X(u-δ1) - C_X(u+δ1)).
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev  = t.to(device)  if isinstance(t,  torch.Tensor) else torch.tensor(t,  device=device, dtype=torch.float64)

        cov = debiased_whittle_likelihood.cov_x_spatiotemporal_model_kernel(
            u1_dev, u2_dev, t_dev, params)

        if torch.isnan(cov).any():
            print("Warning: NaN in cov_spatial_difference (raw) output.")
        return cov

    @staticmethod
    def cn_bar_tapered(u1, u2, t, params, n1, n2, taper_autocorr_grid,
                       delta1, delta2, q_idx=None, r_idx=None):
        """
        c_Y(u) * c_gn(u)  for the RAW (no-filter) model.

        u1, u2: grid-index lags  →  physical lags via delta1, delta2.
        Uses cov_spatial_difference (raw identity) as covariance function.
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev  = t.to(device)  if isinstance(t,  torch.Tensor) else torch.tensor(t,  device=device, dtype=torch.float64)

        lag_u1 = u1_dev * delta1
        lag_u2 = u2_dev * delta2

        cov_value = debiased_whittle_likelihood.cov_spatial_difference(
            lag_u1, lag_u2, t_dev, params, delta1, delta2)

        u1_idx = u1_dev.long()
        u2_idx = u2_dev.long()
        idx1 = torch.clamp(n1 - 1 + u1_idx, 0, 2 * n1 - 2)
        idx2 = torch.clamp(n2 - 1 + u2_idx, 0, 2 * n2 - 2)

        if taper_autocorr_grid.ndim == 4 and q_idx is not None and r_idx is not None:
            taper_val = taper_autocorr_grid[q_idx, r_idx, idx1, idx2]
        else:
            taper_val = taper_autocorr_grid[idx1, idx2]

        if torch.isnan(cov_value).any() or torch.isnan(taper_val).any():
            out_shape = torch.broadcast_shapes(cov_value.shape, taper_val.shape)
            return torch.full(out_shape, float('nan'), device=device, dtype=torch.float64)

        result = cov_value * taper_val
        if torch.isnan(result).any():
            print("Warning: NaN in cn_bar_tapered output.")
        return result

    # =========================================================================
    # 3. Expected Periodogram   (identical structure to debiased_whittle_lat1d)
    # =========================================================================

    @staticmethod
    def expected_periodogram_fft_tapered(params, n1, n2, p_time,
                                          taper_autocorr_grid, delta1, delta2):
        """
        E[I(ω)] = (1/(4π²)) * FFT_2D[ Σ_{q,r} c̃_n^{(qr)}(u) ]

        For the raw model, c̃_n^{(qr)} uses C_X directly (no filter cross-terms).
        Aliasing sum follows Lemma 2 of Guillaumin et al. (2022).
        """
        device = params.device if isinstance(params, torch.Tensor) else params[0].device
        if isinstance(params, list):
            params_tensor = torch.cat([p.to(device) for p in params])
        else:
            params_tensor = params.to(device)

        u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
        u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
        u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing='ij')

        t_lags = torch.arange(p_time, dtype=torch.float64, device=device)

        rows = []
        has_nan = False
        for q in range(p_time):
            cols = []
            for r in range(p_time):
                t_diff = t_lags[q] - t_lags[r]
                _q = q if taper_autocorr_grid.ndim == 4 else None
                _r = r if taper_autocorr_grid.ndim == 4 else None

                term1 = debiased_whittle_likelihood.cn_bar_tapered(
                    u1_mesh,      u2_mesh,      t_diff, params_tensor,
                    n1, n2, taper_autocorr_grid, delta1, delta2, _q, _r)
                term2 = debiased_whittle_likelihood.cn_bar_tapered(
                    u1_mesh - n1, u2_mesh,      t_diff, params_tensor,
                    n1, n2, taper_autocorr_grid, delta1, delta2, _q, _r)
                term3 = debiased_whittle_likelihood.cn_bar_tapered(
                    u1_mesh,      u2_mesh - n2, t_diff, params_tensor,
                    n1, n2, taper_autocorr_grid, delta1, delta2, _q, _r)
                term4 = debiased_whittle_likelihood.cn_bar_tapered(
                    u1_mesh - n1, u2_mesh - n2, t_diff, params_tensor,
                    n1, n2, taper_autocorr_grid, delta1, delta2, _q, _r)

                tilde_cn_grid_qr = term1 + term2 + term3 + term4

                if torch.isnan(tilde_cn_grid_qr).any():
                    has_nan = True
                    cols.append(torch.zeros(n1, n2, dtype=torch.complex128, device=device))
                else:
                    cols.append(tilde_cn_grid_qr.to(torch.complex128))
            rows.append(torch.stack(cols, dim=-1))  # (n1, n2, p_time)
        tilde_cn_tensor = torch.stack(rows, dim=-2)  # (n1, n2, p_time, p_time)

        if has_nan:
            print("Warning: NaN detected in tilde_cn_tensor before FFT.")
            return torch.full((n1, n2, p_time, p_time), float('nan'),
                              dtype=torch.complex128, device=device)

        fft_result = torch.fft.fft2(tilde_cn_tensor, dim=(0, 1))
        result_raw = fft_result * (1.0 / (4.0 * cmath.pi**2))
        result = (result_raw + result_raw.conj().transpose(-1, -2)) / 2.0

        if torch.isnan(result.real).any():
            print("Warning: NaN in expected_periodogram_fft_tapered output.")
        return result

    # =========================================================================
    # 4. Likelihood — averaged, DC term excluded
    #    Note: DC exclusion (ω=(0,0)) is kept as an additional safeguard
    #    alongside per-slice demeaning in apply_raw_passthrough.
    # =========================================================================

    @staticmethod
    def whittle_likelihood_loss_tapered(params, I_sample, n1, n2, p_time,
                                         taper_autocorr_grid, delta1, delta2):
        device = I_sample.device
        params_tensor = params.to(device)

        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print("Warning: NaN/Inf in input parameters.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            print("Warning: NaN/Inf in expected periodogram.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        eye_matrix = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals = torch.abs(I_expected.diagonal(dim1=-2, dim2=-1))
        mean_diag_abs = diag_vals.mean().item() if diag_vals.numel() > 0 and not torch.isnan(diag_vals).all() else 1.0
        diag_load = max(mean_diag_abs * 1e-8, 1e-9)
        I_expected_stable = I_expected + eye_matrix * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_expected_stable)
        if torch.any(sign.real <= 1e-9):
            log_det_term = torch.where(sign.real > 1e-9, logabsdet.real,
                                       torch.tensor(1e10, device=device, dtype=torch.float64))
        else:
            log_det_term = logabsdet.real

        if torch.isnan(I_sample).any() or torch.isinf(I_sample).any():
            print("Warning: NaN/Inf in I_sample.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        try:
            solved_term = torch.linalg.solve(I_expected_stable, I_sample)
            trace_term  = torch.einsum('...ii->...', solved_term).real
        except torch.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError: {e}.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(trace_term).any() or torch.isinf(trace_term).any():
            print("Warning: NaN/Inf in trace term.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        likelihood_terms = log_det_term + trace_term
        if torch.isnan(likelihood_terms).any():
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        total_sum = torch.sum(likelihood_terms)
        # Exclude ω=(0,0): belt-and-suspenders alongside per-slice demeaning.
        dc_term = likelihood_terms[0, 0] if (n1 > 0 and n2 > 0) else \
                  torch.tensor(0.0, device=device, dtype=torch.float64)
        if torch.isnan(dc_term).any() or torch.isinf(dc_term).any():
            dc_term = torch.tensor(0.0, device=device, dtype=torch.float64)

        sum_loss = total_sum - dc_term if (n1 > 1 or n2 > 1) else total_sum

        num_terms = n1 * n2 - 1
        avg_loss  = sum_loss / num_terms if num_terms > 0 else sum_loss

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print("Warning: NaN/Inf in final loss.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        return avg_loss

    # =========================================================================
    # 5. Helpers & Training Loop   (identical to debiased_whittle_lat1d)
    # =========================================================================

    @staticmethod
    def get_printable_params_7param(p_list):
        valid_tensors = [p for p in p_list if isinstance(p, torch.Tensor)]
        if not valid_tensors: return "Invalid params_list"
        p_cat = torch.cat([p.detach().clone().cpu() for p in valid_tensors])
        if len(p_cat) != 7:
            return f"Expected 7 params, got {len(p_cat)}."
        if torch.isnan(p_cat).any() or torch.isinf(p_cat).any():
            return "[NaN/Inf in log_params]"
        try:
            phi1 = torch.exp(p_cat[0]); phi2 = torch.exp(p_cat[1])
            phi3 = torch.exp(p_cat[2]); phi4 = torch.exp(p_cat[3])
            advec_lat = p_cat[4]; advec_lon = p_cat[5]; nugget = torch.exp(p_cat[6])
            eps = 1e-12
            sigmasq    = phi1 / (phi2 + eps)
            range_lon  = 1.0 / (phi2 + eps)
            range_lat  = 1.0 / (torch.sqrt(phi3 + eps) * phi2 + eps)
            range_time = range_lon / torch.sqrt(phi4 + eps)
            return (f"sigmasq: {sigmasq.item():.4f}, range_lat: {range_lat.item():.4f}, "
                    f"range_lon: {range_lon.item():.4f}, range_time: {range_time.item():.4f}, "
                    f"advec_lat: {advec_lat.item():.4f}, advec_lon: {advec_lon.item():.4f}, "
                    f"nugget: {nugget.item():.4f}")
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def get_phi_params_7param(log_params_list):
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            phi1 = torch.exp(p_cat[0]); phi2 = torch.exp(p_cat[1])
            phi3 = torch.exp(p_cat[2]); phi4 = torch.exp(p_cat[3])
            return (f"phi1: {phi1.item():.4f}, phi2: {phi2.item():.4f}, "
                    f"phi3: {phi3.item():.4f}, phi4: {phi4.item():.4f}, "
                    f"advec_lat: {p_cat[4].item():.4f}, advec_lon: {p_cat[5].item():.4f}, "
                    f"nugget: {torch.exp(p_cat[6]).item():.4f}")
        except Exception:
            return "[Error in reparam conversion]"

    @staticmethod
    def get_raw_log_params_7param(log_params_list):
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            return (f"log_phi1: {p_cat[0].item():.4f}, log_phi2: {p_cat[1].item():.4f}, "
                    f"log_phi3: {p_cat[2].item():.4f}, log_phi4: {p_cat[3].item():.4f}, "
                    f"advec_lat: {p_cat[4].item():.4f}, advec_lon: {p_cat[5].item():.4f}, "
                    f"log_nugget: {p_cat[6].item():.4f}")
        except Exception:
            return "[Error in raw param conversion]"

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                           taper_autocorr_grid, max_steps=5, device='cpu', grad_tol=1e-5):
        """L-BFGS training loop (identical to debiased_whittle_lat1d)."""
        best_params_state = [p.detach().clone() for p in params_list]
        steps_completed   = 0
        DELTA_LAT, DELTA_LON = 0.044, 0.063

        loss_tol  = 1e-12
        best_loss = float('inf')
        prev_loss = float('inf')

        I_sample_dev          = I_sample.to(device)
        taper_autocorr_dev    = taper_autocorr_grid.to(device)

        def closure():
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list)
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time,
                taper_autocorr_dev, DELTA_LAT, DELTA_LON)
            if torch.isnan(loss) or torch.isinf(loss):
                return loss
            loss.backward()
            nan_grad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in params_list)
            if nan_grad:
                optimizer.zero_grad()
            return loss

        for i in range(max_steps):
            steps_completed = i + 1
            loss = optimizer.step(closure)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i+1}/{max_steps}: Loss NaN/Inf. Stopping.")
                break

            cur = loss.item()
            if cur < best_loss:
                if not any(torch.isnan(p.data).any() or torch.isinf(p.data).any() for p in params_list):
                    best_loss = cur
                    best_params_state = [p.detach().clone() for p in params_list]

            max_grad = 0.0
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        max_grad = max(max_grad, p.grad.abs().item())

            print(f'--- Step {i+1}/{max_steps} ---')
            print(f' Loss: {cur:.6f} | Max Grad: {max_grad:.6e}')
            print(f'  Params (Raw Log): {debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}')

            if i > 2:
                if max_grad < grad_tol:
                    print(f"\n--- Converged (max|grad| < {grad_tol}) at step {i+1} ---")
                    break
                if abs(cur - prev_loss) < loss_tol:
                    print(f"\n--- Converged (loss_change < {loss_tol}) at step {i+1} ---")
                    break

            prev_loss = cur

        print("\n--- Training Complete ---")
        if best_params_state is None:
            return None, None, None, None, steps_completed

        final_loss = round(best_loss, 3) if best_loss != float('inf') else float('inf')
        return (debiased_whittle_likelihood.get_printable_params_7param(best_params_state),
                debiased_whittle_likelihood.get_phi_params_7param(best_params_state),
                debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state),
                final_loss, steps_completed)
