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

# Path configuration (only run once)
sys.path.append(gems_tco_path)

# --- Third-Party Libraries ---
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
from json import JSONEncoder

# Data manipulation and analysis
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import typer

# Torch and Numerical Libraries
import torch
import torch.optim as optim
import torch.fft
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt 

# --- Custom (GEMS_TCO) Imports ---
from GEMS_TCO import kernels_reparam_space_time 
from GEMS_TCO import data_preprocess, data_preprocess as dmbh
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2


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
            torch.tensor([val], dtype=torch.float64, requires_grad=True, device= 'cpu') for val in params_list
        ]
        self.params_tensor = torch.cat(self.params_list)

    def initiate_model_instance_vecchia(self, v, nns_map, mm_cond_number, nheads):
        self.model_instance = kernels_reparam_space_time.fit_vecchia_adams(
                smooth = v,
                input_map = self.daily_hourly_map,
                aggregated_data = self.daily_aggregated_tensor,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = nheads
            )
    def compute_full_likelihoods(self):
        full_likelihood = self.model_instance.full_likelihood_avg(
            params = self.params_tensor, 
            input_data = self.daily_aggregated_tensor, 
            y = self.daily_aggregated_tensor[:,2], 
            covariance_function = self.model_instance.matern_cov_aniso_STABLE_log_reparam
        )
        return full_likelihood
    
    def compute_vecchia_nll(self):
        cov_map = self.model_instance.cov_structure_saver(self.params_tensor, self.model_instance.matern_cov_aniso_STABLE_log_reparam)
        vecchia_nll = self.model_instance.vecchia_space_time_fullbatch( # Change this to your chosen Vecchia implementation
        params = self.params_tensor, 
        covariance_function = self.model_instance.matern_cov_aniso_STABLE_log_reparam, 
        cov_map = cov_map # Assuming cov_map is precomputed or computed internally
        )
        return vecchia_nll
    

    def likelihood_wrapper(self,daily_aggregated_tensors_dw, daily_hourly_maps_dw):
        full_nll = self.compute_full_likelihoods()
        vecc_nll = self.compute_vecchia_nll()
 

        # --- Debiased Whittle Configuration ---
        dwl = debiased_whittle_likelihood()

        TAPERING_FUNC = dwl.cgn_hamming # Use Hamming taper
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {DEVICE}")

        DELTA_LAT, DELTA_LON = 0.044, 0.063 

        LAT_COL, LON_COL = 0, 1
        VAL_COL = 2 # Spatially differenced value
        TIME_COL = 3


        db = debiased_whittle_preprocess(daily_aggregated_tensors_dw, daily_hourly_maps_dw, day_idx=0, params_list=self.params_list, lat_range=self.lat_range, lon_range=self.lon_range)
        subsetted_aggregated_day = db.generate_spatially_filtered_days(self.lat_range[0],self.lat_range[1],self.lon_range[0],self.lon_range[1])
        
        #(N-1)*(M-1) grid after differencing

        #print(f'subsetted_aggregated_day.shape: {subsetted_aggregated_day.shape}')
        
        ####
        cur_df = subsetted_aggregated_day
        unique_times = torch.unique(cur_df[:, TIME_COL])
        time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

        # --- 1. Pre-compute J-vector, Taper Grid, and Taper Autocorrelation ---
        #print("Pre-computing J-vector (Hamming taper)...")
        J_vec, n1, n2, p, taper_grid = dwl.generate_Jvector_tapered( 
            time_slices_list,
            tapering_func=TAPERING_FUNC, 
            lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL,
            device=DEVICE
        )

        I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
        taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)


        params_list = [
            Parameter(torch.tensor([val], dtype=torch.float64, device=DEVICE), requires_grad=True)
            for val in self.params_list
        ]

        dwnll,n1,n2 = dwl.whittle_likelihood_loss_tapered_sum(
            params=torch.cat(params_list),
            I_sample=I_sample,
            n1=n1,
            n2=n2,
            p_time=p,
            taper_autocorr_grid=taper_autocorr_grid,
            delta1=DELTA_LAT,
            delta2=DELTA_LON
        )

        outputs = [full_nll, vecc_nll, dwnll, n1, n2]
        #outputs = [1, 1,  dwnll, n1, n2]
        return outputs

class debiased_whittle_preprocess(full_vecc_dw_likelihoods):
    def __init__(self, daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range):
        super().__init__(daily_aggregated_tensors, daily_hourly_maps, day_idx, params_list, lat_range, lon_range)

    def subset_tensor(self,df_tensor: torch.Tensor, lat_s: float, lat_e: float, lon_s: float,lon_e: float) -> torch.Tensor:
        """Subsets a tensor to a specific lat/lon range."""
        #lat_mask = (df_tensor[:, 0] >= -5) & (df_tensor[:, 0] <= 6.3)
        #lon_mask = (df_tensor[:, 1] >= 118) & (df_tensor[:, 1] <= 134.2)
        lat_mask = (df_tensor[:, 0] >= lat_s) & (df_tensor[:, 0] <= lat_e)
        lon_mask = (df_tensor[:, 1] >= lon_s) & (df_tensor[:, 1] <= lon_e)

        df_sub = df_tensor[lat_mask & lon_mask].clone()
        return df_sub

    def apply_first_difference_2d_tensor(self, df_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies a 2D first-order difference filter using convolution.
        This approximates Z(s) = [X(s+d_lat) - X(s)] + [X(s+d_lon) - X(s)].
        """
        if df_tensor.size(0) == 0:
            return torch.empty(0, 4, dtype=torch.float64)
        
        # ✅ FIX: Force input to float64 immediately
        if df_tensor.dtype != torch.float64:
            df_tensor = df_tensor.to(torch.float64)

        # 1. Get grid dimensions and validate
        unique_lats = torch.unique(df_tensor[:, 0])
        unique_lons = torch.unique(df_tensor[:, 1])
        lat_count, lon_count = unique_lats.size(0), unique_lons.size(0)

        if df_tensor.size(0) != lat_count * lon_count:
            raise ValueError("Tensor size does not match grid dimensions. Must be a complete grid.")
        if lat_count < 2 or lon_count < 2:
            return torch.empty(0, 4)

        # 2. Reshape data and define the correct kernel
        ozone_data = df_tensor[:, 2].reshape(1, 1, lat_count, lon_count)
        
        # ✅ CORRECT KERNEL: This kernel results in the standard first-order difference:
        # Z(i,j) = X(i+1,j) + X(i,j+1) - 2*X(i,j)
        # Note: F.conv2d in PyTorch actually performs cross-correlation. To get a true
        # convolution result, the kernel would need to be flipped. However, for a 
        # forward difference operator, defining the kernel for cross-correlation is more direct.
        # The kernel below is designed for cross-correlation to achieve the desired differencing.
        diff_kernel = torch.tensor([[[[-2., 1.],
                                    [ 1., 0.]]]], dtype=torch.float64)
        #diff_kernel = torch.tensor([[[[-1, 1],
        #                             [1, -1]]]], dtype=torch.float64)

        # 3. Apply convolution (which acts as cross-correlation)
        filtered_grid = F.conv2d(ozone_data, diff_kernel, padding='valid').squeeze()

        # 4. Determine coordinates for the new, smaller grid
        # The new grid corresponds to the anchor points of the kernel
        new_lats = unique_lats[:-1]
        new_lons = unique_lons[:-1]

        # 5. Reconstruct the output tensor
        new_lat_grid, new_lon_grid = torch.meshgrid(new_lats, new_lons, indexing='ij')
        filtered_values = filtered_grid.flatten()
        time_value = df_tensor[0, 3].repeat(filtered_values.size(0))

        new_tensor = torch.stack([
            new_lat_grid.flatten(),
            new_lon_grid.flatten(),
            filtered_values,
            time_value
        ], dim=1)
        
        return new_tensor

    def generate_spatially_filtered_days(self,lat_s: float, lat_e: float, lon_s: float,  lon_e: float):
        tensors_to_aggregate = []
  
        for key, tensor in self.daily_hourly_map.items():
            subsetted = self.subset_tensor(tensor, lat_s, lat_e, lon_s, lon_e)
            if subsetted.size(0) > 0:
                try:
                    diff_applied = self.apply_first_difference_2d_tensor(subsetted)
                    if diff_applied.size(0) > 0:
                        tensors_to_aggregate.append(diff_applied)
                except ValueError as e:
                    print(f"Skipping data chunk on day {self.day_idx+1} due to error: {e}")

        if tensors_to_aggregate:
            subsetted_aggregated_day= torch.cat(tensors_to_aggregate, dim=0)
            subsetted_aggregated_day = subsetted_aggregated_day
        return subsetted_aggregated_day
    


class debiased_whittle_likelihood: # (full_vecc_dw_likelihoods):
    
    # NOTE: The __init__ was empty. If it needs to call super(), 
    # it should be added back. I'm keeping it as you provided.
    def __init__(self):
        pass
    
    # =========================================================================
    # 1. Tapering & Data Functions
    # =========================================================================
    @staticmethod
    def cgn_hamming(u, n1, n2):
        """Computes a 2D Hamming window."""
        u1, u2 = u
        device = u1.device if isinstance(u1, torch.Tensor) else (u2.device if isinstance(u2, torch.Tensor) else torch.device('cpu'))
        u1_tensor = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_tensor = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        n1_eff = float(n1) if n1 > 0 else 1.0
        n2_eff = float(n2) if n2 > 0 else 1.0
        hamming1 = 0.54 + 0.46 * torch.cos(2.0 * torch.pi * u1_tensor / n1_eff)
        hamming2 = 0.54 + 0.46 * torch.cos(2.0 * torch.pi * u2_tensor / n2_eff)
        return hamming1 * hamming2

    @staticmethod
    def calculate_taper_autocorrelation_fft(taper_grid, n1, n2, device):
        """
        Computes the normalized taper autocorrelation function c_gn(u) using FFT.
        """
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
    def generate_Jvector_tapered(tensor_list, tapering_func, lat_col, lon_col, val_col, device):
        """
        Generates J-vector for a single component using the specified taper,
        placing result on device.
        """
        # --- 💥 REVISED: p_time is the number of time points 💥 ---
        p_time = len(tensor_list)
        if p_time == 0: return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        valid_tensors = [t for t in tensor_list if t.numel() > 0 and t.shape[1] > max(lat_col, lon_col, val_col)]
        if not valid_tensors:
            print("Warning: No valid tensors found in tensor_list.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        try:
            all_lats_cpu = torch.cat([t[:, lat_col] for t in valid_tensors])
            all_lons_cpu = torch.cat([t[:, lon_col] for t in valid_tensors])
        except IndexError:
            print(f"Error: Invalid column index. Check tensor shapes.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        all_lats_cpu = all_lats_cpu[~torch.isnan(all_lats_cpu) & ~torch.isinf(all_lats_cpu)]
        all_lons_cpu = all_lons_cpu[~torch.isnan(all_lons_cpu) & ~torch.isinf(all_lons_cpu)]
        if all_lats_cpu.numel() == 0 or all_lons_cpu.numel() == 0:
            print("Warning: No valid coordinates after NaN/Inf filtering.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        unique_lats_cpu, unique_lons_cpu = torch.unique(all_lats_cpu), torch.unique(all_lons_cpu)
        n1, n2 = len(unique_lats_cpu), len(unique_lons_cpu)
        if n1 == 0 or n2 == 0:
            print("Warning: Grid dimensions are zero.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), 0, 0, 0, None

        lat_map = {lat.item(): i for i, lat in enumerate(unique_lats_cpu)}
        lon_map = {lon.item(): i for i, lon in enumerate(unique_lons_cpu)}

        u1_mesh_cpu, u2_mesh_cpu = torch.meshgrid(
            torch.arange(n1, dtype=torch.float64),
            torch.arange(n2, dtype=torch.float64),
            indexing='ij'
        )
        taper_grid = tapering_func((u1_mesh_cpu, u2_mesh_cpu), n1, n2).to(device) # Taper on device

        fft_results = []
        for tensor in tensor_list:
            data_grid = torch.zeros((n1, n2), dtype=torch.float64, device=device)
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

            data_grid_tapered = data_grid * taper_grid 

            if torch.isnan(data_grid_tapered).any() or torch.isinf(data_grid_tapered).any():
                print("Warning: NaN/Inf detected in data_grid_tapered before FFT. Replacing with zeros.")
                data_grid_tapered = torch.nan_to_num(data_grid_tapered, nan=0.0, posinf=0.0, neginf=0.0)

            fft_results.append(torch.fft.fft2(data_grid_tapered))

        if not fft_results:
            print("Warning: No FFT results generated.")
            return torch.empty(0, 0, 0, device=device, dtype=torch.complex128), n1, n2, 0, taper_grid

        J_vector_tensor = torch.stack(fft_results, dim=2).to(device)

        H = torch.sum(taper_grid**2)
        if H < 1e-12:
            print("Warning: Normalization factor H is near zero.")
            norm_factor = torch.tensor(0.0, device=device, dtype=torch.float64)
        else:
            norm_factor = (torch.sqrt(1.0 / H) / (2.0 * cmath.pi)).to(device)

        result = J_vector_tensor * norm_factor
        if torch.isnan(result).any(): print("Warning: NaN in J_vector output.")
        return result, n1, n2, p_time, taper_grid # <-- Return p_time

    @staticmethod
    def calculate_sample_periodogram_vectorized(J_vector_tensor):
        """Calculates sample periodogram I_n = J J^H (pxp matrix for each spatial freq)."""
        if torch.isnan(J_vector_tensor).any() or torch.isinf(J_vector_tensor).any():
            print("Warning: NaN/Inf detected in J_vector_tensor input.")
            n1, n2, p = J_vector_tensor.shape
            return torch.full((n1, n2, p, p), float('nan'), dtype=torch.complex128, device=J_vector_tensor.device)

        J_col = J_vector_tensor.unsqueeze(-1)
        J_row_conj = J_vector_tensor.unsqueeze(-2).conj()
        result = J_col @ J_row_conj

        if torch.isnan(result).any(): print("Warning: NaN in periodogram matrix output.")
        return result

    # =========================================================================
    # 2. Covariance Functions (7-Parameter Version)
    # =========================================================================
    @staticmethod
    def cov_x_spatiotemporal_model_kernel(u1, u2, t, params):
        """
        Computes autocovariance of X using the 7-PARAMETER spatio-temporal kernel.
        u1, u2 are PHYSICAL lags (already scaled by deltas).
        t is the PHYSICAL time lag.
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        if torch.isnan(params).any() or torch.isinf(params).any():
            out_shape = torch.broadcast_shapes(u1_dev.shape, u2_dev.shape, t_dev.shape)
            return torch.full(out_shape, float('nan'), device=device, dtype=torch.float64)

        # --- A. Unpack and Recover Parameters ---
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) # range_lon_inv
        phi3   = torch.exp(params[2]) # (range_lon / range_lat)^2
        phi4   = torch.exp(params[3]) # beta^2
        advec_lat = params[4]
        advec_lon = params[5]
        nugget = torch.exp(params[6])

        epsilon = 1e-12
        sigmasq = phi1 / (phi2 + epsilon)  
        range_lon_inv = phi2
        range_lat_inv = torch.sqrt(phi3 + epsilon) * phi2
        beta_scaled_inv = torch.sqrt(phi4 + epsilon) * phi2 # This is beta * range_lon_inv

        # --- B. Calculate Anisotropic Advected Distance ---
        u1_adv = u1_dev - advec_lat * t_dev
        u2_adv = u2_dev - advec_lon * t_dev

        dist_sq = (u1_adv * range_lat_inv).pow(2) + \
                (u2_adv * range_lon_inv).pow(2) + \
                (t_dev * beta_scaled_inv).pow(2)
        
        distance = torch.sqrt(dist_sq + epsilon) 

        # --- C. Calculate Covariance (Matern 0.5 = Exponential) ---
        cov_smooth = sigmasq * torch.exp(-distance)

        # --- D. Add Nugget ---
        is_zero_lag = (torch.abs(u1_dev) < 1e-9) & (torch.abs(u2_dev) < 1e-9) & (torch.abs(t_dev) < 1e-9)
        final_cov = torch.where(is_zero_lag, cov_smooth + nugget, cov_smooth)

        if torch.isnan(final_cov).any(): print("Warning: NaN detected in cov_x_spatiotemporal_model_kernel output.")
        return final_cov

    @staticmethod
    def cov_spatial_difference(u1, u2, t, params, delta1, delta2):
        """
        Calculates covariance Cov(Y(s,t_q), Y(s+u,t_r))
        where Y is the spatially differenced field.
        u1, u2 are PHYSICAL lags. t is the PHYSICAL time lag.
        """
        weights = {(0, 0): -2.0, (1, 0): 1.0, (0, 1): 1.0}
        device = params.device
        out_shape = torch.broadcast_shapes(u1.shape if isinstance(u1, torch.Tensor) else (),
                                        u2.shape if isinstance(u2, torch.Tensor) else (),
                                        t.shape if isinstance(t, torch.Tensor) else ())
        cov = torch.zeros(out_shape, device=device, dtype=torch.float64)
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        for (a_idx, b_idx), w_ab in weights.items():
            offset_a1 = a_idx * delta1
            offset_a2 = b_idx * delta2
            for (c_idx, d_idx), w_cd in weights.items():
                offset_c1 = c_idx * delta1
                offset_c2 = d_idx * delta2
                
                lag_u1 = u1_dev + (offset_a1 - offset_c1)
                lag_u2 = u2_dev + (offset_a2 - offset_c2)
                
                term_cov = debiased_whittle_likelihood.cov_x_spatiotemporal_model_kernel(lag_u1, lag_u2, t_dev, params) 
                
                if torch.isnan(term_cov).any():
                    print(f"Warning: NaN in term_cov within cov_spatial_difference.")
                    return torch.full_like(cov, float('nan'))
                cov += w_ab * w_cd * term_cov

        if torch.isnan(cov).any(): print("Warning: NaN in final cov_spatial_difference output.")
        return cov

    @staticmethod
    def cn_bar_tapered(u1, u2, t, params, n1, n2, taper_autocorr_grid, delta1, delta2):
        """
        Computes c_Y(u) * c_gn(u).
        u1, u2 are GRID index lags (e.g., -n1..0..n1)
        t is the PHYSICAL time lag.
        """
        device = params.device
        u1_dev = u1.to(device) if isinstance(u1, torch.Tensor) else torch.tensor(u1, device=device, dtype=torch.float64)
        u2_dev = u2.to(device) if isinstance(u2, torch.Tensor) else torch.tensor(u2, device=device, dtype=torch.float64)
        t_dev = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float64)

        # --- Convert GRID lags to PHYSICAL lags ---
        lag_u1 = u1_dev * delta1
        lag_u2 = u2_dev * delta2

        cov_X_value = debiased_whittle_likelihood.cov_spatial_difference(lag_u1, lag_u2, t_dev, params, delta1, delta2)

        # --- Get Taper Autocorrelation Value c_gn(u) from grid ---
        u1_idx = u1_dev.long()
        u2_idx = u2_dev.long()

        idx1 = (n1 - 1 + u1_idx) # Centering index
        idx2 = (n2 - 1 + u2_idx) # Centering index
        
        idx1 = torch.clamp(idx1, 0, 2 * n1 - 2)
        idx2 = torch.clamp(idx2, 0, 2 * n2 - 2)

        taper_autocorr_value = taper_autocorr_grid[idx1, idx2]

        if torch.isnan(cov_X_value).any() or torch.isnan(taper_autocorr_value).any():
            out_shape = torch.broadcast_shapes(cov_X_value.shape, taper_autocorr_value.shape)
            return torch.full(out_shape, float('nan'), device=device, dtype=torch.float64)

        result = cov_X_value * taper_autocorr_value
        if torch.isnan(result).any(): print("Warning: NaN in cn_bar_tapered output.")
        return result

    @staticmethod
    def expected_periodogram_fft_tapered(params, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        Calculates the expected periodogram I(omega_s) (a pxp matrix in time)
        using the exact taper autocorrelation c_gn(u) and
        CORRECTLY implementing the aliasing sum (Lemma 2).
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
        tilde_cn_tensor = torch.zeros((n1, n2, p_time, p_time), dtype=torch.complex128, device=device)

        for q in range(p_time):
            for r in range(p_time):
                t_diff = t_lags[q] - t_lags[r]
                
                term1 = debiased_whittle_likelihood.cn_bar_tapered(u1_mesh, u2_mesh, t_diff, 
                                    params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
                term2 = debiased_whittle_likelihood.cn_bar_tapered(u1_mesh - n1, u2_mesh, t_diff, 
                                    params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
                term3 = debiased_whittle_likelihood.cn_bar_tapered(u1_mesh, u2_mesh - n2, t_diff, 
                                    params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
                term4 = debiased_whittle_likelihood.cn_bar_tapered(u1_mesh - n1, u2_mesh - n2, t_diff,
                                    params_tensor, n1, n2, taper_autocorr_grid, delta1, delta2)
                
                tilde_cn_grid_qr = (term1 + term2 + term3 + term4)
                
                if torch.isnan(tilde_cn_grid_qr).any():
                    tilde_cn_tensor[:, :, q, r] = float('nan')
                else:
                    tilde_cn_tensor[:, :, q, r] = tilde_cn_grid_qr.to(torch.complex128)

        if torch.isnan(tilde_cn_tensor).any():
            print("Warning: NaN detected in tilde_cn_tensor before FFT.")
            nan_shape = (n1, n2, p_time, p_time)
            return torch.full(nan_shape, float('nan'), dtype=torch.complex128, device=device)

        fft_result = torch.fft.fft2(tilde_cn_tensor, dim=(0, 1))
        fft_result_real = fft_result.real 
        normalization_factor = 1.0 / (4.0 * cmath.pi**2)
        result = fft_result_real * normalization_factor

        if torch.isnan(result).any(): print("Warning: NaN in expected_periodogram_fft_tapered output.")
        return result

    # =========================================================================
    # 4. Likelihood Calculation (Tapered)
    # =========================================================================
    
    @staticmethod
    def whittle_likelihood_loss_tapered(params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        ✅ Whittle Likelihood Loss (AVERAGED) using data tapering.
        """
        device = I_sample.device
        params_tensor = params.to(device)

        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print("Warning: NaN/Inf detected in input parameters to likelihood.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid, 
            delta1, delta2
        )

        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            print("Warning: NaN/Inf returned from expected_periodogram calculation.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        eye_matrix = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals = torch.abs(I_expected.diagonal(dim1=-2, dim2=-1))
        mean_diag_abs = diag_vals.mean().item() if diag_vals.numel() > 0 and not torch.isnan(diag_vals).all() else 1.0
        diag_load = max(mean_diag_abs * 1e-8, 1e-9)
        I_expected_stable = I_expected + eye_matrix * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_expected_stable)
        if torch.any(sign.real <= 1e-9):
            print("Warning: Non-positive determinant encountered. Applying penalty.")
            log_det_term = torch.where(sign.real > 1e-9, logabsdet, torch.tensor(1e10, device=device, dtype=torch.float64))
        else:
            log_det_term = logabsdet

        if torch.isnan(I_sample).any() or torch.isinf(I_sample).any():
            print("Warning: NaN/Inf detected in I_sample input to likelihood.")
            return torch.tensor(float('nan'), device=device)

        try:
            solved_term = torch.linalg.solve(I_expected_stable, I_sample)
            trace_term = torch.einsum('...ii->...', solved_term).real
        except torch.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError during solve: {e}. Applying high loss penalty.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(trace_term).any() or torch.isinf(trace_term).any():
            print("Warning: NaN/Inf detected in trace_term. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        likelihood_terms = log_det_term + trace_term

        if torch.isnan(likelihood_terms).any():
            print("Warning: NaN detected in likelihood_terms before summation. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        total_sum = torch.sum(likelihood_terms)
        dc_term = likelihood_terms[0, 0] if n1 > 0 and n2 > 0 else torch.tensor(0.0, device=device, dtype=torch.float64)
        if torch.isnan(dc_term).any() or torch.isinf(dc_term).any():
            print("Warning: NaN/Inf detected in DC term. Setting to 0.")
            dc_term = torch.tensor(0.0, device=device, dtype=torch.float64)

        # This is the sum of non-zero frequency likelihood terms
        sum_loss = total_sum - dc_term if (n1 > 1 or n2 > 1) else total_sum

        # --- REVISION: Convert sum to average ---
        num_terms = (n1 * n2) - 1
        if num_terms > 0:
            avg_loss = sum_loss / num_terms
        else:
            avg_loss = sum_loss # Handle edge case of 1x1 grid (where num_terms=0)
        # --- End Revision ---

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print("Warning: NaN/Inf detected in final loss. Returning Inf penalty.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        return avg_loss

    @staticmethod
    def whittle_likelihood_loss_tapered_sum(params, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2):
        """
        ✅ Whittle Likelihood Loss (AVERAGED) using data tapering.
        """
        device = I_sample.device
        params_tensor = params.to(device)

        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print("Warning: NaN/Inf detected in input parameters to likelihood.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        I_expected = debiased_whittle_likelihood.expected_periodogram_fft_tapered(
            params_tensor, n1, n2, p_time, taper_autocorr_grid, 
            delta1, delta2
        )

        if torch.isnan(I_expected).any() or torch.isinf(I_expected).any():
            print("Warning: NaN/Inf returned from expected_periodogram calculation.")
            return torch.tensor(float('nan'), device=device)

        eye_matrix = torch.eye(p_time, dtype=torch.complex128, device=device)
        diag_vals = torch.abs(I_expected.diagonal(dim1=-2, dim2=-1))
        mean_diag_abs = diag_vals.mean().item() if diag_vals.numel() > 0 and not torch.isnan(diag_vals).all() else 1.0
        diag_load = max(mean_diag_abs * 1e-8, 1e-9)
        I_expected_stable = I_expected + eye_matrix * diag_load

        sign, logabsdet = torch.linalg.slogdet(I_expected_stable)
        if torch.any(sign.real <= 1e-9):
            print("Warning: Non-positive determinant encountered. Applying penalty.")
            log_det_term = torch.where(sign.real > 1e-9, logabsdet, torch.tensor(1e10, device=device, dtype=torch.float64))
        else:
            log_det_term = logabsdet

        if torch.isnan(I_sample).any() or torch.isinf(I_sample).any():
            print("Warning: NaN/Inf detected in I_sample input to likelihood.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        try:
            solved_term = torch.linalg.solve(I_expected_stable, I_sample)
            trace_term = torch.einsum('...ii->...', solved_term).real
        except torch.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError during solve: {e}. Applying high loss penalty.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        if torch.isnan(trace_term).any() or torch.isinf(trace_term).any():
            print("Warning: NaN/Inf detected in trace_term. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)
        likelihood_terms = log_det_term + trace_term

        if torch.isnan(likelihood_terms).any():
            print("Warning: NaN detected in likelihood_terms before summation. Returning NaN loss.")
            return torch.tensor(float('nan'), device=device, dtype=torch.float64)

        total_sum = torch.sum(likelihood_terms)
        dc_term = likelihood_terms[0, 0] if n1 > 0 and n2 > 0 else torch.tensor(0.0, device=device)
        if torch.isnan(dc_term).any() or torch.isinf(dc_term).any():
            print("Warning: NaN/Inf detected in DC term. Setting to 0.")
            dc_term = torch.tensor(0.0, device=device, dtype=torch.float64)

        # This is the sum of non-zero frequency likelihood terms
        sum_loss = total_sum - dc_term if (n1 > 1 or n2 > 1) else total_sum

        if torch.isnan(sum_loss) or torch.isinf(sum_loss):
            print("Warning: NaN/Inf detected in final loss. Returning Inf penalty.")
            return torch.tensor(float('inf'), device=device, dtype=torch.float64)

        return sum_loss, n1, n2 #total_sum to sum_loss
    

    # =========================================================================
    # 5. Training Loop & Helpers (💥 NEWLY ADDED 💥)
    # =========================================================================
    
    @staticmethod
    def get_printable_params_7param(p_list):
        """Helper to convert 7-param log list to natural scale for printing."""
        valid_tensors = [p for p in p_list if isinstance(p, torch.Tensor)]
        if not valid_tensors: return "Invalid params_list"
        p_cat = torch.cat([p.detach().clone().cpu() for p in valid_tensors])
        
        if len(p_cat) != 7:
            return f"Expected 7 params, got {len(p_cat)}."

        log_params = p_cat
        if torch.isnan(log_params).any() or torch.isinf(log_params).any():
            return "[NaN/Inf in log_params]"
            
        try:
            phi1 = torch.exp(log_params[0])
            phi2 = torch.exp(log_params[1]) # range_lon_inv
            phi3 = torch.exp(log_params[2]) # (range_lon / range_lat)^2
            phi4 = torch.exp(log_params[3]) # beta^2
            advec_lat = log_params[4]       # Not in log scale
            advec_lon = log_params[5]       # Not in log scale
            nugget = torch.exp(log_params[6])

            epsilon = 1e-12
            sigmasq = phi1 / (phi2 + epsilon)
            range_lon = 1.0 / (phi2 + epsilon)
            range_lat = 1.0 / (torch.sqrt(phi3 + epsilon) * phi2 + epsilon)
            range_time = range_lon / torch.sqrt(phi4 + epsilon)
            
            return (f"sigmasq: {sigmasq.item():.4f}, range_lat: {range_lat.item():.4f}, "
                    f"range_lon: {range_lon.item():.4f}, range_time: {range_time.item():.4f}, "
                    f"advec_lat: {advec_lat.item():.4f}, advec_lon: {advec_lon.item():.4f}, "
                    f"nugget: {nugget.item():.4f}")
        except Exception as e:
            return f"[Error in param conversion: {e}]"

    @staticmethod
    def get_phi_params_7param(log_params_list):
        """Helper to print 7-param reparameterized (phi-scale) params."""
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            phi1 = torch.exp(p_cat[0])
            phi2 = torch.exp(p_cat[1])
            phi3 = torch.exp(p_cat[2])
            phi4 = torch.exp(p_cat[3])
            advec_lat = p_cat[4]
            advec_lon = p_cat[5]
            nugget = torch.exp(p_cat[6]) 
            
            return (f"phi1: {phi1.item():.4f}, phi2: {phi2.item():.4f}, phi3: {phi3.item():.4f}, "
                    f"phi4: {phi4.item():.4f}, advec_lat: {advec_lat.item():.4f}, advec_lon: {advec_lon.item():.4f}, "
                    f"nugget: {nugget.item():.4f}")
        except Exception:
            return "[Error in reparam conversion]"

    @staticmethod
    def get_raw_log_params_7param(log_params_list):
        """Helper to print the raw 7 params being optimized."""
        try:
            p_cat = torch.cat([p.detach().clone().cpu() for p in log_params_list])
            return (f"log_phi1: {p_cat[0].item():.4f}, log_phi2: {p_cat[1].item():.4f}, "
                    f"log_phi3: {p_cat[2].item():.4f}, log_phi4: {p_cat[3].item():.4f}, "
                    f"advec_lat: {p_cat[4].item():.4f}, advec_lon: {p_cat[5].item():.4f}, "
                    f"log_nugget: {p_cat[6].item():.4f}")
        except Exception:
            return "[Error in raw param conversion]"

    @staticmethod
    def run_full_tapered(params_list, optimizer, scheduler, I_sample, n1, n2, p_time, taper_autocorr_grid, epochs=600, device='cpu'):
        # *** REVISED: Use p_time ***
        """Corrected training loop with gradient-based convergence."""
        best_loss = float('inf')
        params_list = [p.to(device) for p in params_list]
        best_params_state = [p.detach().clone() for p in params_list]
        epochs_completed = 0
        DELTA_LAT, DELTA_LON = 0.044, 0.063 
        
        grad_tol = 1e-5

        I_sample_dev = I_sample.to(device)
        taper_autocorr_grid_dev = taper_autocorr_grid.to(device) 

        for epoch in range(epochs):
            epochs_completed = epoch + 1
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list) 

            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time, taper_autocorr_grid_dev, DELTA_LAT, DELTA_LON
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss became NaN or Inf at epoch {epoch+1}. Stopping.")
                if epoch == 0: best_params_state = None
                epochs_completed = epoch
                break

            loss.backward()

            nan_grad = False
            max_abs_grad = 0.0
            for param in params_list:
                if param.grad is not None:
                    if (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        nan_grad = True
                        break
                    max_abs_grad = max(max_abs_grad, param.grad.abs().item())
                
            if nan_grad:
                 print(f"Warning: NaN/Inf gradient at epoch {epoch+1}. Skipping step.")
                 optimizer.zero_grad()
                 continue

            all_params_on_device = params_list
            if all_params_on_device:
                torch.nn.utils.clip_grad_norm_(all_params_on_device, max_norm=1.0)
            
            if epoch > 10 and max_abs_grad < grad_tol: # 10-epoch warmup
                print(f"\n--- Converged on gradient norm (max|grad| < {grad_tol}) at epoch {epoch+1} ---")
                epochs_completed = epoch + 1
                break 

            optimizer.step()
            
            current_loss_item = loss.item()
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_loss_item)
            else:
                scheduler.step()

            if current_loss_item < best_loss:
                params_valid = not any(torch.isnan(p.data).any() or torch.isinf(p.data).any() for p in params_list)
                if params_valid:
                    best_loss = current_loss_item
                    best_params_state = [p.detach().clone() for p in params_list]

            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0

            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f'--- Epoch {epoch+1}/{epochs} (LR: {current_lr:.6f}) ---')
                print(f' Loss: {current_loss_item:.4f} | Max Grad: {max_abs_grad:.6e}')
                print(f'  Params (Raw Log): {debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}')


        print("\n--- Training Complete ---")
        if best_params_state is None:
            print("Training failed to find a valid model state.")
            return None, None, None, None, epochs_completed # Return Nones

        final_natural_params_str = debiased_whittle_likelihood.get_printable_params_7param(best_params_state)
        final_phi_params_str = debiased_whittle_likelihood.get_phi_params_7param(best_params_state)
        final_raw_params_str = debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state)
        final_loss_rounded = round(best_loss, 3) if best_loss != float('inf') else float('inf')

        print(f'\nFINAL BEST STATE ACHIEVED (during training):')
        print(f'Best Loss: {final_loss_rounded}')
        
        return final_natural_params_str, final_phi_params_str, final_raw_params_str, final_loss_rounded, epochs_completed
    
    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time, taper_autocorr_grid, max_steps=50, device='cpu',grad_tol=1e-5):
        """Training loop using L-BFGS optimizer with improved convergence checks."""
        
        params_list = [p.to(device) for p in params_list]
        best_params_state = [p.detach().clone() for p in params_list]
        steps_completed = 0
        DELTA_LAT, DELTA_LON = 0.044, 0.063 
        
        loss_tol = 1e-12 
        
        best_loss = float('inf')
        prev_loss_item = float('inf') 

        I_sample_dev = I_sample.to(device)
        taper_autocorr_grid_dev = taper_autocorr_grid.to(device) 

        def closure():
            optimizer.zero_grad()
            params_tensor = torch.cat(params_list) 
            
            loss = debiased_whittle_likelihood.whittle_likelihood_loss_tapered(
                params_tensor, I_sample_dev, n1, n2, p_time, taper_autocorr_grid_dev, DELTA_LAT, DELTA_LON
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN/Inf inside closure. Returning.")
                return loss 
            
            loss.backward()
            
            nan_grad = False
            for param in params_list:
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    nan_grad = True; break
            if nan_grad:
                 print(f"Warning: NaN/Inf gradient detected. Zeroing grad.")
                 optimizer.zero_grad() 
            return loss
        # --- End of closure ---

        for i in range(max_steps):
            steps_completed = i + 1
            
            loss = optimizer.step(closure)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i+1}/{max_steps}: Loss is NaN/Inf. Stopping optimization.")
                break
                
            current_loss_item = loss.item()
            
            if current_loss_item < best_loss:
                params_valid = not any(torch.isnan(p.data).any() or torch.isinf(p.data).any() for p in params_list)
                if params_valid:
                    best_loss = current_loss_item
                    best_params_state = [p.detach().clone() for p in params_list]
            
            max_abs_grad = 0.0
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        max_abs_grad = max(max_abs_grad, p.grad.abs().item())

            print(f'--- Step {i+1}/{max_steps} ---')
            print(f' Loss: {current_loss_item:.6f} | Max Grad: {max_abs_grad:.6e}')
            print(f'  Params (Raw Log): {debiased_whittle_likelihood.get_raw_log_params_7param(params_list)}')
            
            loss_change = abs(current_loss_item - prev_loss_item)
            
            if i > 2: # Warmup period
                if max_abs_grad < grad_tol:
                    print(f"\n--- Converged on gradient norm (max|grad| < {grad_tol}) at step {i+1} ---")
                    break
                if loss_change < loss_tol:
                    print(f"\n--- Converged on loss change (change < {loss_tol}) at step {i+1} ---")
                    break
            
            prev_loss_item = current_loss_item

        print("\n--- Training Complete ---")
        if best_params_state is None:
            print("Training failed to find a valid model state.")
            return None, None, None, None, steps_completed

        final_natural_params_str = debiased_whittle_likelihood.get_printable_params_7param(best_params_state)
        final_phi_params_str = debiased_whittle_likelihood.get_phi_params_7param(best_params_state)
        final_raw_params_str = debiased_whittle_likelihood.get_raw_log_params_7param(best_params_state)
        final_loss_rounded = round(best_loss, 3) if best_loss != float('inf') else float('inf')

        print(f'\nFINAL BEST STATE ACHIEVED (during training):')
        print(f'Best Loss: {final_loss_rounded}')
        
        return final_natural_params_str, final_phi_params_str, final_raw_params_str, final_loss_rounded, steps_completed