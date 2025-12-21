# Standard libraries
# import logging
# import math
# import time
# from collections import defaultdict
# Special functions and optimizations
# from scipy.spatial.distance import cdist  # For space and time distance
# from scipy.optimize import minimize  # For optimization

import sys
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)
import GEMS_TCO
from scipy.special import gamma, kv  # Bessel function and gamma function

from collections import defaultdict
import pandas as pd
import numpy as np

import torch
from torch.func import grad, hessian, jacfwd, jacrev
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import torch.optim as optim
import torch.nn as nn
from scipy.interpolate import splrep, splev

# Fit your "spline" by just storing the x and y
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from typing import Dict, Any, Callable, List, Tuple

import time
import copy    
import logging     # for logging
# Add your custom path

import warnings
from functools import partial

import torch.optim as optim

sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

from torch.special import gammainc, gammaln



# --- BASE CLASS ---
class SpatioTemporalModel:
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
        self.device = aggregated_data.device
        self.smooth = smooth
        self.smooth_tensor = torch.tensor(self.smooth, dtype=torch.float64, device=self.device)
        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64, device=self.device)
        self.matern_const = ( (2**(1-self.smooth)) / gamma_val )

        self.input_map = input_map
        self.aggregated_data = aggregated_data[:,:4]
        self.key_list = list(input_map.keys())
        self.size_per_hour = len(input_map[self.key_list[0]])
        self.mm_cond_number = mm_cond_number

        # Process NNS Map
        nns_map = list(nns_map) 
        for i in range(len(nns_map)):  
            tmp = np.delete(nns_map[i][:self.mm_cond_number], np.where(nns_map[i][:self.mm_cond_number] == -1))
            if tmp.size > 0:
                nns_map[i] = tmp
            else:
                nns_map[i] = []
        self.nns_map = nns_map

    # --- SINGLE MATRIX KERNEL (For "Heads" / Exact GP) ---
    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates distance for the Head points (Exact GP).
        Uses broadcasting logic (Safe & Robust).
        """
        phi3, phi4, advec_lat, advec_lon = dist_params
        
        # x columns: [lat, lon, val, time]
        u_lat_adv = x[:, 0] - advec_lat * x[:, 3]
        u_lon_adv = x[:, 1] - advec_lon * x[:, 3]
        u_t = x[:, 3]
        
        v_lat_adv = y[:, 0] - advec_lat * y[:, 3]
        v_lon_adv = y[:, 1] - advec_lon * y[:, 3]
        v_t = y[:, 3]

        # Broadcasting: (N, 1) - (1, M) -> (N, M)
        delta_lat = u_lat_adv.unsqueeze(1) - v_lat_adv.unsqueeze(0)
        delta_lon = u_lon_adv.unsqueeze(1) - v_lon_adv.unsqueeze(0)
        delta_t   = u_t.unsqueeze(1)       - v_t.unsqueeze(0)

        dist_sq = (delta_lat.pow(2) * phi3) + delta_lon.pow(2) + (delta_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single Matrix Matern Covariance (Log-Reparameterized)"""
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) 
        phi3   = torch.exp(params[2])
        phi4   = torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        
        cov = sigmasq * torch.exp(-distance * phi2)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8) 
        
        return cov
        
    def full_likelihood_avg(self, params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable, temporal_period=None) -> torch.Tensor:
            """Exact GP NLL calculation with optional Harmonic Trend."""
            input_data = input_data.to(torch.float64)
            y = y.to(torch.float64)
            N = input_data.shape[0]
            if N == 0: return torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    
            cov_matrix = covariance_function(params=params, y=input_data, x=input_data)
            
            try:
                jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                L = torch.linalg.cholesky(cov_matrix + jitter)
            except torch.linalg.LinAlgError:
                return torch.tensor(torch.inf, device=params.device, dtype=params.dtype)
            
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            
            # --- [MODIFIED] Spatial + Harmonic Trend ---
            ones = torch.ones(N, 1, device=self.device, dtype=torch.float64)
            coords = input_data[:, :2] # Lat, Lon
            
            if temporal_period:
                # input_data columns: [lat, lon, val, time] -> time is index 3
                t_vals = input_data[:, 3]
                omega = 2 * np.pi / temporal_period
                sin_t = torch.sin(omega * t_vals).unsqueeze(1)
                cos_t = torch.cos(omega * t_vals).unsqueeze(1)
                # Cols: [Intercept, Lat, Lon, Sin, Cos]
                locs = torch.cat((ones, coords, sin_t, cos_t), dim=1)
            else:
                # Cols: [Intercept, Lat, Lon]
                locs = torch.cat((ones, coords), dim=1)
            # -------------------------------------------
            
            if y.dim() == 1: y_col = y.unsqueeze(-1)
            else: y_col = y

            # Optimize: Solve for [X, y] together
            combined_rhs = torch.cat((locs, y_col), dim=1)
            C_inv_combined = torch.cholesky_solve(combined_rhs, L, upper=False)
            
            # Split back based on number of trend columns
            n_trend = locs.shape[1]
            C_inv_X, C_inv_y = C_inv_combined[:, :n_trend], C_inv_combined[:, n_trend:]

            tmp1 = torch.matmul(locs.T, C_inv_X)
            tmp2 = torch.matmul(locs.T, C_inv_y)
            
            try:
                jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2)
            except torch.linalg.LinAlgError:
                return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

            mu = torch.matmul(locs, beta)
            y_mu = y_col - mu
            
            C_inv_y_mu = torch.cholesky_solve(y_mu, L, upper=False)
            quad_form = torch.matmul(y_mu.T, C_inv_y_mu) 

            neg_log_lik_sum = 0.5 * (log_det + quad_form.squeeze())
            return neg_log_lik_sum / N


# --- BATCHED GPU CLASS ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device 
        self.nheads = nheads
        self.max_neighbors = mm_cond_number 
        self.is_precomputed = False
        
        # Batched Tensors (Tails)
        self.X_batch = None
        self.Y_batch = None
        self.Locs_batch = None
        # Heads Tensor
        self.Heads_data = None 

    def precompute_conditioning_sets(self, temporal_period=None):
                
        # [NEW] Store period for use in likelihood function later
        self.temporal_period = temporal_period  
        
        print(f"Pre-computing Batched Tensors (Padding Strategy, Period: {temporal_period})...", end=" ")
        
        # ... (Heads Data, Tails Data 처리 로직은 동일) ...
        
        key_list = list(self.input_map.keys())
        cut_line = self.nheads
        
        # 1. Heads Data
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, np.ndarray):
                head_chunk = torch.from_numpy(day_data[:cut_line]).to(self.device)
            else:
                head_chunk = day_data[:cut_line].clone().detach().to(self.device)
            heads_list.append(head_chunk)
        self.Heads_data = torch.cat(heads_list, dim=0).to(torch.float64)

        # 2. Tails Data (Task List)
        tasks = []
        for time_idx, key in enumerate(key_list):
            day_data = self.input_map[key]
            indices = range(cut_line, len(day_data))
            for idx in indices:
                tasks.append((time_idx, idx))

        total_vecchia_points = len(tasks)
        max_storage_dim = (self.max_neighbors + 1) * 3
        
        # 3. Allocating Tensors
        self.X_batch = torch.full((total_vecchia_points, max_storage_dim, 3), 1e6, device=self.device, dtype=torch.float64)
        self.Y_batch = torch.zeros((total_vecchia_points, max_storage_dim, 1), device=self.device, dtype=torch.float64)
        
        # --- ✅ FIX 1: 컬럼 개수 동적 할당 ---
        n_trend_cols = 5 if temporal_period else 3
        self.Locs_batch = torch.zeros((total_vecchia_points, max_storage_dim, n_trend_cols), device=self.device, dtype=torch.float64)

        # 4. Fill Tensors
        for i, (time_idx, index) in enumerate(tasks):
            # ... (데이터 블록 합치는 로직 동일) ...
            current_np = self.input_map[key_list[time_idx]]
            current_row = current_np[index].reshape(1, -1) 
            mm_neighbors = self.nns_map[index]
            past = list(mm_neighbors)
            
            data_list = []
            if past: data_list.append(current_np[past])
            if time_idx > 0: data_list.append(self.input_map[key_list[time_idx - 1]][past + [index], :])
            if time_idx > 1: data_list.append(self.input_map[key_list[time_idx - 2]][past + [index], :])

            if data_list:
                if isinstance(data_list[0], np.ndarray):
                    neighbors_block = torch.from_numpy(np.vstack(data_list)).to(self.device)
                else:
                    neighbors_block = torch.vstack(data_list).to(self.device)
            else:
                neighbors_block = torch.empty((0, current_row.shape[1]), device=self.device)

            if isinstance(current_row, np.ndarray):
                target_block = torch.from_numpy(current_row).to(self.device)
            else:
                target_block = current_row.clone().detach().to(self.device)

            combined_data = torch.cat([neighbors_block, target_block], dim=0)
            actual_len = combined_data.shape[0]
            start_slot = max_storage_dim - actual_len
            
            self.X_batch[i, start_slot:, 0] = combined_data[:, 0]
            self.X_batch[i, start_slot:, 1] = combined_data[:, 1]
            self.X_batch[i, start_slot:, 2] = combined_data[:, 3]
            self.Y_batch[i, start_slot:, 0] = combined_data[:, 2]
            
            # --- ✅ FIX 2: Loop 안에서 값 채워넣기 ---
            # Col 0, 1, 2
            self.Locs_batch[i, start_slot:, 0] = 1.0 
            self.Locs_batch[i, start_slot:, 1] = combined_data[:, 0] 
            self.Locs_batch[i, start_slot:, 2] = combined_data[:, 1] 
            
            # Col 3, 4 (Harmonic)
            if temporal_period:
                t_vals = combined_data[:, 3]
                omega = 2 * np.pi / temporal_period
                self.Locs_batch[i, start_slot:, 3] = torch.sin(omega * t_vals)
                self.Locs_batch[i, start_slot:, 4] = torch.cos(omega * t_vals)

        self.is_precomputed = True
        print(f"Done. Heads: {self.Heads_data.shape[0]}, Batched Tails: {self.X_batch.shape[0]}")

    def batched_manual_dist(self, dist_params, x_batch):
        """Batched Manual Broadcasting Distance."""
        phi3, phi4, advec_lat, advec_lon = dist_params

        # x_batch is (Batch, N, 3) -> [Lat, Lon, Time]
        x_lat  = x_batch[:, :, 0]
        x_lon  = x_batch[:, :, 1]
        x_time = x_batch[:, :, 2]

        u_lat_adv = x_lat - advec_lat * x_time
        u_lon_adv = x_lon - advec_lon * x_time
        u_time    = x_time

        # Broadcasting: (B, N, 1) - (B, 1, N) -> (B, N, N)
        delta_lat = u_lat_adv.unsqueeze(2) - u_lat_adv.unsqueeze(1)
        delta_lon = u_lon_adv.unsqueeze(2) - u_lon_adv.unsqueeze(1)
        delta_t   = u_time.unsqueeze(2)    - u_time.unsqueeze(1)

        dist_sq = (delta_lat.pow(2) * phi3) + (delta_lon.pow(2)) + (delta_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_batched(self, params, x_batch):
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) 
        phi3   = torch.exp(params[2])
        phi4   = torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2 
        
        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.batched_manual_dist(dist_params, x_batch)
        
        cov = sigmasq * torch.exp(-distance * phi2)
        
        # Add nugget to diagonal
        batch_size, N, _ = x_batch.shape
        identity = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(batch_size, N, N)
        cov = cov + identity * (nugget + 1e-6)
        
        return cov

    def vecchia_batched_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")
            
        # 1. Heads (Exact GP)
        def adapter_cov_func(params, x, y):
            return self.matern_cov_aniso_STABLE_log_reparam(params, x, y)

        if self.Heads_data.shape[0] > 0:
            # [MODIFIED] Pass self.temporal_period to full_likelihood_avg
            head_nll_avg = self.full_likelihood_avg(
                params, 
                self.Heads_data, 
                self.Heads_data[:, 2], 
                adapter_cov_func,
                temporal_period=self.temporal_period  # <--- HERE
            )
            head_nll_sum = head_nll_avg * self.Heads_data.shape[0]
        else:
            head_nll_sum = 0.0



        # 2. Tails (Batched GPU)
        cov_batch = self.matern_cov_batched(params, self.X_batch)
        
        try:
            L_batch = torch.linalg.cholesky(cov_batch)
        except torch.linalg.LinAlgError:
            print("Warning: GPU Cholesky failed.")
            return torch.tensor(float('inf'), device=self.device)
            
        # GLS Trend Removal (Batched)
        C_inv_locs = torch.cholesky_solve(self.Locs_batch, L_batch, upper=False)
        Xt_Cinv_X = torch.bmm(self.Locs_batch.transpose(1, 2), C_inv_locs)
        
        C_inv_y = torch.cholesky_solve(self.Y_batch, L_batch, upper=False)
        Xt_Cinv_y = torch.bmm(self.Locs_batch.transpose(1, 2), C_inv_y)
        


        # --- ✅ FIX 3: Jitter 크기 동적 할당 ---
        n_trend = self.Locs_batch.shape[-1]  # 3 or 5
        jitter = torch.eye(n_trend, device=self.device, dtype=torch.float64).unsqueeze(0) * 1e-8
        
        beta = torch.linalg.solve(Xt_Cinv_X + jitter, Xt_Cinv_y)
        
        mu = torch.bmm(self.Locs_batch, beta)
        residuals = self.Y_batch - mu
        
        # NLL Components
        z_full = torch.linalg.solve_triangular(L_batch, residuals, upper=False)
        z_target = z_full[:, -1, :] 
        
        sigma_cond = L_batch[:, -1, -1]
        log_det = 2 * torch.log(sigma_cond)
        quad_form = z_target.squeeze() ** 2
        
        vecchia_nll_sum = 0.5 * (log_det + quad_form).sum()
        
        total_nll = head_nll_sum + vecchia_nll_sum
        total_count = self.Heads_data.shape[0] + self.X_batch.shape[0]
        
        return total_nll / total_count


# --- FITTING CLASS ---
class fit_vecchia_lbfgs(VecchiaBatched): 
    
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)

    def set_optimizer(self, param_groups, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100):
        optimizer = torch.optim.LBFGS(
            param_groups, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size
        )
        return optimizer

    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """Converts raw optimized parameters back to interpretable model parameters."""
        try:
            log_phi1 = raw_params_list[0]
            log_phi2 = raw_params_list[1]
            log_phi3 = raw_params_list[2]
            log_phi4 = raw_params_list[3]
            advec_lat = raw_params_list[4]
            advec_lon = raw_params_list[5]
            log_nugget = raw_params_list[6]

            phi1 = np.exp(log_phi1)
            phi2 = np.exp(log_phi2)
            phi3 = np.exp(log_phi3)
            phi4 = np.exp(log_phi4)
            nugget = np.exp(log_nugget)
            
            range_lon = 1.0 / phi2
            sigmasq = phi1 / phi2 
            range_lat = range_lon / np.sqrt(phi3)
            range_time = 1/(np.sqrt(phi4) * phi2) 

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "range_time": range_time,
                "advec_lat": advec_lat,
                "advec_lon": advec_lon,
                "nugget": nugget
            }
        except Exception as e:
            print(f"\nWarning: Could not convert raw params. Error: {e}")
            return {}

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-7):
        
        # 1. Prepare GPU Data
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Batched L-BFGS Optimization (GPU) ---")

        def closure():
            optimizer.zero_grad()
            params = torch.stack(params_list)
            # Batched Likelihood Calculation
            loss = self.vecchia_batched_likelihood(params)
            loss.backward()
            return loss

        for i in range(max_steps):
            loss = optimizer.step(closure)
            
            # Monitoring
            max_abs_grad = 0.0
            grad_values = []
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        grad_values.append(abs(p.grad.item())) 
                if grad_values: max_abs_grad = max(grad_values)

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                for j, param_tensor in enumerate(params_list):
                    grad_value = param_tensor.grad.item() if param_tensor.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)

            if max_abs_grad < grad_tol:
                print(f"\nConverged at step {i+1}")
                break

        final_params_tensor = torch.stack(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        
        # Interpretable conversion
        interpretable = self._convert_raw_params_to_interpretable(final_raw_params_list)
        print("Final Interpretable Params:", interpretable)
        
        return final_raw_params_list + [final_loss], i
