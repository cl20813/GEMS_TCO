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



import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from scipy.special import gamma

# ... [SpatioTemporalModel Class remains unchanged] ...
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
    
    def compute_theoretical_semivariogram_vectorized(self, params: torch.Tensor, lag_distances: torch.Tensor, time_lag: float, direction: str = 'lat') -> Tuple[torch.Tensor, torch.Tensor]:
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) 
        phi3   = torch.exp(params[2])
        phi4   = torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2

        if direction == 'lat':
            lat_dist = lag_distances
            lon_dist = torch.zeros_like(lag_distances)
        elif direction == 'lon':
            lat_dist = torch.zeros_like(lag_distances)
            lon_dist = lag_distances
        else:
            raise ValueError("direction must be 'lat' or 'lon'")

        t_dist = torch.full_like(lag_distances, time_lag)

        u_lat = lat_dist - advec_lat * t_dist
        u_lon = lon_dist - advec_lon * t_dist
        
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (t_dist.pow(2) * phi4)
        distance = torch.sqrt(dist_sq + 1e-8)

        cov = sigmasq * torch.exp(-distance * phi2)
        total_sill = sigmasq + nugget
        
        semivariogram = total_sill - cov
        
        return distance, semivariogram
    

    # --- SINGLE MATRIX KERNEL ---
    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates distance for the Head points (Exact GP).
        """
        phi3, phi4, advec_lat, advec_lon = dist_params
        
        u_lat_adv = x[:, 0] - advec_lat * x[:, 3]
        u_lon_adv = x[:, 1] - advec_lon * x[:, 3]
        u_t = x[:, 3]
        
        v_lat_adv = y[:, 0] - advec_lat * y[:, 3]
        v_lon_adv = y[:, 1] - advec_lon * y[:, 3]
        v_t = y[:, 3]

        # 1. Stack vectors [lat, lon, time] -> Shape (N, 3)
        u_vec = torch.stack([u_lat_adv, u_lon_adv, u_t], dim=1)
        v_vec = torch.stack([v_lat_adv, v_lon_adv, v_t], dim=1)

        # 2. Define Anisotropy Weights (Diagonal Matrix)
        # --- FIX APPLIED HERE: Handle 0D vs 1D mismatch ---
        one_tensor = torch.tensor(1.0, device=x.device, dtype=phi3.dtype)
        if phi3.ndim > 0:
            one_tensor = one_tensor.view_as(phi3)
            
        weights = torch.stack([phi3, one_tensor, phi4])
        weights = weights.view(-1) # Force to flat vector
        # --------------------------------------------------

        # 3. Weighted Norms
        u_sq = (u_vec.pow(2) * weights).sum(dim=1, keepdim=True)
        v_sq = (v_vec.pow(2) * weights).sum(dim=1, keepdim=True)

        # 4. Weighted Dot Product
        uv = (u_vec * weights) @ v_vec.T

        dist_sq = u_sq - 2 * uv + v_sq.T
        return torch.sqrt(dist_sq.clamp(min=1e-8))


    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    
    def full_likelihood_avg(self, params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
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
        
        locs = torch.cat((torch.ones(N, 1, device=self.device, dtype=torch.float64), input_data[:,:2]), dim=1)
        if y.dim() == 1: y_col = y.unsqueeze(-1)
        else: y_col = y
        
        combined_rhs = torch.cat((locs, y_col), dim=1)
        Z_combined = torch.linalg.solve_triangular(L, combined_rhs, upper=False)
        
        Z_X = Z_combined[:, :3]
        Z_y = Z_combined[:, 3:]

        tmp1 = torch.matmul(Z_X.T, Z_X)
        tmp2 = torch.matmul(Z_X.T, Z_y)
        
        try:
            jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
            beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2)
        except torch.linalg.LinAlgError:
            return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

        Z_residual = Z_y - torch.matmul(Z_X, beta)
        quad_form = torch.matmul(Z_residual.T, Z_residual)

        neg_log_lik_sum = 0.5 * (log_det + quad_form.squeeze())
        return neg_log_lik_sum / N


# --- BATCHED GPU CLASS (Optimized & Feature Engineered) ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device 
        self.nheads = nheads
        self.max_neighbors = mm_cond_number 
        self.is_precomputed = False
        self.n_features = 6  # Intercept, Lat, Lon, Time, Lat*Time, Lon*Time
        
        # Tensors
        self.X_batch = None
        self.Y_batch = None
        self.Locs_batch = None
        self.Heads_data = None 

    def precompute_conditioning_sets(self):
        print("Pre-computing Batched Tensors (Optimized CPU -> GPU)...", end=" ")
        
        # --- 1. HEADS (Exact GP Block) ---
        key_list = list(self.input_map.keys())
        cut_line = self.nheads
        
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, torch.Tensor): day_data = day_data.cpu().numpy()
            heads_list.append(day_data[:cut_line])
        
        # Stack numpy, send to GPU once
        self.Heads_data = torch.from_numpy(np.concatenate(heads_list, axis=0)).to(self.device, dtype=torch.float64)

        # --- 2. TAILS (Vecchia Blocks) ---
        tasks = []
        for time_idx, key in enumerate(key_list):
            day_data = self.input_map[key]
            indices = range(cut_line, len(day_data))
            for idx in indices: tasks.append((time_idx, idx))

        total_pts = len(tasks)
        max_dim = (self.max_neighbors + 1) * 3
        
        # Build in CPU RAM (NumPy)
        X_cpu = np.full((total_pts, max_dim, 3), 1e6, dtype=np.float64)
        Y_cpu = np.zeros((total_pts, max_dim, 1), dtype=np.float64)
        
        # --- CHANGE 1: Increase dimension of Locs container to self.n_features (6) ---
        L_cpu = np.zeros((total_pts, max_dim, self.n_features), dtype=np.float64)

        for i, (time_idx, index) in enumerate(tasks):
            current_np = self.input_map[key_list[time_idx]]
            if isinstance(current_np, torch.Tensor): current_np = current_np.cpu().numpy()

            current_row = current_np[index].reshape(1, -1) 
            mm_neighbors = self.nns_map[index]
            past = list(mm_neighbors)
            
            data_list = []
            if past: data_list.append(current_np[past])
            if time_idx > 0: 
                prev = self.input_map[key_list[time_idx - 1]]
                if isinstance(prev, torch.Tensor): prev = prev.cpu().numpy()
                data_list.append(prev[past + [index], :])
            if time_idx > 1: 
                prev2 = self.input_map[key_list[time_idx - 2]]
                if isinstance(prev2, torch.Tensor): prev2 = prev2.cpu().numpy()
                data_list.append(prev2[past + [index], :])

            neighbors_block = np.vstack(data_list) if data_list else np.empty((0, current_row.shape[1]))
            combined = np.concatenate([neighbors_block, current_row], axis=0)
            
            slot = max_dim - combined.shape[0]
            
            # Neighbors for covariance
            X_cpu[i, slot:, 0] = combined[:, 0]
            X_cpu[i, slot:, 1] = combined[:, 1]
            X_cpu[i, slot:, 2] = combined[:, 3]
            Y_cpu[i, slot:, 0] = combined[:, 2]
            
            # --- CHANGE 2: Fill Extended Features for Global Beta ---
            # Extract raw coords
            lat_vals  = combined[:, 0]
            lon_vals  = combined[:, 1]
            time_vals = combined[:, 3]

            L_cpu[i, slot:, 0] = 1.0                # Intercept
            L_cpu[i, slot:, 1] = lat_vals           # Lat
            L_cpu[i, slot:, 2] = lon_vals           # Lon
            L_cpu[i, slot:, 3] = time_vals          # Time (Global trend)
            L_cpu[i, slot:, 4] = lat_vals * time_vals  # Lat * Time (Interaction)
            L_cpu[i, slot:, 5] = lon_vals * time_vals  # Lon * Time (Interaction)

        print("Moving to GPU...", end=" ")
        self.X_batch = torch.from_numpy(X_cpu).to(self.device)
        self.Y_batch = torch.from_numpy(Y_cpu).to(self.device)
        self.Locs_batch = torch.from_numpy(L_cpu).to(self.device)
        self.is_precomputed = True
        print(f"Done. Heads: {self.Heads_data.shape[0]}, Tails: {self.X_batch.shape[0]}")

    def batched_manual_dist(self, dist_params, x_batch):
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat, x_lon, x_time = x_batch[:, :, 0], x_batch[:, :, 1], x_batch[:, :, 2]
        
        u_lat = x_lat - advec_lat * x_time
        u_lon = x_lon - advec_lon * x_time
        
        # Broadcasting (Batch, N, N)
        d_lat = u_lat.unsqueeze(2) - u_lat.unsqueeze(1)
        d_lon = u_lon.unsqueeze(2) - u_lon.unsqueeze(1)
        d_t   = x_time.unsqueeze(2) - x_time.unsqueeze(1)

        dist_sq = (d_lat.pow(2) * phi3) + (d_lon.pow(2)) + (d_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        
        cov = (phi1/phi2) * torch.exp(-self.batched_manual_dist(dist_params, x_batch) * phi2)
        
        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def vecchia_batched_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")

        # --- GLOBAL ACCUMULATORS ---
        # Matrix size increased to (6,6) to handle interactions
        XT_Sinv_X_glob = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y_glob = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y_glob = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det_glob   = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        # -------------------------------------------------------
        # PART 1: HEADS (Exact GP Block)
        # -------------------------------------------------------
        if self.Heads_data.shape[0] > 0:
            # --- CHANGE 3: Build Extended Features for Heads ---
            # Must match the columns in Tails Locs_batch
            
            h_lat  = self.Heads_data[:, 0]
            h_lon  = self.Heads_data[:, 1]
            h_time = self.Heads_data[:, 3]
            
            # Construct Feature Matrix (N, 6)
            ones = torch.ones_like(h_lat)
            X_head = torch.stack([
                ones, 
                h_lat, 
                h_lon, 
                h_time, 
                h_lat * h_time, 
                h_lon * h_time
            ], dim=1)
            
            y_head = self.Heads_data[:, 2].unsqueeze(-1)
            
            # Compute Covariance
            cov_func = self.matern_cov_aniso_STABLE_log_reparam
            cov = cov_func(params, self.Heads_data, self.Heads_data)
            
            try:
                L = torch.linalg.cholesky(cov)
            except torch.linalg.LinAlgError:
                return torch.tensor(float('inf'), device=self.device)

            log_det_glob += 2 * torch.sum(torch.log(torch.diag(L)))
            
            # Whitening: Z = L^-1 * Data
            Z_X = torch.linalg.solve_triangular(L, X_head, upper=False)
            Z_y = torch.linalg.solve_triangular(L, y_head, upper=False)
            
            # Accumulate
            XT_Sinv_X_glob += Z_X.T @ Z_X
            XT_Sinv_y_glob += Z_X.T @ Z_y
            yT_Sinv_y_glob += (Z_y.T @ Z_y).squeeze()

        # -------------------------------------------------------
        # PART 2: TAILS (Vecchia Batches)
        # -------------------------------------------------------
        chunk_size = 4000
        total_pts = self.X_batch.shape[0]

        for start in range(0, total_pts, chunk_size):
            end = min(start + chunk_size, total_pts)
            
            X_chunk = self.X_batch[start:end]
            Y_chunk = self.Y_batch[start:end]
            Locs_chunk = self.Locs_batch[start:end] # Already has 6 dims
            
            cov_chunk = self.matern_cov_batched(params, X_chunk)
            
            try:
                L_chunk = torch.linalg.cholesky(cov_chunk) # (Batch, N, N)
            except torch.linalg.LinAlgError:
                return torch.tensor(float('inf'), device=self.device)

            # Whitening (Forward Solve)
            Z_locs = torch.linalg.solve_triangular(L_chunk, Locs_chunk, upper=False)
            Z_y    = torch.linalg.solve_triangular(L_chunk, Y_chunk, upper=False)
            
            # --- VECCHIA MAGIC ---
            # Z[:, -1, :] will now be (Batch, 6)
            u_X = Z_locs[:, -1, :] 
            u_y = Z_y[:, -1, :]    
            
            # Log Det 
            sigma_cond = L_chunk[:, -1, -1]
            log_det_glob += 2 * torch.sum(torch.log(sigma_cond)) 
            # torch sum because batch
        
            # Accumulate Global Stats
            XT_Sinv_X_glob += u_X.T @ u_X
            XT_Sinv_y_glob += u_X.T @ u_y
            yT_Sinv_y_glob += (u_y.T @ u_y).squeeze()

        # -------------------------------------------------------
        # PART 3: SOLVE GLOBAL BETA & FINALIZE NLL
        # -------------------------------------------------------
        
        # 1. Solve Beta (Generalized Least Squares)
        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-8
        try:
            beta_global = torch.linalg.solve(XT_Sinv_X_glob + jitter, XT_Sinv_y_glob)
        except torch.linalg.LinAlgError:
            return torch.tensor(float('inf'), device=self.device)

        # 2. Compute Quadratic Form 
        term1 = yT_Sinv_y_glob
        term2 = 2 * (beta_global.T @ XT_Sinv_y_glob)
        term3 = beta_global.T @ XT_Sinv_X_glob @ beta_global
        quad_form = term1 - term2 + term3

        # 3. Final NLL
        total_N = self.Heads_data.shape[0] + total_pts
        nll = 0.5 * (log_det_glob + quad_form.squeeze())
        
        return nll / total_N


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
