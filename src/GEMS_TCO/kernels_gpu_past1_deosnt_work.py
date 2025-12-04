import sys
# gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
# sys.path.append(gems_tco_path)
# import GEMS_TCO
from scipy.special import gamma, kv 

from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Callable, List, Tuple, Union
import time
import copy    
import warnings
from functools import partial

# --- BASE CLASS ---
class SpatioTemporalModel:
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map: Union[Dict[str, Any], List], mm_cond_number: int):
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

        # --- FIX: ROBUST NNS MAP PROCESSING ---
        # Handle Dictionary vs List inputs separately to avoid data loss
        if isinstance(nns_map, dict):
            # Process in-place without converting to list
            self.nns_map = {}
            for k, v in nns_map.items():
                # Take first mm_cond_number and filter -1
                clean_v = np.array(v)[:self.mm_cond_number]
                clean_v = clean_v[clean_v != -1]
                self.nns_map[k] = clean_v
        else:
            # Assume List
            nns_map_clean = []
            for item in nns_map:
                tmp = np.array(item)[:self.mm_cond_number]
                tmp = tmp[tmp != -1]
                nns_map_clean.append(tmp)
            self.nns_map = nns_map_clean

    # --- SINGLE MATRIX KERNEL (For "Heads" / Exact GP) ---
    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi3, phi4, advec_lat, advec_lon = dist_params
        u_lat_adv = x[:, 0] - advec_lat * x[:, 3]
        u_lon_adv = x[:, 1] - advec_lon * x[:, 3]
        u_t = x[:, 3]
        v_lat_adv = y[:, 0] - advec_lat * y[:, 3]
        v_lon_adv = y[:, 1] - advec_lon * y[:, 3]
        v_t = y[:, 3]
        delta_lat = u_lat_adv.unsqueeze(1) - v_lat_adv.unsqueeze(0)
        delta_lon = u_lon_adv.unsqueeze(1) - v_lon_adv.unsqueeze(0)
        delta_t   = u_t.unsqueeze(1)       - v_t.unsqueeze(0)
        dist_sq = (delta_lat.pow(2) * phi3) + delta_lon.pow(2) + (delta_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat, advec_lon = params[4], params[5]
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
        C_inv_combined = torch.cholesky_solve(combined_rhs, L, upper=False)
        C_inv_X, C_inv_y = C_inv_combined[:, :3], C_inv_combined[:, 3:]
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
        return 0.5 * (log_det + quad_form.squeeze()) / N


# --- OPTIMIZED BATCHED GPU CLASS (Heads Preserved) ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device 
        self.nheads = nheads
        self.is_precomputed = False
        
        self.X_t0 = None
        self.Y_t0 = None
        self.Locs_t0 = None
        
        self.X_template = None    
        self.Locs_template = None 
        self.Y_steady = None      
        
        self.Heads_data = None 
        self.tail_indices = None

    def precompute_conditioning_sets(self):
        print("Pre-computing Optimized Stationary Tensors...", end=" ")
        
        key_list = list(self.input_map.keys())
        num_times = len(key_list)
        
        cut_line = self.nheads
        total_points = len(self.input_map[key_list[0]])
        self.tail_indices = np.arange(cut_line, total_points)
        n_tails = len(self.tail_indices)
        
        print(f"[Heads: {cut_line} pts/time] [Tails: {n_tails} pts/time]...", end=" ")

        # --- 2. Prepare Heads Data (Exact GP) ---
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, np.ndarray):
                head_chunk = torch.from_numpy(day_data[:cut_line]).to(self.device)
            else:
                head_chunk = day_data[:cut_line].clone().detach().to(self.device)
            heads_list.append(head_chunk)
        self.Heads_data = torch.cat(heads_list, dim=0).to(torch.float64)

        # --- 3. Prepare Neighborhoods (Indices) ---
        neighbor_map = {}
        for idx in self.tail_indices:
            # Safe access (handled by __init__)
            try:
                valid_n = self.nns_map[idx]
            except (KeyError, IndexError):
                 raise IndexError(f"Index {idx} not found in nns_map.")
            neighbor_map[idx] = valid_n

        # --- 4. Build T=0 Batch (Spatial Only) ---
        dim_t0 = self.mm_cond_number + 1
        
        self.X_t0 = torch.full((n_tails, dim_t0, 3), 1e6, device=self.device, dtype=torch.float64)
        self.Y_t0 = torch.zeros((n_tails, dim_t0, 1), device=self.device, dtype=torch.float64)
        self.Locs_t0 = torch.zeros((n_tails, dim_t0, 3), device=self.device, dtype=torch.float64)
        
        t0_data = self.input_map[key_list[0]]
        if isinstance(t0_data, np.ndarray): t0_data = torch.from_numpy(t0_data).to(self.device)
        else: t0_data = t0_data.to(self.device)
        
        for i, idx in enumerate(self.tail_indices):
            past = neighbor_map[idx] 
            
            # Gather Data: [Neighbors; Target]
            block_indices = np.append(past, idx)
            block_data = t0_data[block_indices] 
            
            k = block_data.shape[0]
            start = dim_t0 - k
            
            self.X_t0[i, start:, 0] = block_data[:, 0]
            self.X_t0[i, start:, 1] = block_data[:, 1]
            self.X_t0[i, start:, 2] = block_data[:, 3]
            self.Y_t0[i, start:, 0] = block_data[:, 2]
            
            self.Locs_t0[i, start:, 0] = 1.0
            self.Locs_t0[i, start:, 1] = block_data[:, 0]
            self.Locs_t0[i, start:, 2] = block_data[:, 1]

        # --- 5. Build Steady Template (T >= 1) ---
        # Optimization: Cache the pattern.
        # Structure: [Spatial_T(neighbors) ; Spatial_T-1(neighbors+self) ; Target_T]
        dim_steady = (self.mm_cond_number * 2) + 2
        
        self.X_template = torch.full((n_tails, dim_steady, 3), 1e6, device=self.device, dtype=torch.float64)
        self.Locs_template = torch.zeros((n_tails, dim_steady, 3), device=self.device, dtype=torch.float64)
        self.Y_steady = torch.zeros((num_times-1, n_tails, dim_steady, 1), device=self.device, dtype=torch.float64)

        all_data_np = np.stack([self.input_map[k] for k in key_list])
        all_data = torch.tensor(all_data_np, device=self.device, dtype=torch.float64)

        for i, idx in enumerate(self.tail_indices):
            past = neighbor_map[idx]
            
            # Construct Relative Coordinate Block
            coords_n = all_data[0, past, :2]
            coords_self = all_data[0, idx, :2].unsqueeze(0)
            
            # Part 1: Neighbors Current (Time 0)
            blk1_coords = torch.cat([coords_n, torch.zeros((len(past), 1), device=self.device, dtype=torch.float64)], dim=1)
            # Part 2: Neighbors + Self Past (Time -1)
            past_set = torch.cat([coords_n, coords_self], dim=0)
            blk2_coords = torch.cat([past_set, torch.full((len(past_set), 1), -1.0, device=self.device, dtype=torch.float64)], dim=1)
            # Part 3: Target Current (Time 0)
            blk3_coords = torch.cat([coords_self, torch.zeros((1, 1), device=self.device, dtype=torch.float64)], dim=1)
            
            full_block = torch.cat([blk1_coords, blk2_coords, blk3_coords], dim=0)
            
            k = full_block.shape[0]
            start = dim_steady - k
            
            self.X_template[i, start:, :] = full_block
            
            self.Locs_template[i, start:, 0] = 1.0
            self.Locs_template[i, start:, 1] = full_block[:, 0]
            self.Locs_template[i, start:, 2] = full_block[:, 1]
            
            # --- Y Values for ALL Times (1 to T) ---
            # Part 1 Vals: Neighbors at T
            y_p1 = all_data[1:, past, 2] # (Time-1, n_neighbors)
            # Part 2 Vals: Neighbors+Self at T-1
            past_indices = np.append(past, idx)
            y_p2 = all_data[:-1, past_indices, 2] # (Time-1, n_neighbors+1)
            # Part 3 Vals: Target at T
            y_p3 = all_data[1:, idx, 2].unsqueeze(1) # (Time-1, 1)
            
            y_combined = torch.cat([y_p1, y_p2, y_p3], dim=1) # (Time-1, Total_K)
            self.Y_steady[:, i, start:, 0] = y_combined

        self.is_precomputed = True
        print("Done.")

    def batched_manual_dist(self, dist_params, x_batch):
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat, x_lon, x_time = x_batch[..., 0], x_batch[..., 1], x_batch[..., 2]
        u_lat_adv = x_lat - advec_lat * x_time
        u_lon_adv = x_lon - advec_lon * x_time
        u_time    = x_time
        delta_lat = u_lat_adv.unsqueeze(-2) - u_lat_adv.unsqueeze(-3)
        delta_lon = u_lon_adv.unsqueeze(-2) - u_lon_adv.unsqueeze(-3)
        delta_t   = u_time.unsqueeze(-2)    - u_time.unsqueeze(-3)
        dist_sq = (delta_lat.pow(2) * phi3) + (delta_lon.pow(2)) + (delta_t.pow(2) * phi4)
        return torch.sqrt(dist_sq + 1e-8)

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat, advec_lon = params[4], params[5]
        sigmasq = phi1 / phi2 
        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.batched_manual_dist(dist_params, x_batch)
        cov = sigmasq * torch.exp(-distance * phi2)
        eye_shape = cov.shape[-2:]
        identity = torch.eye(eye_shape[0], device=self.device, dtype=torch.float64)
        cov = cov + identity * (nugget + 1e-6)
        return cov

    def vecchia_batched_likelihood(self, params):
            if not self.is_precomputed: raise RuntimeError("Run precompute first!")
            
            # --- 1. Heads (Exact GP) ---
            def adapter_cov_func(params, x, y):
                return self.matern_cov_aniso_STABLE_log_reparam(params, x, y)

            head_nll_sum = 0.0
            if self.Heads_data.shape[0] > 0:
                heads_reshaped = self.Heads_data.view(len(self.input_map), self.nheads, -1)
                for t in range(heads_reshaped.shape[0]):
                    day_head = heads_reshaped[t]
                    head_nll_sum += self.full_likelihood_avg(params, day_head, day_head[:, 2], adapter_cov_func) * day_head.shape[0]

            # --- 2. Tails T=0 (Spatial Only) ---
            cov_t0 = self.matern_cov_batched(params, self.X_t0)
            try:
                L_t0 = torch.linalg.cholesky(cov_t0)
                C_inv_locs = torch.cholesky_solve(self.Locs_t0, L_t0, upper=False)
                Xt_Cinv_X = torch.bmm(self.Locs_t0.transpose(1, 2), C_inv_locs)
                C_inv_y = torch.cholesky_solve(self.Y_t0, L_t0, upper=False)
                Xt_Cinv_y = torch.bmm(self.Locs_t0.transpose(1, 2), C_inv_y)
                jitter = torch.eye(3, device=self.device, dtype=torch.float64).unsqueeze(0) * 1e-8
                beta_t0 = torch.linalg.solve(Xt_Cinv_X + jitter, Xt_Cinv_y)
                mu_t0 = torch.bmm(self.Locs_t0, beta_t0)
                res_t0 = self.Y_t0 - mu_t0
                z_t0 = torch.linalg.solve_triangular(L_t0, res_t0, upper=False)[:, -1, :]
                nll_t0 = 0.5 * (2 * torch.log(L_t0[:, -1, -1]) + z_t0.squeeze()**2).sum()
            except torch.linalg.LinAlgError:
                return torch.inf

            # --- 3. Tails T>=1 (Steady State Broadcast - CHUNKED) ---
            # A. Compute Spatial Template ONCE (Expensive part)
            cov_template = self.matern_cov_batched(params, self.X_template) 
            try:
                L_template = torch.linalg.cholesky(cov_template) 
            except torch.linalg.LinAlgError:
                return torch.inf

            # B. Prepare Broadcasting Tensors
            L_broad = L_template.unsqueeze(0)      # (1, N_tails, M, M)
            Locs_broad = self.Locs_template.unsqueeze(0) # (1, N_tails, M, 3)
            
            # C. Solve for Spatial Trend (X) ONCE
            C_inv_locs = torch.cholesky_solve(Locs_broad, L_broad, upper=False) 
            Xt_Cinv_X = torch.matmul(Locs_broad.transpose(-1, -2), C_inv_locs)  
            jitter = torch.eye(3, device=self.device, dtype=torch.float64).view(1, 1, 3, 3) * 1e-8

            # D. Loop through Y in CHUNKS to prevent OOM
            #    This keeps the speed of reusing L, but keeps memory usage low.
            chunk_size = 200  # Adjust this: 100-500 is usually safe for 24GB GPUs
            total_time_steps = self.Y_steady.shape[0]
            quad_form_sum = 0.0
            
            for start_t in range(0, total_time_steps, chunk_size):
                end_t = min(start_t + chunk_size, total_time_steps)
                
                # Slice Y (Cheap view)
                Y_chunk = self.Y_steady[start_t:end_t] 
                
                # Solve Y against the cached L (Broadcasting happens here)
                C_inv_y = torch.cholesky_solve(Y_chunk, L_broad, upper=False) 
                Xt_Cinv_y = torch.matmul(Locs_broad.transpose(-1, -2), C_inv_y)     
                
                # Solve Beta for this chunk
                beta_chunk = torch.linalg.solve(Xt_Cinv_X + jitter, Xt_Cinv_y)
                
                # Compute Residuals
                mu_chunk = torch.matmul(Locs_broad, beta_chunk)
                res_chunk = Y_chunk - mu_chunk
                
                # Compute Quadratic Form Contribution
                z_chunk = torch.linalg.solve_triangular(L_broad, res_chunk, upper=False)
                z_target = z_chunk[..., -1, :] 
                quad_form_sum = quad_form_sum + (z_target.squeeze() ** 2).sum()

            # E. Log Determinant (Computed once, multiplied by total counts)
            sigma_cond = L_template[:, -1, -1]
            log_det_total = 2 * torch.log(sigma_cond).sum() * total_time_steps
            
            nll_steady = 0.5 * (log_det_total + quad_form_sum)

            total_nll = head_nll_sum + nll_t0 + nll_steady
            total_count = self.Heads_data.shape[0] + self.X_t0.shape[0] + (self.Y_steady.shape[0] * self.Y_steady.shape[1])
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
        try:
            log_phi1, log_phi2, log_phi3, log_phi4 = raw_params_list[0], raw_params_list[1], raw_params_list[2], raw_params_list[3]
            advec_lat, advec_lon, log_nugget = raw_params_list[4], raw_params_list[5], raw_params_list[6]
            phi1, phi2, phi3, phi4, nugget = np.exp(log_phi1), np.exp(log_phi2), np.exp(log_phi3), np.exp(log_phi4), np.exp(log_nugget)
            return {
                "sigma_sq": phi1 / phi2,
                "range_lon": 1.0 / phi2,
                "range_lat": (1.0 / phi2) / np.sqrt(phi3),
                "range_time": 1/(np.sqrt(phi4) * phi2),
                "advec_lat": advec_lat,
                "advec_lon": advec_lon,
                "nugget": nugget
            }
        except Exception as e:
            print(f"Warning: {e}")
            return {}

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-7):
        if not self.is_precomputed:
            self.precompute_conditioning_sets()

        print("--- Starting Batched L-BFGS Optimization (GPU Optimized) ---")

        def closure():
            optimizer.zero_grad()
            params = torch.cat(params_list)
            loss = self.vecchia_batched_likelihood(params)
            loss.backward()
            return loss

        for i in range(max_steps):
            loss = optimizer.step(closure)
            max_abs_grad = 0.0
            with torch.no_grad():
                grad_values = [abs(p.grad.item()) for p in params_list if p.grad is not None]
                if grad_values: max_abs_grad = max(grad_values)
                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)

            if max_abs_grad < grad_tol:
                print(f"\nConverged at step {i+1}")
                break

        final_params_tensor = torch.cat(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        interpretable = self._convert_raw_params_to_interpretable(final_raw_params_list)
        print("Final Interpretable Params:", interpretable)
        return final_raw_params_list + [final_loss], i
