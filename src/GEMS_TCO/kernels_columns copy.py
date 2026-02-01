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
import sys
import time
import copy
import logging
import warnings
from typing import Dict, Any, Callable, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.neighbors import BallTree  # [중요] 희소 데이터 검색용

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1. BASE CLASS
# ------------------------------------------------------------------------------
class SpatioTemporalModel:
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
        self.device = aggregated_data.device
        self.smooth = smooth
        self.smooth_tensor = torch.tensor(self.smooth, dtype=torch.float64, device=self.device)
        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64, device=self.device)
        
        self.input_map = input_map
        self.aggregated_data = aggregated_data 
        self.key_list = list(input_map.keys())
        self.mm_cond_number = mm_cond_number
        self.nns_map = nns_map 

# ------------------------------------------------------------------------------
# 2. CORE LOGIC: Vecchia Reverse L-Shape (Padded & Fast)
# ------------------------------------------------------------------------------
class VecchiaReverseLGridPadded(SpatioTemporalModel):
    def __init__(self, smooth: float, input_map: dict, aggregated_data: torch.Tensor, 
                 nns_map: dict, mm_cond_number: int, 
                 n_lat: int = 114, n_lon: int = 159, n_time: int = 8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device
        self.d_lat = 0.044 
        self.d_lon = 0.063 
        self.n_features = 6 
        
        self.is_precomputed = False
        
        # Batches
        self.Indices_Tensor = None
        self.Offsets_Tensor = None
        self.Target_Tensor = None

    def precompute_conditioning_sets(self):
        print("🚀 Pre-computing (Padding Strategy)...")
        t0 = time.time()
        
        # 1. Prepare Data
        full_data_list = []
        keys = sorted(list(self.input_map.keys()))
        for k in keys:
            d = self.input_map[k]
            tensor_d = torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            full_data_list.append(tensor_d.to(self.device))
        
        real_data = torch.cat(full_data_list, dim=0).to(torch.float64)
        N_real = real_data.shape[0]
        N_cols = real_data.shape[1] # [수정] 컬럼 수 자동 감지 (4개)
        
        # [수정] Dummy Point 생성 (데이터 컬럼 수에 맞춤)
        # Lat=1e8, Lon=1e8, Val=0, Time=0 ...
        dummy_row = torch.zeros((1, N_cols), device=self.device, dtype=torch.float64)
        dummy_row[0, 0] = 1e8 # Lat
        dummy_row[0, 1] = 1e8 # Lon
        
        self.Full_Data_Grid = torch.cat([real_data, dummy_row], dim=0) 
        dummy_idx = N_real 
        
        # 2. Build Trees (CPU)
        np_data = real_data.detach().cpu().numpy()
        coords_spatial = np_data[:, :2]
        times = np_data[:, 3]
        
        tree_by_time = {}
        indices_by_time = {}
        unique_times = np.unique(times)
        
        print(f"   Building Trees for {len(unique_times)} time steps...")
        for t in unique_times:
            mask = (times == t)
            idxs = np.where(mask)[0]
            if len(idxs) == 0: continue
            tree_by_time[t] = BallTree(coords_spatial[idxs], metric='euclidean')
            indices_by_time[t] = idxs

        # 3. Define Fixed Stencil (13 Neighbors)
        stencil_deltas = []
        for k in range(1, 4): stencil_deltas.append((0, k, 0)) # Vertical
        for k_col in [1, 2, 3]: # Horizontal
            for k_row in range(-1, 2): stencil_deltas.append((0, k_row, -k_col))
        stencil_deltas.append((-1, 0, 0)) # Temporal
        
        # 4. Global Batch Arrays
        all_neighbor_indices = []
        all_offsets = []
        search_radius = 0.015
        
        print("   Searching & Padding...")
        
        for t_curr in unique_times:
            if t_curr not in indices_by_time: continue
            curr_indices = indices_by_time[t_curr]
            curr_coords = coords_spatial[curr_indices]
            
            for i_local, idx_global in enumerate(curr_indices):
                lat_c, lon_c = curr_coords[i_local]
                
                row_neigh_indices = []
                row_offsets = []
                
                for (dt, d_lat_step, d_lon_step) in stencil_deltas:
                    t_target = t_curr + dt
                    
                    found = False
                    if t_target in tree_by_time:
                        ideal_lat = lat_c + d_lat_step * self.d_lat
                        ideal_lon = lon_c + d_lon_step * self.d_lon
                        
                        ind = tree_by_time[t_target].query_radius([[ideal_lat, ideal_lon]], r=search_radius)
                        
                        if len(ind[0]) > 0:
                            found_idx = indices_by_time[t_target][ind[0][0]]
                            row_neigh_indices.append(found_idx)
                            
                            d_lat = lat_c - coords_spatial[found_idx][0]
                            d_lon = lon_c - coords_spatial[found_idx][1]
                            d_t   = t_curr - t_target
                            row_offsets.append([d_lat, d_lon, d_t])
                            found = True
                    
                    if not found:
                        # Padding
                        row_neigh_indices.append(dummy_idx)
                        row_offsets.append([0.0, 0.0, 100.0]) 
                
                all_neighbor_indices.append(row_neigh_indices)
                all_offsets.append(row_offsets)
                
        # 5. Create Tensors
        self.Indices_Tensor = torch.tensor(all_neighbor_indices, device=self.device, dtype=torch.long)
        self.Offsets_Tensor = torch.tensor(all_offsets, device=self.device, dtype=torch.float64)
        
        sorted_targets = []
        for t in unique_times:
            if t in indices_by_time: sorted_targets.extend(indices_by_time[t])
        self.Target_Tensor = torch.tensor(sorted_targets, device=self.device, dtype=torch.long)

        self.is_precomputed = True
        print(f"✅ Done. Batch Shape: {self.Indices_Tensor.shape}")
        print(f"   Total Time: {time.time() - t0:.2f}s")


    # --- LIKELIHOOD ---
    def vecchia_structured_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")

        raw_p = [torch.exp(params[i]) if i < 4 or i == 6 else params[i] for i in range(7)]
        phi1, phi2, phi3, phi4, advec_lat, advec_lon, nugget = raw_p
        
        total_sill = (phi1/phi2 + nugget).squeeze()
        
        # 1. Covariance Matrix (Batched)
        M = self.Offsets_Tensor.shape[1]
        offsets = self.Offsets_Tensor 
        
        diff = offsets.unsqueeze(2) - offsets.unsqueeze(1) 
        d_lat = diff[..., 0]; d_lon = diff[..., 1]; d_t = diff[..., 2]
        
        u_lat = d_lat - advec_lat * d_t
        u_lon = d_lon - advec_lon * d_t
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (d_t.pow(2) * phi4)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        K = (phi1 / phi2) * torch.exp(-dist * phi2) 
        eye = torch.eye(M, device=self.device, dtype=torch.float64).unsqueeze(0)
        K = K + eye * (nugget + 1e-6)
        
        # 2. Cross Covariance
        d_lat_c = offsets[..., 0]; d_lon_c = offsets[..., 1]; d_t_c = offsets[..., 2]
        u_lat_c = d_lat_c - advec_lat * d_t_c
        u_lon_c = d_lon_c - advec_lon * d_t_c
        dist_c_sq = (u_lat_c.pow(2) * phi3) + (u_lon_c.pow(2)) + (d_t_c.pow(2) * phi4)
        dist_c = torch.sqrt(dist_c_sq + 1e-8)
        
        k_cross = (phi1 / phi2) * torch.exp(-dist_c * phi2) 
        
        # 3. Cholesky Solve
        try:
            L = torch.linalg.cholesky(K)
            z = torch.linalg.solve_triangular(L, k_cross.unsqueeze(2), upper=False)
        except: return torch.tensor(float('inf'), device=self.device)
        
        # 4. Sigma Cond
        var_reduction = torch.sum(z.squeeze()**2, dim=1) 
        sigma_cond = total_sill - var_reduction
        sigma_cond = torch.clamp(sigma_cond, min=1e-6)
        
        # 5. Weights
        w = torch.linalg.solve_triangular(L.transpose(1,2), z, upper=True).squeeze(2)
        
        # 6. Data Extraction
        y_target = self.Full_Data_Grid[self.Target_Tensor, 2] 
        target_rows = self.Full_Data_Grid[self.Target_Tensor]
        lat = target_rows[:, 0]; lon = target_rows[:, 1]; time = target_rows[:, 3]
        ones = torch.ones_like(lat)
        X_target = torch.stack([ones, lat, lon, time, lat*time, lon*time], dim=1) 
        
        flat_neigh_idx = self.Indices_Tensor.flatten()
        y_neigh = self.Full_Data_Grid[flat_neigh_idx, 2].view(-1, M)
        
        neigh_rows = self.Full_Data_Grid[flat_neigh_idx]
        n_lat = neigh_rows[:, 0]; n_lon = neigh_rows[:, 1]; n_time = neigh_rows[:, 3]
        n_ones = torch.ones_like(n_lat)
        X_neigh = torch.stack([n_ones, n_lat, n_lon, n_time, n_lat*n_time, n_lon*n_time], dim=1).view(-1, M, 6)
        
        # 7. Effective Residuals
        weighted_y_n = (y_neigh * w).sum(dim=1)
        y_eff = (y_target - weighted_y_n).unsqueeze(1) 
        
        weighted_X_n = torch.einsum('nmf,nm->nf', X_neigh, w)
        X_eff = X_target - weighted_X_n 
        
        # 8. Beta Solve
        inv_sigma = 1.0 / sigma_cond.unsqueeze(1) 
        X_eff_scaled = X_eff * torch.sqrt(inv_sigma)
        y_eff_scaled = y_eff * torch.sqrt(inv_sigma)
        
        XT_Sinv_X = X_eff_scaled.T @ X_eff_scaled
        XT_Sinv_y = X_eff_scaled.T @ y_eff_scaled
        
        jitter_mat = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-8
        try: beta = torch.linalg.solve(XT_Sinv_X + jitter_mat, XT_Sinv_y)
        except: return torch.tensor(float('inf'), device=self.device)
        
        # 9. NLL
        resid_scaled = y_eff_scaled - (X_eff_scaled @ beta)
        quad_form = (resid_scaled**2).sum()
        log_det = torch.log(sigma_cond).sum()
        
        nll = 0.5 * (log_det + quad_form)
        return nll / self.Target_Tensor.shape[0]

# ------------------------------------------------------------------------------
# 3. FITTING CLASS
# ------------------------------------------------------------------------------
class fit_vecchia_lbfgs(VecchiaReverseLGridPadded): 
    
    def __init__(self, smooth: float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, 
                 nns_map: Dict[str, Any], mm_cond_number: int, 
                 n_lat: int = 114, n_lon: int = 159, n_time: int = 8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, 
                         n_lat, n_lon, n_time)

    def set_optimizer(self, param_groups, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100):
        optimizer = torch.optim.LBFGS(
            param_groups, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size
        )
        return optimizer

    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        try:
            phi1 = np.exp(raw_params_list[0])
            phi2 = np.exp(raw_params_list[1])
            phi3 = np.exp(raw_params_list[2])
            phi4 = np.exp(raw_params_list[3])
            return {
                "sigma_sq": phi1 / phi2,
                "range_lon": 1.0 / phi2,
                "range_lat": (1.0 / phi2) / np.sqrt(phi3),
                "range_time": 1.0 / (np.sqrt(phi4) * phi2),
                "advec_lat": raw_params_list[4],
                "advec_lon": raw_params_list[5],
                "nugget": np.exp(raw_params_list[6])
            }
        except: return {}

    def fit_vecc_lbfgs(self, params_list: List[torch.Tensor], optimizer: torch.optim.LBFGS, max_steps: int = 50, grad_tol: float = 1e-7):
        if not self.is_precomputed: self.precompute_conditioning_sets()
        print("--- Starting Optimization (Padded & Fast) ---")

        def closure():
            optimizer.zero_grad()
            params = torch.stack(params_list)
            loss = self.vecchia_structured_likelihood(params)
            if torch.isfinite(loss) and loss.requires_grad: loss.backward()
            return loss

        for i in range(max_steps):
            loss = optimizer.step(closure)
            with torch.no_grad():
                grads = [abs(p.grad.item()) for p in params_list if p.grad is not None]
                max_grad = max(grads) if grads else 0.0
                print(f"Step {i+1} | Loss: {loss.item():.6f} | Max Grad: {max_grad:.2e}")
            if max_grad < grad_tol: break

        final_vals = [p.item() for p in params_list]
        print("Final Params:", self._convert_raw_params_to_interpretable(final_vals))
        return final_vals, i