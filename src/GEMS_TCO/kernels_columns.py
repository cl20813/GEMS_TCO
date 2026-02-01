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
# 2. CORE LOGIC: Sparse Safe (Dense Stencil)
# ------------------------------------------------------------------------------
class VecchiaReverseLGridSparse(SpatioTemporalModel):
    def __init__(self, smooth: float, input_map: dict, aggregated_data: torch.Tensor, 
                 nns_map: dict, mm_cond_number: int, 
                 n_lat: int = 114, n_lon: int = 159, n_time: int = 8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device
        self.d_lat = 0.044 
        self.d_lon = 0.063 
        self.n_features = 6 
        
        self.is_precomputed = False
        self.Grouped_Batches = [] 
        self.Full_Data_Grid = None 
        self.coord_to_idx = {}

    def precompute_conditioning_sets(self):
        print("🚀 Pre-computing (ULTRA DENSE Stencil: ~120 Neighbors)...")
        t0 = time.time()
        
        # 1. Prepare Data
        full_data_list = []
        keys = sorted(list(self.input_map.keys()))
        for k in keys:
            d = self.input_map[k]
            tensor_d = torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            full_data_list.append(tensor_d.to(self.device))
        
        self.Full_Data_Grid = torch.cat(full_data_list, dim=0).to(torch.float64)
        N_total = self.Full_Data_Grid.shape[0]
        
        # 2. Build Coordinate Map
        np_data = self.Full_Data_Grid.detach().cpu().numpy()
        self.coord_to_idx = {}
        
        for i in range(N_total):
            lat = round(np_data[i, 0], 5)
            lon = round(np_data[i, 1], 5)
            t   = round(np_data[i, 3], 5)
            self.coord_to_idx[(t, lat, lon)] = i
            
        # 3. Define DENSE Stencil Candidates
        # -------------------------------------------------------------
        spatial_deltas = []
        
        # [A] Vertical (Same Lon): Up 10 points
        # (dt, d_lat_step, d_lon_step)
        for k in range(1, 11): 
            spatial_deltas.append((0, k, 0)) 
            
        # [B] Horizontal (Prev 3 Cols): ~50 neighbors
        # To get ~50 neighbors from 3 cols, we need window size approx +/- 8
        # (8 up + 1 center + 8 down) * 3 cols = 17 * 3 = 51 neighbors
        prev_cols = [1, 2, 3]
        for k_col in prev_cols:
            for k_row in range(-8, 9): # -8 to +8
                spatial_deltas.append((0, k_row, -k_col))
                
        # -> Spatial Total: 10 + 51 = 61 Neighbors
        
        candidates = []
        
        # 1. Current Time Spatial Neighbors (t)
        candidates.extend(spatial_deltas)
        
        # 2. Past Time Self (t-1)
        candidates.append((-1, 0, 0))
        
        # 3. Past Time Spatial Neighbors (Mirroring t at t-1)
        # Apply the same huge stencil to the previous time step
        for (dt, dr, dc) in spatial_deltas:
            candidates.append((-1, dr, dc))
            
        # -> Grand Total: 61 (Spatial) + 1 (Self) + 61 (Temp_Spatial) = 123 Neighbors
        # -------------------------------------------------------------
        
        groups = {}
        
        # 4. Search Neighbors
        print(f"   Searching Neighbors (Stencil Candidates: {len(candidates)})...")
        
        for i in range(N_total):
            t_curr   = round(np_data[i, 3], 5)
            lat_curr = round(np_data[i, 0], 5)
            lon_curr = round(np_data[i, 1], 5)
            
            valid_neigh_indices = []
            offsets_list = []
            
            for (dt, d_lat_step, d_lon_step) in candidates:
                t_neigh   = round(t_curr + dt * 1.0, 5)
                lat_neigh = round(lat_curr + d_lat_step * self.d_lat, 5)
                lon_neigh = round(lon_curr + d_lon_step * self.d_lon, 5)
                
                # Sparse Check (No Dummy)
                if (t_neigh, lat_neigh, lon_neigh) in self.coord_to_idx:
                    neigh_idx = self.coord_to_idx[(t_neigh, lat_neigh, lon_neigh)]
                    valid_neigh_indices.append(neigh_idx)
                    
                    real_d_lat = lat_curr - lat_neigh
                    real_d_lon = lon_curr - lon_neigh
                    real_d_t   = t_curr - t_neigh
                    offsets_list.append((real_d_lat, real_d_lon, real_d_t))
            
            if not offsets_list:
                pattern_key = "empty"
            else:
                pattern_key = str(np.round(offsets_list, 6).tolist())
            
            if pattern_key not in groups:
                t_offsets = torch.tensor(offsets_list, device=self.device, dtype=torch.float64) if offsets_list else torch.empty((0,3), device=self.device)
                groups[pattern_key] = {
                    'offsets': t_offsets,
                    'batch_idx': [],
                    'target_idx': []
                }
            groups[pattern_key]['batch_idx'].append(valid_neigh_indices)
            groups[pattern_key]['target_idx'].append(i)

        # 5. Finalize Batches
        self.Grouped_Batches = []
        for key, val in groups.items():
            if len(val['target_idx']) == 0: continue
            
            if key == "empty":
                 offsets = torch.empty((0, 3), device=self.device, dtype=torch.float64)
                 b_idx = torch.empty((len(val['target_idx']), 0), device=self.device, dtype=torch.long)
            else:
                 offsets = val['offsets']
                 b_idx = torch.tensor(val['batch_idx'], device=self.device, dtype=torch.long)
            
            t_idx = torch.tensor(val['target_idx'], device=self.device, dtype=torch.long)
            
            self.Grouped_Batches.append({'offsets': offsets, 'batch_idx': b_idx, 'target_idx': t_idx})

        self.is_precomputed = True
        print(f"✅ Precompute Done. Unique Patterns: {len(self.Grouped_Batches)}")
        print(f"   Total Time: {time.time() - t0:.2f}s")


    # --- KERNEL FUNCTIONS ---
    def _compute_stencil_cov_matrix(self, offsets, params):
        diff = offsets.unsqueeze(1) - offsets.unsqueeze(0)
        phi1, phi2, phi3, phi4, advec_lat, advec_lon, log_nugget = params
        d_lat, d_lon, d_t = diff[:,:,0], diff[:,:,1], diff[:,:,2]
        u_lat = d_lat - advec_lat * d_t; u_lon = d_lon - advec_lon * d_t
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (d_t.pow(2) * phi4)
        dist = torch.sqrt(dist_sq + 1e-8)
        cov = (phi1 / phi2) * torch.exp(-dist * phi2)
        cov.diagonal().add_(torch.exp(log_nugget) + 1e-6)
        return cov

    def _compute_cross_cov_vector(self, offsets, params):
        phi1, phi2, phi3, phi4, advec_lat, advec_lon, _ = params
        d_lat, d_lon, d_t = offsets[:, 0], offsets[:, 1], offsets[:, 2]
        u_lat = d_lat - advec_lat * d_t; u_lon = d_lon - advec_lon * d_t
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (d_t.pow(2) * phi4)
        dist = torch.sqrt(dist_sq + 1e-8)
        return (phi1 / phi2) * torch.exp(-dist * phi2)


    # --- LIKELIHOOD FUNCTION ---
    def vecchia_structured_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")

        XT_Sinv_X_glob = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y_glob = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y_glob = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det_glob   = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        raw_p = [torch.exp(params[i]) if i < 4 or i == 6 else params[i] for i in range(7)]
        total_sill = (raw_p[0]/raw_p[1] + raw_p[6]).squeeze()

        for batch_data in self.Grouped_Batches:
            offsets = batch_data['offsets']; b_idx = batch_data['batch_idx']; t_idx = batch_data['target_idx']
            N_batch = len(t_idx); N_neigh = offsets.shape[0]
            
            y_target = self.Full_Data_Grid[t_idx, 2].view(N_batch, 1)
            target_rows = self.Full_Data_Grid[t_idx]
            lat = target_rows[:, 0:1]; lon = target_rows[:, 1:2]; time = target_rows[:, 3:4]
            X_target = torch.cat([torch.ones(N_batch,1,device=self.device,dtype=torch.float64), 
                                  lat, lon, time, lat*time, lon*time], dim=1)

            if N_neigh == 0:
                inv_s = 1.0 / total_sill
                XT_Sinv_X_glob += (X_target.T @ X_target) * inv_s
                XT_Sinv_y_glob += (X_target.T @ y_target) * inv_s
                yT_Sinv_y_glob += ((y_target.T @ y_target).squeeze() * inv_s).squeeze()
                log_det_glob   += N_batch * torch.log(total_sill)
                
            else:
                K = self._compute_stencil_cov_matrix(offsets, raw_p)
                k = self._compute_cross_cov_vector(offsets, raw_p)
                try:
                    L = torch.linalg.cholesky(K)
                    z = torch.linalg.solve_triangular(L, k.unsqueeze(1), upper=False)
                except: return torch.tensor(float('inf'), device=self.device)
                
                sigma_cond = total_sill - torch.dot(z.flatten(), z.flatten())
                if sigma_cond <= 0: sigma_cond = torch.tensor(1e-6, device=self.device, dtype=torch.float64)
                
                flat_b = b_idx.flatten()
                y_n = self.Full_Data_Grid[flat_b, 2].view(N_batch, N_neigh)
                rows_n = self.Full_Data_Grid[flat_b]
                n_lat = rows_n[:, 0:1]; n_lon = rows_n[:, 1:2]; n_time = rows_n[:, 3:4]
                X_n = torch.cat([torch.ones(N_batch*N_neigh,1,device=self.device,dtype=torch.float64),
                                 n_lat, n_lon, n_time, n_lat*n_time, n_lon*n_time], dim=1).view(N_batch, N_neigh, 6)
                
                w = torch.linalg.solve_triangular(L.T, z, upper=True).flatten()
                y_eff = y_target - (y_n @ w).unsqueeze(1)
                X_eff = X_target - torch.einsum('bmf,m->bf', X_n, w)
                
                inv_s = 1.0 / sigma_cond
                XT_Sinv_X_glob += (X_eff.T @ X_eff) * inv_s
                XT_Sinv_y_glob += (X_eff.T @ y_eff) * inv_s
                yT_Sinv_y_glob += ((y_eff.T @ y_eff).squeeze() * inv_s).squeeze()
                log_det_glob += N_batch * torch.log(sigma_cond)

        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-8
        try: beta = torch.linalg.solve(XT_Sinv_X_glob + jitter, XT_Sinv_y_glob)
        except: return torch.tensor(float('inf'), device=self.device)
        
        quad = yT_Sinv_y_glob - (2 * beta.T @ XT_Sinv_y_glob).squeeze() + (beta.T @ XT_Sinv_X_glob @ beta).squeeze()
        return 0.5 * (log_det_glob + quad) / self.Full_Data_Grid.shape[0]

# ------------------------------------------------------------------------------
# 3. FITTING CLASS
# ------------------------------------------------------------------------------
class fit_vecchia_lbfgs(VecchiaReverseLGridSparse): 
    
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
        print("--- Starting Optimization (Dense Stencil) ---")

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