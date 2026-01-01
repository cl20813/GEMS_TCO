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
import torch
import numpy as np
import sys
from scipy.special import gamma
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

    # SpatioTemporalModel 클래스 내부에 추가
    
    def compute_theoretical_semivariogram_vectorized(self, params: torch.Tensor, lag_distances: torch.Tensor, time_lag: float, direction: str = 'lat') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        벡터화된 이론적 Semivariogram 계산
        direction: 'lat' 또는 'lon' (계산할 방향 지정)
        """
        # 파라미터 추출
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) 
        phi3   = torch.exp(params[2])
        phi4   = torch.exp(params[3])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2

        # --- 방향에 따른 좌표 설정 ---
        if direction == 'lat':
            lat_dist = lag_distances
            lon_dist = torch.zeros_like(lag_distances)
        elif direction == 'lon':
            lat_dist = torch.zeros_like(lag_distances)
            lon_dist = lag_distances
        else:
            raise ValueError("direction must be 'lat' or 'lon'")

        t_dist = torch.full_like(lag_distances, time_lag)

        # Advection 적용
        u_lat = lat_dist - advec_lat * t_dist
        u_lon = lon_dist - advec_lon * t_dist
        
        # 거리 제곱 계산
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (t_dist.pow(2) * phi4)
        distance = torch.sqrt(dist_sq + 1e-8)

        # Covariance & Semivariogram
        cov = sigmasq * torch.exp(-distance * phi2)
        total_sill = sigmasq + nugget
        
        semivariogram = total_sill - cov
        
        return distance, semivariogram
    

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
        
    def full_likelihood_avg(self, params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
        """Exact GP NLL calculation."""
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
        
        # Spatial Trend
        locs = torch.cat((torch.ones(N, 1, device=self.device, dtype=torch.float64), input_data[:,:2]), dim=1)
        
        if y.dim() == 1: y_col = y.unsqueeze(-1)
        else: y_col = y

        # Optimize: Solve for [X, y] together
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

        neg_log_lik_sum = 0.5 * (log_det + quad_form.squeeze())
        return neg_log_lik_sum / N




import torch
import numpy as np

import torch
import numpy as np
import time

import torch
import numpy as np
import time

class VecchiaStructuredGrid(SpatioTemporalModel):
    def __init__(self, smooth: float, input_map: dict, aggregated_data: torch.Tensor, 
                 nns_map: dict, mm_cond_number: int, 
                 n_lat: int = 114, n_lon: int = 159, n_time: int = 8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        
        self.device = aggregated_data.device
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_time = n_time
        self.is_precomputed = False
        
        # Grid Spacing Detection
        p0 = aggregated_data[0]
        p1 = aggregated_data[1]
        p_col = aggregated_data[n_lat]
        self.d_lat = torch.abs(p0[0] - p1[0]).item()
        self.d_lon = torch.abs(p0[1] - p_col[1]).item()
        
        # Storage
        self.Grouped_Batches = [] 
        self.Full_Data_Grid = None 

    def _get_idx(self, t, col, row):
        return t * (self.n_lon * self.n_lat) + col * self.n_lat + row

    def precompute_structured_conditioning(self):
        print("Pre-computing with Dynamic Stencil Grouping (User's Logic)...")
        
        # 1. Prepare Data
        full_data_list = []
        keys = sorted(list(self.input_map.keys()))
        for k in keys:
            d = self.input_map[k]
            tensor_d = torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            full_data_list.append(tensor_d.to(self.device))
        self.Full_Data_Grid = torch.cat(full_data_list, dim=0).to(torch.float64)
        
        # 2. Logic Definitions
        # "같은 줄에서는 위에 8점"
        up_range = 8 
        # "오른쪽(이전) 3줄에서는 lat +- ..." (여기선 +-4로 해서 8~9점 정도 커버한다고 가정)
        # 사용자 의도: lat +- 0.044 * 1...*8 이면 범위가 꽤 큼.
        # 최적화를 위해 +-4 (총 9개) 정도로 설정 (원하시면 늘려도 됨, 속도 영향 적음)
        prev_col_window = 4 
        prev_cols = [1, 2, 3] # c-1, c-2, c-3
        
        # 임시 저장소: { "Offset_Hash_Key": { 'offsets': tensor, 'indices': [], 'targets': [] } }
        groups = {}
        
        # Heads(경계) 제외: Row는 자동 처리되므로, Col과 Time만 여유 있게 잡음
        valid_cols = range(3, self.n_lon)
        valid_times = range(1, self.n_time)

        # --- Row Loop (0 to 113) ---
        # 모든 Row를 돌면서, 자신에게 가능한 이웃 오프셋을 계산합니다.
        for r in range(self.n_lat):
            
            # A. Calculate Valid Relative Offsets for Row 'r'
            # (이웃이 맵 밖으로 나가면 제외됨 -> 즉, 모양이 달라짐)
            
            offsets_curr = [] # (d_lat, d_lon, d_t)
            
            # 1. Same Column (Up to 8 points above)
            for k in range(1, up_range + 1):
                r_neigh = r - k
                if 0 <= r_neigh < self.n_lat: # 맵 안에 있을 때만
                    offsets_curr.append(((r - r_neigh) * self.d_lat, 0.0, 0.0))
            
            # 2. Previous Columns (3 cols, +/- window)
            for k_col in prev_cols:
                d_lon_val = k_col * self.d_lon
                # Window centered at r
                for k_row in range(-prev_col_window, prev_col_window + 1):
                    r_neigh = r + k_row # +- 적용
                    if 0 <= r_neigh < self.n_lat: # 맵 안에 있을 때만
                        # 거리는 (내위치 - 이웃위치)
                        d_lat_val = (r - r_neigh) * self.d_lat
                        offsets_curr.append((d_lat_val, d_lon_val, 0.0))
            
            # 3. Spatio-Temporal (t-1)
            # t-1은 t와 동일한 구조 + Self(1개)
            st_offsets = [list(x) for x in offsets_curr] # Copy t patterns for t
            
            # Copy t patterns for t-1 (with time lag)
            for x in offsets_curr:
                st_offsets.append((x[0], x[1], -1.0))
            
            # Self at t-1
            st_offsets.append((0.0, 0.0, -1.0))
            
            # B. Create a Unique Key for this Pattern
            # 오프셋 리스트를 튜플로 변환하여 딕셔너리 키로 사용
            # (부동소수점 오차 방지를 위해 소수점 반올림하여 문자열화)
            pattern_key = str(np.round(st_offsets, 6).tolist())
            
            # C. Collect Indices for this Row 'r' over all Valid Times/Cols
            # 이 Row 'r'은 'pattern_key' 모양을 가집니다.
            # 이 Row에 해당하는 모든 (t, c) 데이터를 모읍니다.
            
            row_indices = []
            row_targets = []
            
            tensor_offsets = torch.tensor(st_offsets, device=self.device, dtype=torch.float64)
            
            # Calculate neighbor relative indices (integer steps) to avoid re-calculation in inner loop
            # (Offset을 다시 인덱스 차이로 변환)
            rel_indices = []
            
            # Re-construct relative indices logic matching above
            # Same Col
            for k in range(1, up_range + 1):
                if 0 <= r - k < self.n_lat: rel_indices.append((0, 0, -k)) # (dt, dc, dr)
            # Prev Cols
            for k_col in prev_cols:
                for k_row in range(-prev_col_window, prev_col_window + 1):
                    if 0 <= r + k_row < self.n_lat: rel_indices.append((0, -k_col, k_row))
            # Time t-1 (Same spatial)
            n_spatial = len(rel_indices)
            spatial_rel = list(rel_indices)
            for item in spatial_rel:
                rel_indices.append((-1, item[1], item[2]))
            # Self t-1
            rel_indices.append((-1, 0, 0))

            # Inner Loop: Columns & Time
            for t in valid_times:
                for c in valid_cols:
                    target_idx = self._get_idx(t, c, r)
                    
                    neighs = []
                    for (dt, dc, dr) in rel_indices:
                        neighs.append(self._get_idx(t+dt, c+dc, r+dr))
                    
                    row_indices.append(neighs)
                    row_targets.append(target_idx)
            
            # D. Save to Groups
            if len(row_indices) > 0:
                if pattern_key not in groups:
                    groups[pattern_key] = {
                        'offsets': tensor_offsets,
                        'batch_idx': [],
                        'target_idx': []
                    }
                groups[pattern_key]['batch_idx'].extend(row_indices)
                groups[pattern_key]['target_idx'].extend(row_targets)

        # 4. Finalize Batches
        # 딕셔너리를 리스트로 변환 (각 그룹을 하나의 배치로 만듦)
        self.Grouped_Batches = []
        for key, val in groups.items():
            self.Grouped_Batches.append({
                'offsets': val['offsets'],
                'batch_idx': torch.tensor(val['batch_idx'], device=self.device, dtype=torch.long),
                'target_idx': torch.tensor(val['target_idx'], device=self.device, dtype=torch.long)
            })
            
        print(f"Precompute Done. Unique Geometric Patterns: {len(self.Grouped_Batches)}")
        # 대부분의 데이터(중간 Row들)는 하나의 그룹으로 뭉쳤을 것입니다.
        for i, batch in enumerate(self.Grouped_Batches):
            print(f"  Batch {i}: {len(batch['batch_idx'])} points (Stencil Size: {len(batch['offsets'])})")

        self.is_precomputed = True

    # ... [Likelihood 및 Solver 함수는 이전과 동일 (Cholesky 재사용)] ...
    # 다만 vecchia_structured_likelihood 에서 self.Row_Batches 대신 self.Grouped_Batches를 씁니다.

    def _compute_stencil_cov(self, offsets, params):
        phi1, phi2, phi3, phi4, advec_lat, advec_lon, log_nugget = params
        diff = offsets.unsqueeze(1) - offsets.unsqueeze(0)
        d_lat = diff[:,:,0]; d_lon = diff[:,:,1]; d_t = diff[:,:,2]
        u_lat = d_lat - advec_lat * d_t
        u_lon = d_lon - advec_lon * d_t
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (d_t.pow(2) * phi4)
        dist = torch.sqrt(dist_sq + 1e-8)
        cov = (phi1 / phi2) * torch.exp(-dist * phi2)
        cov.diagonal().add_(torch.exp(log_nugget) + 1e-6)
        return cov

    def _compute_cross_cov_vector(self, offsets, params):
        phi1, phi2, phi3, phi4, advec_lat, advec_lon, _ = params
        d_lat = offsets[:, 0]; d_lon = offsets[:, 1]; d_t = offsets[:, 2]
        u_lat = d_lat - advec_lat * d_t; u_lon = d_lon - advec_lon * d_t
        dist_sq = (u_lat.pow(2) * phi3) + (u_lon.pow(2)) + (d_t.pow(2) * phi4)
        dist = torch.sqrt(dist_sq + 1e-8)
        return (phi1 / phi2) * torch.exp(-dist * phi2)

    def _solve_tails_batched(self, batch_idx, target_idx, stencil_offsets, params):
        if batch_idx is None or batch_idx.numel() == 0: return 0.0, 0
        raw_p = [torch.exp(params[i]) if i < 4 or i == 6 else params[i] for i in range(7)]
        
        K = self._compute_stencil_cov(stencil_offsets, raw_p)
        L = torch.linalg.cholesky(K)
        
        N_batch, N_neigh = batch_idx.shape
        flat_neigh_idx = batch_idx.flatten()
        y_neighs = self.Full_Data_Grid[flat_neigh_idx, 2].view(N_batch, N_neigh, 1)
        locs_neighs = self.Full_Data_Grid[flat_neigh_idx, :2].view(N_batch, N_neigh, 2)
        y_target = self.Full_Data_Grid[target_idx, 2].view(N_batch, 1)
        locs_target = self.Full_Data_Grid[target_idx, :2] 

        X_neighs = torch.cat([torch.ones(N_batch, N_neigh, 1, device=self.device, dtype=torch.float64), locs_neighs], dim=2)
        X_target = torch.cat([torch.ones(N_batch, 1, device=self.device, dtype=torch.float64), locs_target], dim=1)
        
        L_batch = L.unsqueeze(0).expand(N_batch, -1, -1)
        z_y = torch.linalg.solve_triangular(L_batch, y_neighs, upper=False)
        z_X = torch.linalg.solve_triangular(L_batch, X_neighs, upper=False)
        
        Xt_Kinv_X = torch.bmm(z_X.transpose(1, 2), z_X) 
        Xt_Kinv_y = torch.bmm(z_X.transpose(1, 2), z_y) 
        jitter = torch.eye(3, device=self.device, dtype=torch.float64).unsqueeze(0) * 1e-6
        beta = torch.linalg.solve(Xt_Kinv_X + jitter, Xt_Kinv_y) 
        
        resid_neigh = y_neighs - torch.bmm(X_neighs, beta)
        z_r = torch.linalg.solve_triangular(L_batch, resid_neigh, upper=False)
        
        k0 = self._compute_cross_cov_vector(stencil_offsets, raw_p)
        z_k = torch.linalg.solve_triangular(L, k0.unsqueeze(1), upper=False)
        
        trend = torch.bmm(X_target.unsqueeze(1), beta).view(N_batch)
        kriging_weights = z_k.unsqueeze(0)
        cond_adjust = (kriging_weights * z_r).sum(dim=1).view(N_batch)
        mu = trend + cond_adjust
        
        total_sill = raw_p[0]/raw_p[1] + raw_p[6]
        var_reduction = torch.dot(z_k.flatten(), z_k.flatten())
        sigma_cond_sq = total_sill - var_reduction
        
        residuals_target = y_target.view(N_batch) - mu
        nll = 0.5 * (torch.log(sigma_cond_sq) + (residuals_target**2) / sigma_cond_sq)
        
        return nll.sum(), N_batch

    def vecchia_structured_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Precompute first")
        total_nll = 0.0; total_count = 0
        
        # 순회 대상이 Row_Batches -> Grouped_Batches로 변경됨
        for batch_data in self.Grouped_Batches:
            offsets = batch_data['offsets']
            b_idx = batch_data['batch_idx']
            t_idx = batch_data['target_idx']
            nll, count = self._solve_tails_batched(b_idx, t_idx, offsets, params)
            total_nll += nll; total_count += count
        return total_nll / total_count

    def fit_vecc_lbfgs_structured(self, params_list, optimizer, max_steps=50):
        if not self.is_precomputed: self.precompute_structured_conditioning()
        def closure():
            optimizer.zero_grad()
            params = torch.stack(params_list)
            loss = self.vecchia_structured_likelihood(params)
            loss.backward()
            return loss
        print(f"--- Starting Optimization ---")
        for i in range(max_steps):
            loss = optimizer.step(closure)
            grads = [p.grad.item() if p.grad is not None else 0.0 for p in params_list]
            grad_norm = np.max(np.abs(grads))
            print(f"Step {i+1} | Loss: {loss.item():.6f} | Max Grad: {grad_norm:.2e}")
            if grad_norm < 1e-5: break
        return [p.item() for p in params_list]
