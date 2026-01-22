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
    '''  3 features
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
    '''

    ''' 6 features

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
        
        # --- [MODIFIED] Construct 6 Features (Same as VecchiaBatched) ---
        # Assuming input_data columns: [Lat(0), Lon(1), Val(2), Time(3)] 
        # based on your other methods using index 3 for time.
        
        lat  = input_data[:, 0]
        lon  = input_data[:, 1]
        time = input_data[:, 3] 
        
        ones = torch.ones(N, device=self.device, dtype=torch.float64)
        
        # Stack: [Intercept, Lat, Lon, Time, Lat*Time, Lon*Time]
        locs = torch.stack([
            ones,
            lat,
            lon,
            time,
            lat * time,
            lon * time
        ], dim=1)  # Shape becomes (N, 6)
        # -------------------------------------------------------------

        if y.dim() == 1: y_col = y.unsqueeze(-1)
        else: y_col = y
        
        combined_rhs = torch.cat((locs, y_col), dim=1)
        Z_combined = torch.linalg.solve_triangular(L, combined_rhs, upper=False)
        
        # Slice appropriately for 6 features
        Z_X = Z_combined[:, :6] 
        Z_y = Z_combined[:, 6:]

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
    
    
    '''

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
        print("🚀 Pre-computing (Corrected Vectorization)...", end=" ")
        t0 = time.time()

        # -----------------------------------------------------------
        # 1. 데이터 통합 및 Dummy Point 생성
        # -----------------------------------------------------------
        key_list = list(self.input_map.keys())
        all_data_list = []
        
        # 날짜별 데이터 통합
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, np.ndarray):
                day_data = torch.from_numpy(day_data)
            all_data_list.append(day_data)
        
        # 실제 데이터 합치기
        # L40S 최적화를 위해 float32 저장 -> 계산시 float64 승격 추천
        # (원하시면 float64로 바꾸셔도 됩니다)
        Real_Data = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        
        # [핵심 수정 1] Cholesky 방지용 Dummy Point 생성 (아주 먼 곳)
        # 좌표를 (1e8, 1e8, ...)로 설정하여 모든 점과의 거리를 무한대로 만듦 -> 공분산 0
        dummy_point = torch.tensor([[1e8, 1e8, 0.0, 1e8]], device=self.device, dtype=torch.float32)
        
        # Full_Data의 맨 마지막 인덱스는 이제 "빈 칸(Padding)"을 의미함
        Full_Data = torch.cat([Real_Data, dummy_point], dim=0)
        dummy_idx = Full_Data.shape[0] - 1  # 패딩용 인덱스
        
        # -----------------------------------------------------------
        # 2. 인덱스 매핑 (CPU에서 로직 수행)
        # -----------------------------------------------------------
        day_lengths = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        
        # Heads / Tails 인덱스 분리
        heads_indices = []
        batch_indices_list = []
        
        # 최대 차원 계산 (원래 코드 로직: (M+1)*3)
        # 과거 2시점 + 현재 시점
        max_dim = (self.max_neighbors + 1) * 3 
        
        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset = cumulative_len[time_idx]
            
            # Heads
            n_heads = min(day_len, self.nheads)
            heads_indices.extend(range(offset, offset + n_heads))
            
            # Tails
            start_local = self.nheads
            if start_local >= day_len: continue
                
            # 이 날짜의 Tails 루프
            for local_idx in range(start_local, day_len):
                my_global_idx = offset + local_idx
                
                # 로컬 이웃 가져오기
                nbs_local = self.nns_map[local_idx] # numpy array 가정
                
                # --- 원래 코드의 시간차 이웃 로직 복구 ---
                current_indices = []
                
                # (1) 이웃들 + 나 자신 (현재 시점)
                # nbs_local은 array, local_idx는 scalar -> 합침
                targets = np.append(nbs_local, local_idx)
                current_indices.extend(offset + targets)
                
                # (2) 1일 전 (time_idx > 0)
                if time_idx > 0:
                    prev_offset = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]
                    # 범위 체크 (이전 날짜 데이터 길이를 넘으면 안됨)
                    valid_targets = targets[targets < prev_len]
                    current_indices.extend(prev_offset + valid_targets)
                    
                # (3) 2일 전 (time_idx > 1)
                if time_idx > 1:
                    prev2_offset = cumulative_len[time_idx - 2]
                    prev2_len = day_lengths[time_idx - 2]
                    valid_targets = targets[targets < prev2_len]
                    current_indices.extend(prev2_offset + valid_targets)
                
                # --- 패딩 로직 수정 ---
                # 원래 코드: slot = max_dim - combined.shape[0] (앞쪽을 비움)
                # 여기서도 앞쪽을 dummy_idx로 채움
                pad_len = max_dim - len(current_indices)
                if pad_len > 0:
                    # [dummy, dummy, ..., real_data] 순서
                    padded_row = [dummy_idx] * pad_len + current_indices
                else:
                    # 혹시 max_dim보다 길면 자름 (보통 그럴 일 없지만)
                    padded_row = current_indices[-max_dim:]
                
                batch_indices_list.append(padded_row)

        # -----------------------------------------------------------
        # 3. GPU Fancy Indexing & Batch 구성
        # -----------------------------------------------------------
        
        # (1) Heads
        heads_tensor = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = Full_Data[heads_tensor].contiguous().to(torch.float64) # Heads는 작으니까 64로
        
        # (2) Tails (Batch)
        if len(batch_indices_list) > 0:
            Indices_Tensor = torch.tensor(batch_indices_list, device=self.device, dtype=torch.long)
            
            # [한 방에 가져오기]
            Gathered_Data = Full_Data[Indices_Tensor] # (N_tails, max_dim, 4)
            
            # [최적화] X_batch: 좌표 정보
            self.X_batch = Gathered_Data[..., [0, 1, 3]].contiguous() # (N, max_dim, 3)
            
            # [최적화] Y_batch: 값 정보
            self.Y_batch = Gathered_Data[..., 2].unsqueeze(-1).contiguous() # (N, max_dim, 1)
            
            # [핵심 수정 2] Locs_batch 차원 및 값 수정
            # 원래 코드 L_cpu는 (N, max_dim, 6) 이었음.
            # 그리고 dummy(빈칸)인 곳은 0이어야 하고, 실제 데이터인 곳은 1(Intercept) 등 값이 있어야 함.
            
            # Feature Engineering
            g_lat  = Gathered_Data[..., 0]
            g_lon  = Gathered_Data[..., 1]
            g_time = Gathered_Data[..., 3]
            
            ones = torch.ones_like(g_lat)
            
            # (N, max_dim, 6) 생성
            self.Locs_batch = torch.stack([
                ones, g_lat, g_lon, g_time, g_lat*g_time, g_lon*g_time
            ], dim=-1).contiguous()
            
            # [Masking] Dummy Point(패딩된 곳)의 Feature를 0으로 만듦
            # Dummy point는 인덱스가 dummy_idx인 곳
            mask = (Indices_Tensor == dummy_idx).unsqueeze(-1) # (N, max_dim, 1)
            self.Locs_batch = self.Locs_batch.masked_fill(mask, 0.0)
            
        else:
            self.X_batch = torch.empty(0, device=self.device)
            self.Y_batch = torch.empty(0, device=self.device)
            self.Locs_batch = torch.empty(0, device=self.device)

        self.is_precomputed = True
        print(f"✅ Done in {time.time()-t0:.4f}s. (Heads: {len(heads_indices)}, Tails: {len(batch_indices_list)})")

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
        chunk_size = 4096    # 4096:15  better than 2000, 6000
        # maybe nvidia L40S:  18176/48gb   nvidia h100. 16896/80 
        # red hat linux 코어는 18176 그런데 메모리는 48 기가로 에이 100 (80기가) 보다 작지만 계산은 빠름
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
