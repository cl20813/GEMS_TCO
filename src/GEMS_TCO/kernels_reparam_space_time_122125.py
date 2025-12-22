# Standard libraries
# import logging
# import math
# import time
# from collections import defaultdict
# Special functions and optimizations
# from scipy.spatial.distance import cdist  # For space and time distance
# from scipy.optimize import minimize  # For optimization
from sklearn.neighbors import KDTree
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

    

    def build_dilated_nns_map(coords, k_near=8, k_far=4, k_farther=4):
        """
        구성: [Near(8) | Far(4) | Farther(4)] 총 16개 저장
        - Far: 약 3~4칸 거리 (인덱스 15~30 구간에서 추출)
        - Farther: 약 5~8칸 거리 (인덱스 40~60 구간에서 추출)
        """
        tree = KDTree(coords)
        
        # 넉넉하게 100개 검색
        _, indices = tree.query(coords, k=100)
        
        final_map = {}
        for i in range(len(coords)):
            # 1. Near: 가장 가까운 8개 (자기 자신 제외, indices[i, 0]은 본인)
            near_idx = indices[i, 1:1+k_near] # 1~8
            
            # 2. Far: 조금 떨어진 곳 (예: 15번째~30번째 중 듬성듬성 4개)
            # 방향성을 모르므로 4방위를 커버하기 위해 간격을 둠
            far_idx = indices[i, [15, 20, 25, 30]] 
            
            # 3. Farther: 더 떨어진 곳 (예: 45번째~60번째 중 4개)
            farther_idx = indices[i, [45, 50, 55, 60]]
            
            # 하나로 합침 (순서 중요: Near -> Far -> Farther)
            combined = np.concatenate([near_idx, far_idx, farther_idx])
            final_map[i] = combined.astype(int)
            
        return final_map

# [사용법] 모델에 넣기 전에 이걸로 맵을 업데이트
# updated_nns_map = build_dilated_nns_map(coords_only, k_near=8, k_far=4, k_farther=4)

    def precompute_conditioning_sets(self, temporal_period=None):
        self.temporal_period = temporal_period
        print(f"Pre-computing with Hybrid Strategy (Reuse + Upwind Bracketing ±0.022)...", end=" ")
        
        key_list = list(self.input_map.keys())
        cut_line = self.nheads
        
        # 1. Heads Data (기존 동일)
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, np.ndarray):
                head_chunk = torch.from_numpy(day_data[:cut_line]).to(self.device)
            else:
                head_chunk = day_data[:cut_line].clone().detach().to(self.device)
            heads_list.append(head_chunk)
        self.Heads_data = torch.cat(heads_list, dim=0).to(torch.float64)

        # 2. Tasks (기존 동일)
        tasks = []
        for time_idx, key in enumerate(key_list):
            day_data = self.input_map[key]
            indices = range(cut_line, len(day_data))
            for idx in indices:
                tasks.append((time_idx, idx))

        # -------------------------------------------------------------------------
        # [중요] Upwind Bracketing Strategy (위/아래 양방향 오프셋)
        # Latitude 변동성이 크므로, 소스 위치를 위(+0.022)/아래(-0.022)로 
        # 벌려서(Bracket) 보간 효과를 노립니다.
        # -------------------------------------------------------------------------
        ref_map = self.input_map[key_list[0]]
        if isinstance(ref_map, torch.Tensor):
            coords_np = ref_map[:, :2].detach().cpu().numpy()
        else:
            coords_np = ref_map[:, :2]

        from scipy.spatial import cKDTree
        tree = cKDTree(coords_np)
        
        # 기본 동쪽 이동 거리 (Longitude)
        base_east = 0.20
        
        # (A) Look North-East (동쪽 + 북쪽 0.044)
        upwind_NE_offsets = np.zeros_like(coords_np)
        upwind_NE_offsets[:, 0] = 0.04   # North (+Lat)
        upwind_NE_offsets[:, 1] = base_east
        
        # (B) Look South-East (동쪽 - 남쪽 0.044)
        upwind_SE_offsets = np.zeros_like(coords_np)
        upwind_SE_offsets[:, 0] = -0.04  # South (-Lat)
        upwind_SE_offsets[:, 1] = base_east
        
        target_NE = coords_np + upwind_NE_offsets
        target_SE = coords_np + upwind_SE_offsets
        
        # 두 번의 쿼리로 각각의 인덱스 확보
        _, indices_NE = tree.query(target_NE, k=1)
        _, indices_SE = tree.query(target_SE, k=1)
        
        # -------------------------------------------------------------------------

        total_vecchia_points = len(tasks)
        
        # 메모리 할당 계산 (넉넉하게 수정):
        # T:   Self(1) + Neighbor(8) = 9
        # T-1: Self(1) + Neighbor(8) + Upwind_NE(1) + Upwind_SE(1) = 11
        # T-2: Self(1) + Neighbor(8) + Upwind_NE(1) + Upwind_SE(1) = 11
        # Target: 1
        # Total per row ≈ 32 ~ 33
        # Safety Margin을 위해 40으로 설정
        max_storage_dim = 40
        
        self.X_batch = torch.zeros((total_vecchia_points, max_storage_dim, 3), device=self.device, dtype=torch.float64)
        
        # 패딩 초기화 (Singular 방지)
        dummy = torch.arange(max_storage_dim, device=self.device, dtype=torch.float64) * 10000.0 + 1e6
        self.X_batch[:, :, 0] = dummy.unsqueeze(0)
        self.X_batch[:, :, 1] = dummy.unsqueeze(0)
        
        self.Y_batch = torch.zeros((total_vecchia_points, max_storage_dim, 1), device=self.device, dtype=torch.float64)
        n_trend = 5 if temporal_period else 3
        self.Locs_batch = torch.zeros((total_vecchia_points, max_storage_dim, n_trend), device=self.device, dtype=torch.float64)

        for i, (time_idx, index) in enumerate(tasks):
            current_np = self.input_map[key_list[time_idx]]
            current_row = current_np[index].reshape(1, -1)
            
            # 1. Spatial Neighbors (Reuse Strategy)
            spatial_neighbors = self.nns_map[index]
            
            data_list = []
            
            # --- (A) Time T (Current) ---
            if len(spatial_neighbors) > 0:
                data_list.append(current_np[spatial_neighbors])

            # --- (B) Time T-1 (Lag 1) ---
            if time_idx > 0:
                prev_np = self.input_map[key_list[time_idx - 1]]
                
                # [Strategy] Self + Neighbors + Upwind(NE) + Upwind(SE)
                idx_ne = indices_NE[index]
                idx_se = indices_SE[index]
                
                idx_t_1 = np.concatenate([
                    [index],            # Self
                    spatial_neighbors,  # Reuse Neighbors
                    [idx_ne, idx_se]    # Upwind Bracket (2 points)
                ]).astype(int)
                
                # 중복 제거
                idx_t_1 = np.unique(idx_t_1)
                
                data_list.append(prev_np[idx_t_1])

            # --- (C) Time T-2 (Lag 2) ---
            if time_idx > 1:
                prev_prev_np = self.input_map[key_list[time_idx - 2]]
                
                idx_ne = indices_NE[index]
                idx_se = indices_SE[index]
                
                idx_t_2 = np.concatenate([
                    [index],
                    spatial_neighbors,
                    [idx_ne, idx_se]    # Upwind Bracket (2 points)
                ]).astype(int)
                
                idx_t_2 = np.unique(idx_t_2)
                
                data_list.append(prev_prev_np[idx_t_2])

            # --- Data Merge (기존과 동일) ---
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
            
            self.Locs_batch[i, start_slot:, 0] = 1.0 
            self.Locs_batch[i, start_slot:, 1] = combined_data[:, 0] 
            self.Locs_batch[i, start_slot:, 2] = combined_data[:, 1]
            
            if temporal_period:
                t_vals = combined_data[:, 3]
                omega = 2 * np.pi / temporal_period
                self.Locs_batch[i, start_slot:, 3] = torch.sin(omega * t_vals)
                self.Locs_batch[i, start_slot:, 4] = torch.cos(omega * t_vals)

        self.is_precomputed = True
        print(f"Done. Batch size: {self.X_batch.shape[0]}")
    
    # Final Interpretable Params: {'sigma_sq': 7.510395564042606, 'range_lon': 0.09945995588743121, 'range_lat': 0.09998786309540021, 'range_time': 0.5801881001939552, 'advec_lat': 0.00711489265840434, 'advec_lon': -0.44135718085641384, 'nugget': 0.3863708436821134}
    # for below

    '''
    import numpy as np
    import torch
    from sklearn.neighbors import KDTree 
    from GEMS_TCO import orderings as _orderings
    from typing import Tuple

    def get_spatial_ordering_dilated(
            input_maps: dict,
            mm_cond_number: int = 16 
        ) -> Tuple[np.ndarray, list]:  # <--- 반환 타입이 list로 변경됨
            
            key_list = list(input_maps.keys())
            data_for_coord = input_maps[key_list[0]]
            
            if isinstance(data_for_coord, torch.Tensor):
                data_for_coord = data_for_coord.cpu().numpy()

            x1 = data_for_coord[:, 0]
            y1 = data_for_coord[:, 1]
            coords1 = np.stack((x1, y1), axis=-1)

            # MaxMin Ordering
            ord_mm = _orderings.maxmin_cpp(coords1)
            
            data_for_coord_reordered = data_for_coord[ord_mm]
            coords1_reordered = np.stack(
                (data_for_coord_reordered[:, 0], data_for_coord_reordered[:, 1]), 
                axis=-1
            )
            
            print("Building KDTree for Dilated Neighbors...")
            tree = KDTree(coords1_reordered)
            
            _, indices = tree.query(coords1_reordered, k=65)
            
            # --- [FIX] Dict가 아니라 List로 생성 ---
            nns_map_list = [] 
            
            for i in range(len(coords1_reordered)):
                # (1) Near: 가장 가까운 8개
                near_idx = indices[i, 1:9]
                
                # (2) Far: 격자 3~4칸 거리 (15, 20, 25, 30번째)
                far_idx = indices[i, [15, 20, 25, 30]]
                
                # (3) Farther: 격자 5~8칸 거리 (45, 50, 55, 60번째)
                farther_idx = indices[i, [45, 50, 55, 60]]
                
                # 합치기
                combined = np.concatenate([near_idx, far_idx, farther_idx])
                
                # 과거(자신보다 순서가 앞선) 이웃만 필터링
                valid_neighbors = combined[combined < i]
                
                # --- [FIX] 리스트에 Append (순서대로 쌓임) ---
                nns_map_list.append(valid_neighbors.astype(int))
                
            return ord_mm, nns_map_list  # 리스트 반환

    ord_mm, nns_map = get_spatial_ordering_dilated(input_map, mm_cond_number=16)


    mm_input_map = {}
    for key in input_map:
        mm_input_map[key] = input_map[key][ord_mm]  # Extract only Lat and Lon columns
        

    def precompute_conditioning_sets(self, temporal_period=None):
        self.temporal_period = temporal_period
        print(f"Pre-computing Batched Tensors (Dilated Strategy, Period: {temporal_period})...", end=" ")
        
        key_list = list(self.input_map.keys())
        cut_line = self.nheads
        
        # 1. Heads Data 처리 (기존 동일)
        heads_list = []
        for key in key_list:
            day_data = self.input_map[key]
            if isinstance(day_data, np.ndarray):
                head_chunk = torch.from_numpy(day_data[:cut_line]).to(self.device)
            else:
                head_chunk = day_data[:cut_line].clone().detach().to(self.device)
            heads_list.append(head_chunk)
        self.Heads_data = torch.cat(heads_list, dim=0).to(torch.float64)

        # 2. Tasks 생성 (기존 동일)
        tasks = []
        for time_idx, key in enumerate(key_list):
            day_data = self.input_map[key]
            indices = range(cut_line, len(day_data))
            for idx in indices:
                tasks.append((time_idx, idx))

        total_vecchia_points = len(tasks)
        
        # --- 💥 FIX: 메모리 할당 계산 ---
        # 전략에 따른 포인트 수 계산:
        # T   (Current): Target(1) + Near(8) = 9
        # T-1 (Lag 1)  : Self(1)   + Near(5) + Far(4) = 10
        # T-2 (Lag 2)  : Self(1)   + Near(4) + Far(4) + Farther(4) = 13
        # Total per row = 9 + 10 + 13 = 32
        max_storage_dim = 32 + 5  # 여유분 5개 추가
        
        # 3. Allocating Tensors (패딩: 서로 다른 먼 좌표로 초기화)
        self.X_batch = torch.zeros((total_vecchia_points, max_storage_dim, 3), device=self.device, dtype=torch.float64)
        
        # 패딩 좌표 분산 (Singular Matrix 방지)
        dummy_coords = torch.arange(max_storage_dim, device=self.device, dtype=torch.float64) * 10000.0 + 1e6
        self.X_batch[:, :, 0] = dummy_coords.unsqueeze(0)
        self.X_batch[:, :, 1] = dummy_coords.unsqueeze(0)
        
        self.Y_batch = torch.zeros((total_vecchia_points, max_storage_dim, 1), device=self.device, dtype=torch.float64)
        
        n_trend_cols = 5 if temporal_period else 3
        self.Locs_batch = torch.zeros((total_vecchia_points, max_storage_dim, n_trend_cols), device=self.device, dtype=torch.float64)

        # 4. Fill Tensors (Dilated Logic 적용)
        for i, (time_idx, index) in enumerate(tasks):
            current_np = self.input_map[key_list[time_idx]]
            current_row = current_np[index].reshape(1, -1) 
            
            # nns_map에서 전체 리스트 가져오기 (16개 가정: 0~7 Near, 8~11 Far, 12~15 Farther)
            all_neighbors = self.nns_map[index] 
            
            data_list = []
            
            # --- (1) Time t (Current) ---
            # Strategy: 이웃 8개 (Near 8)
            # all_neighbors[:8] 사용
            idx_t = all_neighbors[:8]
            if len(idx_t) > 0:
                data_list.append(current_np[idx_t])
            
            # --- (2) Time t-1 (Lag 1) ---
            if time_idx > 0:
                prev_np = self.input_map[key_list[time_idx - 1]]
                # Strategy: 본인 + 이웃 5개(Near) + 먼 4개(Far)
                # 본인: [index]
                # Near 5: all_neighbors[:5]
                # Far 4: all_neighbors[8:12] (인덱스 주의: build_dilated_nns_map 기준)
                
                idx_t_1 = np.concatenate([
                    [index],              # Self
                    all_neighbors[:5],    # Near 5
                    all_neighbors[8:12]   # Far 4
                ]).astype(int)
                
                data_list.append(prev_np[idx_t_1])

            # --- (3) Time t-2 (Lag 2) ---
            if time_idx > 1:
                prev_prev_np = self.input_map[key_list[time_idx - 2]]
                # Strategy: 본인 + 이웃 4개(Near) + 먼 4개(Far) + 더 먼 4개(Farther)
                # Far 4: all_neighbors[8:12]
                # Farther 4: all_neighbors[12:16]
                
                idx_t_2 = np.concatenate([
                    [index],               # Self
                    all_neighbors[:4],     # Near 4
                    all_neighbors[8:12],   # Far 4 (Reuse from t-1 logic)
                    all_neighbors[12:16]   # Farther 4
                ]).astype(int)
                
                data_list.append(prev_prev_np[idx_t_2])

            # --- Merge & Fill ---
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
            
            # Data Fill
            self.X_batch[i, start_slot:, 0] = combined_data[:, 0]
            self.X_batch[i, start_slot:, 1] = combined_data[:, 1]
            self.X_batch[i, start_slot:, 2] = combined_data[:, 3]
            self.Y_batch[i, start_slot:, 0] = combined_data[:, 2]
            
            # Trend Fill
            self.Locs_batch[i, start_slot:, 0] = 1.0 
            self.Locs_batch[i, start_slot:, 1] = combined_data[:, 0] 
            self.Locs_batch[i, start_slot:, 2] = combined_data[:, 1] 
            
            if temporal_period:
                t_vals = combined_data[:, 3]
                omega = 2 * np.pi / temporal_period
                self.Locs_batch[i, start_slot:, 3] = torch.sin(omega * t_vals)
                self.Locs_batch[i, start_slot:, 4] = torch.cos(omega * t_vals)

        self.is_precomputed = True
        print(f"Done. Heads: {self.Heads_data.shape[0]}, Batched Tails: {self.X_batch.shape[0]}")
    '''


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
        # Adapter to map params correctly for the single-matrix function
        def adapter_cov_func(params, x, y):
            return self.matern_cov_aniso_STABLE_log_reparam(params, x, y)

        if self.Heads_data.shape[0] > 0:
            head_nll_avg = self.full_likelihood_avg(params, self.Heads_data, self.Heads_data[:, 2], adapter_cov_func)
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
        
        jitter = torch.eye(3, device=self.device, dtype=torch.float64).unsqueeze(0) * 1e-8
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
