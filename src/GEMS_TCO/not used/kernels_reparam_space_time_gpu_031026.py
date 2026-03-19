import sys
# gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src" # 필요시 경로 수정
# sys.path.append(gems_tco_path)
# import GEMS_TCO # 필요시 주석 해제

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
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from typing import Dict, Any, Callable, List, Tuple
import time
import copy    
import logging
import warnings
from functools import partial
from torch.special import gammainc, gammaln

# --- SpatioTemporalModel Class ---
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
        # mm_cond_number로 truncate하지 않음 → 저장된 이웃 전체를 유지
        # precompute_conditioning_sets()에서 cap(limit_A/B/C)으로 실제 사용 수를 제한
        nns_map = list(nns_map)
        for i in range(len(nns_map)):
            tmp = np.delete(nns_map[i], np.where(nns_map[i] == -1))
            nns_map[i] = tmp if tmp.size > 0 else np.array([], dtype=np.int64)
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
        one_tensor = torch.tensor(1.0, device=x.device, dtype=phi3.dtype)
        if phi3.ndim > 0:
            one_tensor = one_tensor.view_as(phi3)
            
        weights = torch.stack([phi3, one_tensor, phi4])
        weights = weights.view(-1) # Force to flat vector

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

# --- BATCHED GPU CLASS (Intercept + Centered Lat + 7 Dummies = 9 Features) ---
class VecchiaBatched(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int,
                 limit_A:int=8, limit_B:int=8, limit_C:int=8, daily_stride:int=8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)

        self.device = aggregated_data.device
        self.nheads = nheads
        self.max_neighbors = mm_cond_number
        self.is_precomputed = False

        # Conditioning set sizes (A: current t spatial, B: t-1, C: t-daily_stride)
        self.limit_A = limit_A
        self.limit_B = limit_B
        self.limit_C = limit_C
        self.daily_stride = daily_stride

        # Feature 개수: 9개
        # 1. Intercept (Base Mean)
        # 2. Latitude (Centered)
        # 3~9. Time Dummies (D1 ~ D7)
        self.n_features = 9

        self.X_batch = None
        self.Y_batch = None
        self.Locs_batch = None
        self.Heads_data = None

        # Centering용 평균 위도
        self.lat_mean_val = 0.0

    def batched_manual_dist(self, dist_params, x_batch):
        phi3, phi4, advec_lat, advec_lon = dist_params
        x_lat, x_lon, x_time = x_batch[:, :, 0], x_batch[:, :, 1], x_batch[:, :, 2]
        
        u_lat = x_lat - advec_lat * x_time
        u_lon = x_lon - advec_lon * x_time
        
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

    def precompute_conditioning_sets(self):
        limit_A, limit_B, limit_C = self.limit_A, self.limit_B, self.limit_C
        daily_stride  = self.daily_stride

        # 3개 그룹별 고정 행렬 크기 (dummy 없이 over-fetch로 채움)
        # Group A  : time_idx=0              → Set A only
        # Group AB : 0 < time_idx < stride   → Set A + B
        # Group ABC: time_idx >= stride       → Set A + B + C
        max_dim_A   = limit_A
        max_dim_AB  = limit_A + (limit_B + 1)
        max_dim_ABC = limit_A + (limit_B + 1) + (limit_C + 1)

        n_stored = next((len(m) for m in self.nns_map if len(m) > 0), 0)
        print(f"🚀 Pre-computing 3-group Vecchia "
              f"[A={max_dim_A}, AB={max_dim_AB}, ABC={max_dim_ABC}, stored={n_stored}]...", end=" ")

        # 1. 데이터 통합
        key_list      = list(self.input_map.keys())
        all_data_list = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                         for d in self.input_map.values()]
        Real_Data  = torch.cat(all_data_list, dim=0).to(self.device, dtype=torch.float32)
        n_real     = Real_Data.shape[0]
        num_cols   = Real_Data.shape[1]

        # NaN 마스크 (Python 루프용 numpy)
        is_nan_real   = torch.isnan(Real_Data[:, 2])
        valid_lats    = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = valid_lats.mean() if valid_lats.numel() > 0 else Real_Data[:, 0].mean()
        print(f"[Mean Lat: {self.lat_mean_val:.4f}]", end=" ")
        is_nan_mask_np = is_nan_real.cpu().numpy()

        # over-fetch로 거의 항상 cap까지 채우지만, 극단적 NaN 지역 안전망으로
        # 슬롯별 고유 dummy 생성 (서로 좌표가 달라 C[d_i,d_j]≈0 → near-singular 없음)
        n_dummies   = max_dim_ABC
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8   # 고유 lat
            dummy_block[k, 1] = (k + 1) * 1e8   # 고유 lon
            dummy_block[k, 3] = (k + 1) * 1e8   # 고유 time
        Full_Data      = torch.cat([Real_Data, dummy_block], dim=0)
        dummy_start    = n_real
        is_nan_mask_np = np.append(is_nan_mask_np, np.zeros(n_dummies, dtype=bool))

        # 2. 배치 인덱스 구성
        day_lengths    = [len(d) for d in all_data_list]
        cumulative_len = np.cumsum([0] + day_lengths)
        n_time_steps   = len(key_list)
        use_set_C      = daily_stride < n_time_steps

        heads_indices  = []
        batch_list_A   = []   # (N_A,   max_dim_A)
        batch_list_AB  = []   # (N_AB,  max_dim_AB)
        batch_list_ABC = []   # (N_ABC, max_dim_ABC)

        for time_idx, key in enumerate(key_list):
            day_len = day_lengths[time_idx]
            offset  = cumulative_len[time_idx]

            # Heads
            n_heads = min(day_len, self.nheads)
            for local_idx in range(n_heads):
                idx = offset + local_idx
                if not is_nan_mask_np[idx]:
                    heads_indices.append(idx)
            if self.nheads >= day_len:
                continue

            for local_idx in range(self.nheads, day_len):
                target_idx = offset + local_idx
                if is_nan_mask_np[target_idx]:
                    continue

                current_indices = []

                def add_valid_neighbors(indices_to_check, cap):
                    count = 0
                    for idx in indices_to_check:
                        if count >= cap:
                            break
                        if not is_nan_mask_np[idx]:
                            current_indices.append(idx)
                            count += 1

                # Set A: 현재 시점 t 공간 이웃 (over-fetch & NaN filter)
                add_valid_neighbors(
                    (offset + self.nns_map[local_idx]).tolist(), cap=limit_A)

                has_B = time_idx > 0
                has_C = time_idx >= daily_stride

                # Set B: t-1, independent filtering
                # x 자신 + nns_map 이웃을 t-1에서 독립적으로 NaN 필터링
                if has_B:
                    prev_off = cumulative_len[time_idx - 1]
                    prev_len = day_lengths[time_idx - 1]
                    if local_idx < prev_len:
                        add_valid_neighbors([prev_off + local_idx], cap=1)
                    nbs = self.nns_map[local_idx]
                    add_valid_neighbors(
                        (prev_off + nbs[nbs < prev_len]).tolist(), cap=limit_B)

                # Set C: t-daily_stride, independent filtering
                # x 자신 + nns_map 이웃을 t-2에서 독립적으로 NaN 필터링
                if has_C:
                    pd_idx = time_idx - daily_stride
                    pd_off = cumulative_len[pd_idx]
                    pd_len = day_lengths[pd_idx]
                    if local_idx < pd_len:
                        add_valid_neighbors([pd_off + local_idx], cap=1)
                    nbs = self.nns_map[local_idx]
                    add_valid_neighbors(
                        (pd_off + nbs[nbs < pd_len]).tolist(), cap=limit_C)

                # 그룹 결정
                if has_C:
                    max_d = max_dim_ABC; target_list = batch_list_ABC
                elif has_B:
                    max_d = max_dim_AB;  target_list = batch_list_AB
                else:
                    max_d = max_dim_A;   target_list = batch_list_A

                # 슬롯별 고유 dummy로 패딩 (over-fetch로 거의 불필요)
                n_valid = len(current_indices)
                if n_valid < max_d:
                    pad = [dummy_start + k for k in range(max_d - n_valid)]
                    row = pad + current_indices
                else:
                    row = current_indices[-max_d:]
                target_list.append(row)

        # 3. 그룹별 GPU 텐서 빌드
        heads_tensor   = torch.tensor(heads_indices, device=self.device, dtype=torch.long)
        self.Heads_data = (Full_Data[heads_tensor].contiguous().to(torch.float64)
                           if len(heads_indices) > 0
                           else torch.empty(0, num_cols, device=self.device, dtype=torch.float64))

        def build_tensors(idx_list, max_d):
            if not idx_list:
                return None, None, None
            T = torch.tensor(idx_list, device=self.device, dtype=torch.long)  # (N, max_d)
            G = Full_Data[T]                                                   # (N, max_d, num_cols)
            X    = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y    = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat  = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11]
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y    = Y.masked_fill(is_dummy, 0.0)
            return X, Y, Locs

        self.X_A,   self.Y_A,   self.Locs_A   = build_tensors(batch_list_A,   max_dim_A)
        self.X_AB,  self.Y_AB,  self.Locs_AB  = build_tensors(batch_list_AB,  max_dim_AB)
        self.X_ABC, self.Y_ABC, self.Locs_ABC = build_tensors(batch_list_ABC, max_dim_ABC)

        self.n_tails = len(batch_list_A) + len(batch_list_AB) + len(batch_list_ABC)
        self.is_precomputed = True
        print(f"[Set C: {use_set_C}] ✅ Done. "
              f"(Heads: {len(heads_indices)}, "
              f"Tails A/AB/ABC: {len(batch_list_A)}/{len(batch_list_AB)}/{len(batch_list_ABC)})")

    def vecchia_batched_likelihood(self, params):
        if not self.is_precomputed: raise RuntimeError("Run precompute first!")

        XT_Sinv_X_glob = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y_glob = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y_glob = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det_glob   = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        # -------------------------------------------------------
        # PART 1: HEADS
        # -------------------------------------------------------
        if self.Heads_data.shape[0] > 0:
            h_ones = torch.ones((self.Heads_data.shape[0], 1), device=self.device, dtype=torch.float64)
            h_lat = (self.Heads_data[:, 0] - self.lat_mean_val).unsqueeze(-1)
            h_dummies = self.Heads_data[:, 4:11]
            
            X_head = torch.cat([h_ones, h_lat, h_dummies], dim=1).to(torch.float64)
            y_head = self.Heads_data[:, 2].unsqueeze(-1).to(torch.float64)
            
            cov = self.matern_cov_aniso_STABLE_log_reparam(params, self.Heads_data, self.Heads_data)
            try:
                L = torch.linalg.cholesky(cov)
            except torch.linalg.LinAlgError: return torch.tensor(float('inf'), device=self.device)

            log_det_glob += 2 * torch.sum(torch.log(torch.diag(L)))
            Z_X = torch.linalg.solve_triangular(L, X_head, upper=False)
            Z_y = torch.linalg.solve_triangular(L, y_head, upper=False)
            
            XT_Sinv_X_glob += Z_X.T @ Z_X
            XT_Sinv_y_glob += Z_X.T @ Z_y
            yT_Sinv_y_glob += (Z_y.T @ Z_y).squeeze()

        # -------------------------------------------------------
        # PART 2: TAILS (3개 그룹: A / AB / ABC)
        # 각 그룹은 서로 다른 max_dim을 가지며 독립적으로 Cholesky 처리
        # -------------------------------------------------------
        chunk_size   = 4096
        batch_groups = [
            (self.X_A,   self.Y_A,   self.Locs_A),
            (self.X_AB,  self.Y_AB,  self.Locs_AB),
            (self.X_ABC, self.Y_ABC, self.Locs_ABC),
        ]
        for X_b, Y_b, Locs_b in batch_groups:
            if X_b is None or X_b.shape[0] == 0:
                continue
            for start in range(0, X_b.shape[0], chunk_size):
                end       = min(start + chunk_size, X_b.shape[0])
                cov_chunk = self.matern_cov_batched(params, X_b[start:end])
                try:
                    L_chunk = torch.linalg.cholesky(cov_chunk)
                except torch.linalg.LinAlgError:
                    return torch.tensor(float('inf'), device=self.device)

                Z_locs = torch.linalg.solve_triangular(L_chunk, Locs_b[start:end], upper=False)
                Z_y    = torch.linalg.solve_triangular(L_chunk, Y_b[start:end],    upper=False)

                u_X, u_y = Z_locs[:, -1, :], Z_y[:, -1, :]
                log_det_glob   += 2 * torch.sum(torch.log(L_chunk[:, -1, -1]))
                XT_Sinv_X_glob += u_X.T @ u_X
                XT_Sinv_y_glob += u_X.T @ u_y
                yT_Sinv_y_glob += (u_y.T @ u_y).squeeze()

        # -------------------------------------------------------
        # PART 3: SOLVE BETA
        # -------------------------------------------------------
        jitter = torch.eye(self.n_features, device=self.device, dtype=torch.float64) * 1e-6
        try:
            beta_global = torch.linalg.solve(XT_Sinv_X_glob + jitter, XT_Sinv_y_glob)
        except torch.linalg.LinAlgError:
            return torch.tensor(float('inf'), device=self.device)

        quad_form = (yT_Sinv_y_glob
                     - 2 * (beta_global.T @ XT_Sinv_y_glob)
                     + (beta_global.T @ XT_Sinv_X_glob @ beta_global))
        total_N = self.Heads_data.shape[0] + self.n_tails

        return 0.5 * (log_det_glob + quad_form.squeeze()) / total_N

# --- FITTING CLASS ---
class fit_vecchia_lbfgs(VecchiaBatched):

    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int,
                 limit_A:int=8, limit_B:int=8, limit_C:int=8, daily_stride:int=8):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads,
                         limit_A=limit_A, limit_B=limit_B, limit_C=limit_C, daily_stride=daily_stride)

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
