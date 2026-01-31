# Standard libraries
import logging
import math
import sys
from collections import defaultdict
import time

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Nearest neighbor search
import sklearn
from sklearn.neighbors import BallTree

# Special functions and optimizations
from scipy.spatial.distance import cdist  # For space and time distance
from scipy.special import gamma, kv  # Bessel function and gamma function
from scipy.optimize import minimize
from scipy.optimize import basinhopping, minimize
from scipy.stats import norm,uniform

import torch
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import FuncFormatter
# Type hints
from typing import Callable, Union, Tuple
from pathlib import Path



# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

import sys
# Add your custom path
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)


# Custom imports
# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/evaluate.log'

from GEMS_TCO.kernels_reparam_space_time_gpu_before_010126 import SpatioTemporalModel  

import numpy as np
import torch

class diagnosis:
    ''' 
    Investigate the value (Data - conditional mean from Vecchia) / Conditional sd from Vecchia
    See if these are standard normal.
    Using composition with SpatioTemporalModel instance.
    '''
    def __init__(self, model_instance):
        """
        model_instance: SpatioTemporalModel 또는 VecchiaBatched의 인스턴스
        (이미 데이터와 nns_map이 로드된 상태여야 함)
        """
        self.model = model_instance
        self.input_map = model_instance.input_map
        self.nns_map = model_instance.nns_map
        self.key_list = model_instance.key_list
        self.size_per_hour = model_instance.size_per_hour
        self.device = model_instance.device

    def diagnosis_method1(self, params):
        # 결과 저장소 (Timestamps x Data Points)
        res = np.zeros((len(self.key_list), self.size_per_hour))
        
        # 파라미터가 텐서가 아니면 변환 (모델의 Covariance 함수가 텐서를 요구함)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, device=self.device, dtype=torch.float64)

        for time_idx in range(len(self.key_list)):
            # 현재 시간대의 데이터 가져오기
            current_np = self.input_map[self.key_list[time_idx]]
            
            # 31번째 데이터부터 시작 (Vecchia neighbor가 충분히 확보된 시점)
            for index in range(31, self.size_per_hour):
                
                # 1. 타겟 포인트 (Current Data)
                current_row = current_np[index].reshape(1,-1)
                current_y = current_row[0][2] # Target Value
                
                # 2. 이웃 포인트 가져오기 (Conditioning Data)
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                # 같은 시간대의 공간적 이웃
                if past: 
                    data_list.append(current_np[past])
                
                # 이전 시간대의 이웃 (시간적 상관관계)
                if time_idx > 0:
                    last_hour_np = self.input_map[self.key_list[time_idx-1]]
                    # 현재 포인트(index)와 이웃(past)의 과거 위치 데이터를 모두 가져옴
                    data_list.append(last_hour_np[(past+[index]), :])
                
                if data_list: 
                    conditioning_data = np.vstack(data_list)
                else: 
                    conditioning_data = np.empty((0, 4)) # 데이터 구조에 맞게 shape 조정

                # 3. 전체 데이터셋 구성 [Target; Neighbors]
                # Kriging을 위해 (0번째 행: 타겟, 1~N번째 행: 이웃)으로 쌓음
                np_arr = np.vstack((current_row, conditioning_data))
                
                # 4. 공분산 행렬 계산 (GPU Model 사용)
                # Numpy -> Tensor
                full_data_t = torch.tensor(np_arr, device=self.device, dtype=torch.float64)
                
                # 모델의 공분산 함수 호출 (params, x, y)
                cov_matrix = self.model.matern_cov_aniso_STABLE_log_reparam(params, full_data_t, full_data_t)
                
                # Tensor -> Numpy (Linear Algebra Solver는 CPU/Numpy가 안정적일 때가 많음)
                cov_matrix_np = cov_matrix.detach().cpu().numpy()
                
                # 5. Kriging (Simple Kriging / GLS) 수행
                # 데이터 분리
                y_and_neighbors = np_arr[:, 2] # 값 컬럼
                locs = np_arr[:, :2]           # 좌표 컬럼 (Trend 계산용)
                
                # 공분산 행렬 분할
                cov_xx = cov_matrix_np[1:, 1:] # 이웃 간 공분산 (Conditioning set)
                cov_yx = cov_matrix_np[0, 1:]  # 타겟-이웃 간 공분산
                sigma = cov_matrix_np[0][0]    # 타겟 자체 분산

                try:
                    # (1) Trend (Beta) 추정 - GLS
                    # [X' C^-1 X]^-1 [X' C^-1 y]
                    inv_cov_locs = np.linalg.solve(cov_matrix_np, locs)
                    inv_cov_y = np.linalg.solve(cov_matrix_np, y_and_neighbors)
                    
                    tmp1 = np.dot(locs.T, inv_cov_locs)
                    tmp2 = np.dot(locs.T, inv_cov_y)
                    beta = np.linalg.solve(tmp1, tmp2)
                    
                    # (2) Mean 추정
                    mu = np.dot(locs, beta)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]

                    # (3) Conditional Mean & Variance (Kriging Equations)
                    # Mean = mu_0 + cov_yx * cov_xx^-1 * (y_neighbors - mu_neighbors)
                    # Var  = sigma - cov_yx * cov_xx^-1 * cov_yx'
                    
                    # solve(A, b) is faster and more stable than dot(inv(A), b)
                    term_solved = np.linalg.solve(cov_xx, cov_yx) # cov_xx^-1 * cov_yx
                    
                    cov_ygivenx = sigma - np.dot(cov_yx.T, term_solved)
                    sd_ygivenx = np.sqrt(cov_ygivenx)
                    
                    mean_ygivenx = mu_current + np.dot(cov_yx.T, np.linalg.solve(cov_xx, (y_and_neighbors[1:] - mu_neighbors)))

                    # (4) Z-score (Diagnosis Metric)
                    if sd_ygivenx > 0:
                        res[time_idx, index] = (current_y - mean_ygivenx) / sd_ygivenx
                    else:
                        res[time_idx, index] = np.nan
                        
                except np.linalg.LinAlgError:
                    # 역행렬 계산 실패 시 NaN 처리
                    res[time_idx, index] = np.nan
                    
        return res
    
class CrossVariogram:
    def __init__(self, save_path, length_of_analysis):
        self.save_path = save_path
        self.length_of_analysis = length_of_analysis

    def compute_directional_semivariogram(self, deltas, map_data, days, tolerance):
        """
        Optimized to pre-compute spatial indices once.
        Fixed to handle partial data loading safely using relative indexing.
        """
        lon_lag_sem = {0: deltas}  # Save deltas for reference
        num_lags = 8
        key_list = sorted(map_data.keys())

        # --- STEP 1: Pre-compute Pair Indices (Run Once) ---
        print("Pre-computing spatial pairs for defined lags...")
        
        if not key_list:
            print("Error: No data in map_data.")
            return {}

        first_data = map_data[key_list[0]]
        coordinates = first_data[:, :2]  # Shape (N, 2)

        # Use GPU if available for faster indexing
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        coords_t = torch.tensor(coordinates, device=device, dtype=torch.float32)
        
        # Store valid indices for each delta: list of (row_idx, col_idx) tensors
        delta_indices_cache = []

        # Calculate difference matrices ONCE
        # Note: If N is very large (>30k), do this in blocks to avoid OOM.
        lat_diffs = coords_t[:, None, 0] - coords_t[None, :, 0]
        lon_diffs = coords_t[:, None, 1] - coords_t[None, :, 1]

        for (delta_lat, delta_lon) in deltas:
            # Create boolean mask for pairs matching this specific lag
            mask = (torch.abs(lat_diffs - delta_lat) <= tolerance) & \
                   (torch.abs(lon_diffs - delta_lon) <= tolerance)
            
            # Extract indices of valid pairs
            pairs = torch.nonzero(mask, as_tuple=True)
            delta_indices_cache.append(pairs)
        
        # Free memory of large matrices immediately
        del lat_diffs, lon_diffs, mask, coords_t
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        print("Pre-computation complete. Starting daily analysis...")

        # --- STEP 2: Fast Analysis Loop ---
        # [FIX]: Use enumerate to get relative index (for loaded data) and day_num (for keys)
        for relative_idx, day_num in enumerate(days):
            # Initialize storage for this day
            lon_lag_sem[day_num + 1] = [[np.nan] * len(deltas) for _ in range(num_lags)]
            
            for i in range(num_lags):
                # Calculate global index based on RELATIVE position in the loaded list
                list_idx = 8 * relative_idx + i
                
                if list_idx >= len(key_list):
                    break

                # Load data for this specific hour
                cur_data = map_data[key_list[list_idx]]
                
                # Prepare values (centered)
                vals = torch.tensor(cur_data[:, 2], device=device, dtype=torch.float32)
                vals = vals - torch.mean(vals)

                # Iterate through pre-computed lags
                for j, (idx_row, idx_col) in enumerate(delta_indices_cache):
                    if len(idx_row) == 0:
                        continue
                    
                    # Vectorized calculation
                    diffs = vals[idx_col] - vals[idx_row]
                    semivariance = 0.5 * torch.mean(diffs ** 2)
                    
                    lon_lag_sem[day_num + 1][i][j] = round(semivariance.item(), 4)
                    
        return lon_lag_sem

    def compute_cross_lon_lat(self, deltas, map_data, days, tolerance):
        # 1. Initialize logic matches original
        lon_lag_sem = {0: deltas}
        num_lags = 7  # Matches your variable
        key_list = sorted(map_data.keys())
        
        # --- PRE-COMPUTATION (Optimization) ---
        if not key_list: return {}
        first_data = map_data[key_list[0]]
        coordinates = first_data[:, :2]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        coords_t = torch.tensor(coordinates, device=device, dtype=torch.float32)
        
        lat_diffs = coords_t[:, None, 0] - coords_t[None, :, 0]
        lon_diffs = coords_t[:, None, 1] - coords_t[None, :, 1]
        
        delta_indices_cache = []
        for (delta_lat, delta_lon) in deltas:
            mask = (torch.abs(lat_diffs - delta_lat) <= tolerance) & \
                   (torch.abs(lon_diffs - delta_lon) <= tolerance)
            pairs = torch.nonzero(mask, as_tuple=True)
            delta_indices_cache.append(pairs)
            
        del lat_diffs, lon_diffs, mask, coords_t
        if device == 'cuda': torch.cuda.empty_cache()
        # --------------------------------------

        # [FIX]: Use enumerate for relative indexing
        for relative_idx, day_num in enumerate(days):
            # Matches your initialization
            lon_lag_sem[day_num + 1] = [[np.nan] * len(deltas) for _ in range(num_lags)]
            
            for i in range(num_lags): 
                # Use relative index to find position in key_list
                t_idx = 8 * relative_idx + i
                
                # Check bounds for t_idx AND t_idx+1
                if t_idx + 1 >= len(key_list): break
                
                cur_data = map_data[key_list[t_idx]]
                next_data = map_data[key_list[t_idx + 1]]
                
                cur_vals = torch.tensor(cur_data[:, 2], device=device, dtype=torch.float32)
                cur_vals = cur_vals - torch.mean(cur_vals)
                
                next_vals = torch.tensor(next_data[:, 2], device=device, dtype=torch.float32)
                next_vals = next_vals - torch.mean(next_vals)

                for j, (idx_row, idx_col) in enumerate(delta_indices_cache):
                    if len(idx_row) == 0:
                        continue
                    
                    # Cross Variogram Logic: Z(u+h, t) - Z(u, t+1)
                    diffs = cur_vals[idx_col] - next_vals[idx_row]
                    
                    semivariance = 0.5 * torch.mean(diffs ** 2)
                    
                    lon_lag_sem[day_num + 1][i][j] = round(semivariance.item(), 4)
                    
        return lon_lag_sem

    def cross_directional_sem(self, deltas, map_data, days, tolerance, direction1, direction2):
        directional_sem = {}
        directional_sem[0] = deltas
        
        # --- PRE-COMPUTATION ---
        key_list = sorted(map_data.keys())
        if not key_list: return {}
        
        first_data = map_data[key_list[0]]
        coordinates = first_data[:, :2]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        coords_t = torch.tensor(coordinates, device=device, dtype=torch.float32)

        lat_diffs = coords_t[:, None, 0] - coords_t[None, :, 0]
        lon_diffs = coords_t[:, None, 1] - coords_t[None, :, 1]
        
        # Pre-calc geometry
        dist_matrix = torch.sqrt(lat_diffs**2 + lon_diffs**2)
        angle_matrix = torch.atan2(lat_diffs, lon_diffs)

        delta_indices_cache = []
        mid_point = len(deltas) / 2
        
        for j, distance in enumerate(deltas):
            # Direction Logic
            target_dir = direction1 if j <= mid_point else direction2
            
            # Mask Logic
            dir_mask = torch.abs(angle_matrix - target_dir) <= (np.pi / 8)
            dist_mask = torch.abs(dist_matrix - distance) <= tolerance
            
            if distance == 0:
                final_mask = dist_mask
            else:
                final_mask = dir_mask & dist_mask 

            pairs = torch.nonzero(final_mask, as_tuple=True)
            delta_indices_cache.append(pairs)

        del lat_diffs, lon_diffs, dist_matrix, angle_matrix, coords_t
        if device == 'cuda': torch.cuda.empty_cache()
        # -----------------------

        # [FIX]: Use enumerate for relative indexing
        for relative_idx, day_num in enumerate(days):
            directional_sem[day_num] = [[np.nan]*len(deltas) for _ in range(7)]
            
            # Note: Original logic had t = day - 1. 
            # If 'days' list implies we are processing those specific days, 
            # we should iterate through the loaded data sequentially.
            
            for i in range(7): 
                list_idx = 8 * relative_idx + i
                
                if list_idx + 1 >= len(key_list): break

                cur_data = map_data[key_list[list_idx]]
                next_data = map_data[key_list[list_idx + 1]]

                cur_vals = torch.tensor(cur_data[:, 2], device=device, dtype=torch.float32)
                cur_vals = cur_vals - torch.mean(cur_vals)
                
                next_vals = torch.tensor(next_data[:, 2], device=device, dtype=torch.float32)
                next_vals = next_vals - torch.mean(next_vals)

                for j, (idx_row, idx_col) in enumerate(delta_indices_cache):
                    if len(idx_row) == 0:
                        continue

                    diffs = cur_vals[idx_row] - next_vals[idx_col]
                    semivariance = 0.5 * torch.mean(diffs ** 2)
                    
                    directional_sem[day_num][i][j] = semivariance.item()

        return directional_sem

    def theoretical_gamma_kv(self, params: torch.Tensor, lat_diff: torch.Tensor, lon_diff: torch.Tensor, time_diff: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Computes the Semivariogram for anisotropic space-time data.
        """
        # [수정 1] 파라미터 Unpacking 순서 및 이름 일치 (Estimates DF와 동일하게)
        # params: [sigmasq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget]
        sigmasq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget = params
        
        # [수정 2] Advection 적용
        tmp1 = lat_diff - advec_lat * time_diff
        tmp2 = lon_diff - advec_lon * time_diff
        
        # [수정 3] 시간 거리 계산 (beta 곱하기 -> range_time 나누기)
        # time scale은 보통 range_time으로 나누는 것이 일반적입니다.
        tmp3 = time_diff / range_time 
        
        # 4. 전체 시공간 거리 (Ellipsoidal Distance) 제곱
        # d는 (distance / range)^2 형태가 됩니다.
        d = tmp1**2 / range_lat**2 + tmp2**2 / range_lon**2 + tmp3**2
        d = d.clone().detach()

        # [안전 장치] kv 함수는 보통 scipy(CPU/Numpy) 기반이므로 Tensor 변환 필요
        # d가 GPU에 있다면 CPU로 내리고 Numpy로 변환해야 안전합니다.
        if d.is_cuda:
            d_np = d.cpu().numpy()
        else:
            d_np = d.numpy()
            
        # Matérn Correlation 계산 (Distance > 0 가정)
        # np.sqrt(d)는 scaled distance입니다.
        # kv 함수의 인자가 Bessel K_v(h) 형태인지 확인 필요 (보통 GEMS_TCO는 이 형태 사용)
        import scipy.special # 안전을 위해 명시
        tmp_np = scipy.special.kv(self.smooth, np.sqrt(d_np))
        
        # 다시 Tensor로 변환
        tmp = torch.from_numpy(tmp_np).to(d.device)

        # Spatial Covariance 부분 (Nugget 제외)
        out_tmp = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                (torch.sqrt(d))**self.smooth * tmp)
        
        # [결론] Variogram = Total Sill - Spatial Covariance
        # (nugget + sigmasq) - (sigmasq * rho) = nugget + sigmasq(1 - rho)
        # 이 식은 h > 0 일 때 정확합니다.
        out = (nugget + sigmasq) - out_tmp
        
        # 거리 0인 지점(NaN 등) 처리 (선택사항)
        out = torch.nan_to_num(out, nan=0.0) 

        return torch.sqrt(d), out
        
    def squared_formatter(self, x: float, pos: int) -> str:
        """
        Formats the input value by taking its square root and returning it as a string.

        Parameters:
        - x (float): Input value.
        - pos (int): Position (unused in the function).

        Returns:
        - str: Formatted string of the square root of x.
        """
        return f'{np.sqrt(x):.2f}'


class CrossVariogram_empirical(CrossVariogram):
    def __init__(self, save_path, length_of_analysis, smooth):
        super().__init__(save_path, length_of_analysis)
        self.smooth = smooth
        
    def plot_cross_lon_empirical(self, lon_lag_sem, days, deltas):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        for index, day in enumerate(days):
            ax = axs[index // 2, index % 2]

            lon_lags = [lon for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(lon_lags, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')
                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('symlog', linthresh=0.95)
            # ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(lon_lags)
            ticks = [str(round(x, 2)) for x in lon_lags]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'
            ax.legend()

        plt.tight_layout()
        # plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_longitude_days{days[0]}_{days[-1]}.png')
        save_path = Path(self.save_path) / f'cross_sem_longitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        plt.show()


    def plot_cross_lat_empirical(self, lat_lag_sem, days, deltas):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            # Separate positive and negative lags and assign appropriate indices
        
            lat_lags = [lat for lat, lon in deltas]

            # for j, (lat, lon) in enumerate(deltas):
            #    x_values.append(lat)  # Use negative index for negative lags
            # weight = [-0.55, -0.5, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 0.55]
            # weight2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            # Plotting for each orbit
            for i in range(7):
                y_values = [y**2 for y in lat_lag_sem[day][i]]

                ax.plot(lat_lags, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('symlog', linthresh=0.95)
            # ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(lat_lags)
            ticks = [str(round(x, 2)) for x in lat_lags]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=65, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'
            ax.legend()

        plt.tight_layout()
        save_path = Path(self.save_path) / f'cross_sem_latitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        # plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_latitude_days{days[0]}_{days[-1]}.png')
        plt.show()

    def plot_lat_empirical(self, lat_lag_sem, days, deltas):
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            colors = [
                (222/255, 235/255, 247/255),
                (198/255, 219/255, 239/255),
                (158/255, 202/255, 225/255),
                (107/255, 174/255, 214/255),
                (66/255, 146/255, 198/255),
                (33/255, 113/255, 181/255),
                (8/255, 69/255, 148/255),
                (0/255, 51/255, 128/255)
            ]

            # 1. Extract raw lags from deltas
            raw_lat_lags = [lat for lat, lon in deltas]

            # 2. Get indices that would sort these lags
            # This ensures we plot from smallest lag to largest lag (removes zig-zags)
            sorted_indices = sorted(range(len(raw_lat_lags)), key=lambda k: raw_lat_lags[k])
            
            # 3. Create the sorted x-axis list
            lat_lags = [raw_lat_lags[i] for i in sorted_indices]

            for index, day in enumerate(days):
                ax = axs[index // 2, index % 2]

                for i in range(8):
                    # Get raw y values for this hour
                    raw_y_values = lat_lag_sem[day][i]
                    
                    # 4. Reorder y_values using the SAME sorted_indices
                    # This ensures the Y value matches the correct X (lag) value
                    y_values = [raw_y_values[i] for i in sorted_indices]
                    
                    ax.plot(lat_lags, y_values, color=colors[i], label=f'Empirical Sem. Hour {i + 1}')

                ax.grid(True)
                ax.set_xlabel('Latitude Lags', fontsize=12)
                ax.set_ylabel('Variogram Values', fontsize=12)
                ax.set_title(f'Semi-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
                
                # Use symlog as requested
                ax.set_xscale('symlog', linthresh=0.3)
                

                # if 0.95 needed instead, change linthresh value above

                # Ticks must also be sorted (lat_lags is now sorted)
                ax.set_xticks(lat_lags)
                
                # Format ticks
                ticks = [str(round(x, 2)) for x in lat_lags]
                ax.set_xticklabels(ticks)
                
                ax.set_ylim(1e-4, 25)
                plt.setp(ax.get_xticklabels(), rotation=65, ha='right')

                ax.axvline(x=0, color='red', linestyle='--')

                ax.set_xlim(0, 1.8)
                ax.legend()

            plt.tight_layout()
            save_path = Path(self.save_path) / f'emp_sem_latitude_days{days[0]}_{days[-1]}.png'
            plt.savefig(save_path)
            plt.show()

    def plot_lon_empirical(self, lon_lag_sem, days, deltas):
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            colors = [
                (222/255, 235/255, 247/255),
                (198/255, 219/255, 239/255),
                (158/255, 202/255, 225/255),
                (107/255, 174/255, 214/255),
                (66/255, 146/255, 198/255),
                (33/255, 113/255, 181/255),
                (8/255, 69/255, 148/255),
                (0/255, 51/255, 128/255)  # New 8th color
            ]

            # 1. Extract raw lags
            raw_lon_lags = [lon for lat, lon in deltas]

            # 2. Get indices that would sort these lags
            # This guarantees the line is drawn left-to-right without zig-zags
            sorted_indices = sorted(range(len(raw_lon_lags)), key=lambda k: raw_lon_lags[k])
            
            # 3. Create the sorted x-axis list
            lon_lags = [raw_lon_lags[i] for i in sorted_indices]

            for index, day in enumerate(days):
                ax = axs[index // 2, index % 2]

                for i in range(8):
                    # Get raw y values
                    raw_y_values = lon_lag_sem[day][i]

                    # 4. Reorder y_values using the SAME sorted_indices
                    y_values = [raw_y_values[i] for i in sorted_indices]

                    ax.plot(lon_lags, y_values, color=colors[i], label=f'Empirical Sem. Hour {i + 1}')
                    
                ax.grid(True)
                ax.set_xlabel('Longitude Lags', fontsize=12)
                ax.set_ylabel('Variogram Values', fontsize=12)
                ax.set_title(f'Semi-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
                
                
                ax.set_xscale('symlog', linthresh=0.95)
                
                # Use the sorted lags for ticks
                ax.set_xticks(lon_lags)
                ticks = [str(round(x, 2)) for x in lon_lags]
                ax.set_xticklabels(ticks)
                
                ax.set_ylim(1e-4, 25)
                plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

                # Add vertical red line at x=0
                ax.axvline(x=0, color='red', linestyle='--') 
                ax.legend()

            plt.tight_layout()
            save_path = Path(self.save_path) / f'emp_sem_longitude_days{days[0]}_{days[-1]}.png'
            plt.savefig(save_path)
            plt.show()



    def plot_directional_sem_empirical(self, deltas,direictional_sem, days, direction1, direction2):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        tmp = np.concatenate((np.linspace(-2, -0.2, 10), [-0.1, 0, 0.1], np.linspace(0.2, 2, 10)))


        for index, day in enumerate(days):
            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            # Plotting for each orbit
            for i in range(7):
                y_values = [y**2 for y in direictional_sem[day][i]]
                sign_distance = deltas
                ax.plot(sign_distance, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')
                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            ax.grid(True)
            ax.set_xlabel('sign*Euclidean disstance', fontsize=12)
            ax.set_ylabel('Cross-Variogram Value', fontsize=12)
            ax.set_title(f'Directional Cross-Variogram {direction1*(180/np.pi)}_{direction2*(180/np.pi)} on 2024-07-{day:02d}', fontsize=14)
            ax.set_xscale('linear')  # Linear scale for x-axis
            ax.set_yscale('linear')  # Linear scale for y-axis
            ax.set_xticks(deltas)
            ticks = [ str( round(x,1)) for x in deltas]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)

            plt.setp(ax.get_xticklabels(), rotation=65, ha='right')
            ax.axvline(x=0, color='red', linestyle='--', label='x=0')

        plt.tight_layout()
    
            # Save the plot to the specified directory
        plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_{direction1*(180/np.pi)}_{direction2*(180/np.pi)}_days{days[0]}_{days[-1]}.png')
        plt.show()

# 12/25/25 modified theoretical class for cross-variogram plotting


class CrossVariogram_emp_theory(CrossVariogram):
    def __init__(self, save_path, length_of_analysis, smooth):
        super().__init__(save_path, length_of_analysis)
        self.smooth = smooth
        
    def plot_cross_lon_emp_the(self, lon_lag_sem, days, deltas, df, cov_func):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Color palette
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        for index, day in enumerate(days):
            ax = axs[index // 2, index % 2]

            x_values = [lon for lat, lon in deltas]

            # 1. Plot Empirical
            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')
                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            # 2. Prepare Theoretical Params
            deltas2 = torch.linspace(-2, 2, 40) # 벡터 (크기 40)
            row = df.iloc[day - 1]
            
            # 순서: [sigmasq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget]
            params_list = [
                row['sigma'],       
                row['range_lat'],
                row['range_lon'],
                row['range_time'],  
                row['advec_lat'],
                row['advec_lon'],
                row['nugget']
            ]
            params = torch.tensor(params_list, dtype=torch.float64)

            # 3. Plot Theoretical (Vectorized - Loop Removed)
            # Longitude direction: lat_diff=Scalar(0), lon_diff=Vector(deltas2), time_diff=Scalar(1)
            # 한 번에 40개 포인트를 계산하므로 결과는 배열이 되어 torch.from_numpy 오류가 사라집니다.
            d, gamma = cov_func(
                params, 
                torch.tensor(0.00001),  # lat_diff (scalar)
                deltas2,                # lon_diff (vector)
                torch.tensor(1.0)       # time_diff (scalar)
            )
            
            # gamma는 이제 Tensor입니다. .pow(2)로 제곱하고 numpy로 변환합니다.
            gamma_values = gamma.pow(2).detach().numpy()
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color=(8/255, 69/255, 0), linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # 4. Styling & Scaling
            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            
            ax.set_xscale('symlog', linthresh=0.3)
            
            ax.set_xticks(x_values)
            ticks = [str(round(x, 2)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlim(-1.8, 1.8) 
            ax.legend()

        plt.tight_layout()
        save_path = Path(self.save_path) / f'cross_emp_the_longitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        plt.show()


    def plot_cross_lat_emp_the(self, lon_lag_sem, days, deltas, df, cov_func):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        # 1. Prepare X values (Lags) and Sort Indices
        raw_x_values = [lat for lat, lon in deltas]
        
        sorted_indices = sorted(range(len(raw_x_values)), key=lambda k: raw_x_values[k])
        x_values = [raw_x_values[i] for i in sorted_indices]

        for index, day in enumerate(days):
            ax = axs[index // 2, index % 2]

            # 2. Plot Empirical (Sorted)
            for i in range(7):
                raw_y_values = [y**2 for y in lon_lag_sem[day][i]]
                y_values = [raw_y_values[i] for i in sorted_indices]

                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')
                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            # 3. Prepare Theoretical Params
            deltas2 = torch.linspace(-2, 2, 40) # 벡터
            row = df.iloc[day - 1]
            
            params_list = [
                row['sigma'],       
                row['range_lat'],
                row['range_lon'],
                row['range_time'],  
                row['advec_lat'],
                row['advec_lon'],
                row['nugget']
            ]
            params = torch.tensor(params_list, dtype=torch.float64)

            # 4. Plot Theoretical (Vectorized - Loop Removed)
            # Latitude direction: lat_diff=Vector(deltas2), lon_diff=Scalar(0), time_diff=Scalar(1)
            d, gamma = cov_func(
                params, 
                deltas2,                # lat_diff (vector)
                torch.tensor(0.00001),  # lon_diff (scalar)
                torch.tensor(1.0)       # time_diff (scalar)
            )
            
            # gamma는 Tensor입니다.
            gamma_values = gamma.pow(2).detach().numpy()
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color=(8/255, 69/255, 0), linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # 5. Styling & Scaling
            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            
            ax.set_xscale('symlog', linthresh=0.3)
            
            ax.set_xticks(x_values)
            ticks = [str(round(x, 2)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            ax.axvline(x=0, color='red', linestyle='--')
            
            ax.set_xlim(-1.8, 1.8) 
            ax.legend()

        plt.tight_layout()
        save_path = Path(self.save_path) / f'cross_emp_the_latitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        plt.show()
    


    #########################
    ######################### 아래 사용 안함


    def plot_two_latitude(self, days,lon_lag_sem,  deltas, df1, df2):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))


        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    
        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            x_values = [lat for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # Separate positive and negative lags and assign appropriate indices
        
            
            deltas2 = torch.linspace(-2,2, 40)
            params1 = list(df1.iloc[day - 1][:-1])
            params2 = list(df2.iloc[day - 1][:-1])
    
            gamma_values1 = []
            d_values1 = []
            gamma_values2 = []
            d_values2 = []

            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d1, gamma1 = self.theoretical_gamma_ani_st(params1,delta, 0.00001, 1)
                gamma_values1.append(gamma1.item()**2)  # Convert tensor to scalar for plotting
                d_values1.append(d1.item())

                d2, gamma2 = self.theoretical_gamma_ani_st(params2,delta, 0.00001, 1)
                gamma_values2.append(gamma2.item()**2)  # Convert tensor to scalar for plotting
                d_values2.append(d2.item())
            
            ax.plot(deltas2.numpy(), gamma_values1, label=f"Morning Fitted CV, time lag {1}", color= 'green', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values2, label=f"Noon Fitted CV, time lag {1}", color= 'purple', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 3000)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()
        plt.tight_layout()
        save_path = Path(self.save_path) / f'dir_sem_latitude_halfday{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
       
        plt.show()


    def plot_two_longitude(self, days,lon_lag_sem,  deltas, df1, df2):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))


        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    
        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            x_values = [lon for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # Separate positive and negative lags and assign appropriate indices
        
            deltas2 = torch.linspace(-2,2, 40)
            params1 = list(df1.iloc[day - 1][:-1])
            params2 = list(df2.iloc[day - 1][:-1])
    
            gamma_values1 = []
            d_values1 = []
            gamma_values2 = []
            d_values2 = []

            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d1, gamma1 = self.theoretical_gamma_ani_st(params1,delta, 0.00001, 1)
                gamma_values1.append(gamma1.item()**2)  # Convert tensor to scalar for plotting
                d_values1.append(d1.item())

                d2, gamma2 = self.theoretical_gamma_ani_st(params2,delta, 0.00001, 1)
                gamma_values2.append(gamma2.item()**2)  # Convert tensor to scalar for plotting
                d_values2.append(d2.item())
            
            ax.plot(deltas2.numpy(), gamma_values1, label=f"Morning Fitted CV, time lag {1}", color= 'green', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values2, label=f"Noon Fitted CV, time lag {1}", color= 'purple', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()
        plt.tight_layout()
        save_path = Path(self.save_path) / f'dir_sem_longitude_halfday{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
       
        plt.show()


    def plot_four_latitude(self, days,lon_lag_sem,  deltas, df1, df2,df3,df4):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))


        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    
        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            x_values = [lat for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # Separate positive and negative lags and assign appropriate indices
        
            

            deltas2 = torch.linspace(-2,2, 40)
            params1 = list(df1.iloc[day - 1][:-1])
            params2 = list(df2.iloc[day - 1][:-1])
            params3 = list(df3.iloc[day - 1][:-1])
            params4 = list(df4.iloc[day - 1][:-1])
    
            gamma_values1 = []
            d_values1 = []
            gamma_values2 = []
            d_values2 = []
            gamma_values3 = []
            d_values3 = []
            gamma_values4 = []
            d_values4 = []


            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d1, gamma1 = self.theoretical_gamma_ani_st(params1,delta, 0.00001, 1)
                gamma_values1.append(gamma1.item()**2)  # Convert tensor to scalar for plotting
                d_values1.append(d1.item())

                d2, gamma2 = self.theoretical_gamma_ani_st(params2,delta, 0.00001, 1)
                gamma_values2.append(gamma2.item()**2)  # Convert tensor to scalar for plotting
                d_values2.append(d2.item())

                d3, gamma3 = self.theoretical_gamma_ani_st(params3,delta, 0.00001, 1)
                gamma_values3.append(gamma3.item()**2)  # Convert tensor to scalar for plotting
                d_values3.append(d3.item())

                d4, gamma4 = self.theoretical_gamma_ani_st(params4,delta, 0.00001, 1)
                gamma_values4.append(gamma4.item()**2)  # Convert tensor to scalar for plotting
                d_values4.append(d4.item())
            
            ax.plot(deltas2.numpy(), gamma_values1, label=f"Q1 Fitted CV, time lag {1}", color= 'green', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values2, label=f"Q2 CV, time lag {1}", color= 'orange', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values3, label=f"Q3 CV, time lag {1}", color= 'purple', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values4, label=f"Q4 CV, time lag {1}", color= 'red', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 3000)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()
        plt.tight_layout()
        save_path = Path(self.save_path) / f'dir_sem_latitude_quartday{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
       
        plt.show()
    def plot_four_longitude(self, days,lon_lag_sem,  deltas, df1, df2,df3,df4):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))


        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
        colors = [
            (222/255, 235/255, 247/255),
            (198/255, 219/255, 239/255),
            (158/255, 202/255, 225/255),
            (107/255, 174/255, 214/255),
            (66/255, 146/255, 198/255),
            (33/255, 113/255, 181/255),
            (8/255, 69/255, 148/255)
        ]

        # https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    
        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            ####### this part is important for debugging
            x_values = [lon for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            # Separate positive and negative lags and assign appropriate indices
        
            deltas2 = torch.linspace(-2,2, 40)
            params1 = list(df1.iloc[day - 1][:-1])
            params2 = list(df2.iloc[day - 1][:-1])
            params3 = list(df3.iloc[day - 1][:-1])
            params4 = list(df4.iloc[day - 1][:-1])
    
            gamma_values1 = []
            d_values1 = []
            gamma_values2 = []
            d_values2 = []
            gamma_values3 = []
            d_values3 = []
            gamma_values4 = []
            d_values4 = []


            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d1, gamma1 = self.theoretical_gamma_ani_st(params1,delta, 0.00001, 1)
                gamma_values1.append(gamma1.item()**2)  # Convert tensor to scalar for plotting
                d_values1.append(d1.item())

                d2, gamma2 = self.theoretical_gamma_ani_st(params2,delta, 0.00001, 1)
                gamma_values2.append(gamma2.item()**2)  # Convert tensor to scalar for plotting
                d_values2.append(d2.item())

                d3, gamma3 = self.theoretical_gamma_ani_st(params3,delta, 0.00001, 1)
                gamma_values3.append(gamma3.item()**2)  # Convert tensor to scalar for plotting
                d_values3.append(d3.item())

                d4, gamma4 = self.theoretical_gamma_ani_st(params4,delta, 0.00001, 1)
                gamma_values4.append(gamma4.item()**2)  # Convert tensor to scalar for plotting
                d_values4.append(d4.item())
            
            ax.plot(deltas2.numpy(), gamma_values1, label=f"Q1 Fitted CV, time lag {1}", color= 'green', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values2, label=f"Q2 CV, time lag {1}", color= 'orange', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values3, label=f"Q3 CV, time lag {1}", color= 'purple', linestyle='--', alpha=0.7)
            ax.plot(deltas2.numpy(), gamma_values4, label=f"Q4 CV, time lag {1}", color= 'red', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))

            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()
        plt.tight_layout()
        save_path = Path(self.save_path) / f'dir_sem_longitude_quartday{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
       
        plt.show()