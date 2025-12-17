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

# Custom imports
# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/evaluate.log'

from GEMS_TCO.kernels_oct25 import spatio_temporal_kernels

class diagnosis(spatio_temporal_kernels):
    ''' 
    Investigate the value  (Data - conditional mean from Vecchia)/ Conditional sd from Vecchia
    See if these are standard normal
    '''
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        super().__init__(smooth, input_map, nns_map, mm_cond_number)
        # Any additional initialization for dignosis class can go here
    
    def diagnosis_method1(self, params, covariance_function):
        res = np.zeros( (self.number_of_timestamps, self.size_per_hour))
        for time_idx in range(self.number_of_timestamps):
            current_np = self.input_map[self.key_list[time_idx]]
            
            for index in range(31, self.size_per_hour):

                current_row = current_np[index]
    
                current_row = current_row.reshape(1,-1)
                current_y = current_row[0][2]

                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    data_list.append( current_np[past])
            
                if time_idx >0:
                    last_hour_np = self.input_map[self.key_list[time_idx-1]]
                
                    past_conditioning_data = last_hour_np[ (past+[index]),: ]
                    data_list.append( past_conditioning_data)
                
                if data_list:
                    conditioning_data = np.vstack(data_list)
                else:
                    conditioning_data = np.array([]).reshape(0, current_row.shape[1])

                np_arr = np.vstack( (current_row, conditioning_data) )
                y_and_neighbors = np_arr[:,2]
                locs = np_arr[:,:2]

                cov_matrix = covariance_function(params=params, y = np_arr, x = np_arr)
        
                cov_xx = cov_matrix[1:,1:]
                cov_yx = cov_matrix[0,1:]
                
                tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
                tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y_and_neighbors))
                beta = np.linalg.solve(tmp1, tmp2)

                mu = np.dot(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # mean and variance of y|x
                sigma = cov_matrix[0][0]
                cov_ygivenx = sigma - np.dot(cov_yx.T,np.linalg.solve(cov_xx, cov_yx))
                sd_ygivenx = np.sqrt(cov_ygivenx)
                # cov_ygivenx = max(cov_ygivenx, 7)
                mean_ygivenx = mu_current + np.dot(cov_yx.T, np.linalg.solve( cov_xx, (y_and_neighbors[1:]-mu_neighbors) ))   # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
                # print(f'cond_mean{mean_z}')

                res[time_idx,index ] = (current_y - mean_ygivenx)/ sd_ygivenx
        return res
    
class CrossVariogram:
    def __init__(self, save_path, length_of_analysis):
        self.save_path = save_path
        self.length_of_analysis = length_of_analysis

    def compute_directional_semivariogram(self, deltas, map_data, days, tolerance):
            """
            Optimized to pre-compute spatial indices once, avoiding N^2 loops per time step.
            """
            lon_lag_sem = {0: deltas}  # Save deltas for reference
            num_lags = 8
            key_list = sorted(map_data.keys())

            # --- STEP 1: Pre-compute Pair Indices (Run Once) ---
            print("Pre-computing spatial pairs for defined lags...")
            
            # Get coordinates from the first available timestamp
            first_data = map_data[key_list[0]]
            coordinates = first_data[:, :2]  # Shape (N, 2)

            # Use GPU if available for faster indexing
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            coords_t = torch.tensor(coordinates, device=device, dtype=torch.float32)
            
            # Store valid indices for each delta: list of (row_idx, col_idx) tensors
            delta_indices_cache = []

            # Calculate difference matrices ONCE
            # Note: If N is very large (>30k), do this in blocks to avoid OOM.
            # For N=20k, this takes ~3GB VRAM/RAM which is usually fine.
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
            for day in days:
                # Initialize storage for this day
                lon_lag_sem[day + 1] = [[np.nan] * len(deltas) for _ in range(num_lags)]
                
                for i in range(num_lags):
                    # Calculate global index (e.g., day 0 -> indices 0-7)
                    global_idx = 8 * day + i
                    if global_idx >= len(key_list):
                        break

                    # Load data for this specific hour
                    cur_data = map_data[key_list[global_idx]]
                    
                    # Prepare values (centered)
                    vals = torch.tensor(cur_data[:, 2], device=device, dtype=torch.float32)
                    vals = vals - torch.mean(vals)

                    # Iterate through pre-computed lags
                    for j, (idx_row, idx_col) in enumerate(delta_indices_cache):
                        if len(idx_row) == 0:
                            # No pairs found for this lag (already set to NaN or handled here)
                            continue
                        
                        # Vectorized calculation: gather values using cached indices
                        # This is O(K) where K is number of pairs, extremely fast
                        diffs = vals[idx_col] - vals[idx_row]
                        semivariance = 0.5 * torch.mean(diffs ** 2)
                        
                        lon_lag_sem[day + 1][i][j] = round(semivariance.item(), 4)
                        
            return lon_lag_sem

    def compute_cross_lon_lat(self, deltas, map_data, days, tolerance):
            # 1. Initialize logic matches original
            lon_lag_sem = {0: deltas}
            num_lags = 7  # Matches your variable
            key_list = sorted(map_data.keys())
            
            # --- PRE-COMPUTATION (Optimization) ---
            # Calculates indices ONCE. Logic remains: (row, col)
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

            for day in days:
                # Matches your initialization
                lon_lag_sem[day + 1] = [[np.nan] * len(deltas) for _ in range(num_lags)]
                
                # Loop range matches: 8*day to 8*day + num_lags
                for i in range(num_lags): 
                    t_idx = 8 * day + i
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
                        
                        # LOGIC CHECK: Preserved Original Indexing
                        # Original: cur_values[valid_pairs[1]] - next_values[valid_pairs[0]]
                        # valid_pairs[1] is col, valid_pairs[0] is row
                        diffs = cur_vals[idx_col] - next_vals[idx_row]
                        
                        semivariance = 0.5 * torch.mean(diffs ** 2)
                        
                        # LOGIC CHECK: Rounding enabled (matches original)
                        lon_lag_sem[day + 1][i][j] = round(semivariance.item(), 4)
                        
            return lon_lag_sem

    def cross_directional_sem(self, deltas, map_data, days, tolerance, direction1, direction2):
        directional_sem = {}
        directional_sem[0] = deltas
        
        # --- PRE-COMPUTATION ---
        key_list = sorted(map_data.keys())
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
            # Direction Logic: Matches your "j <= len/2" check
            target_dir = direction1 if j <= mid_point else direction2
            
            # Mask Logic: Matches "abs(angle - dir) <= pi/8"
            dir_mask = torch.abs(angle_matrix - target_dir) <= (np.pi / 8)
            dist_mask = torch.abs(dist_matrix - distance) <= tolerance
            
            if distance == 0:
                final_mask = dist_mask # Matches "if distance==0"
            else:
                final_mask = dir_mask & dist_mask 
                # Note: This is SAFER than original "filtered_lat_diffs != 0" 
                # because it doesn't accidentally drop pairs with lat_diff=0

            pairs = torch.nonzero(final_mask, as_tuple=True)
            delta_indices_cache.append(pairs)

        del lat_diffs, lon_diffs, dist_matrix, angle_matrix, coords_t
        if device == 'cuda': torch.cuda.empty_cache()
        # -----------------------

        for index, day in enumerate(days):
            # LOGIC FIX: Initialize with NaN, not 0
            directional_sem[day] = [[np.nan]*len(deltas) for _ in range(7)]
            
            # Logic Check: Preserved "t = day - 1"
            t = day - 1 

            for i in range(7): # Matches "range(8*t, 8*t+7)"
                global_idx = 8 * t + i
                
                # Safety for negative indexing or out of bounds
                if global_idx < 0: global_idx += len(key_list) 
                if global_idx + 1 >= len(key_list): break

                cur_data = map_data[key_list[global_idx]]
                next_data = map_data[key_list[global_idx + 1]]

                cur_vals = torch.tensor(cur_data[:, 2], device=device, dtype=torch.float32)
                cur_vals = cur_vals - torch.mean(cur_vals)
                
                next_vals = torch.tensor(next_data[:, 2], device=device, dtype=torch.float32)
                next_vals = next_vals - torch.mean(next_vals)

                for j, (idx_row, idx_col) in enumerate(delta_indices_cache):
                    if len(idx_row) == 0:
                        continue

                    # LOGIC CHECK: Preserved Original Indexing (Swapped compared to first func)
                    # Original: cur_values[valid_pairs[0]] - next_values[valid_pairs[1]]
                    # valid_pairs[0] is row, valid_pairs[1] is col
                    diffs = cur_vals[idx_row] - next_vals[idx_col]
                    
                    semivariance = 0.5 * torch.mean(diffs ** 2)
                    
                    # LOGIC CHECK: No Rounding (matches original)
                    directional_sem[day][i][j] = semivariance.item()

        return directional_sem

    def theoretical_gamma_kv(self, params: torch.Tensor, lat_diff: torch.Tensor, lon_diff: torch.Tensor, time_diff: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Computes the covariance for anisotropic spatial-temporal data using the Matérn covariance function.

        Parameters:
        - params (torch.Tensor): Tensor containing [sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget].
        - lat_diff (torch.Tensor): Tensor of latitude differences.
        - lon_diff (torch.Tensor): Tensor of longitude differences.
        - time_diff (torch.Tensor): Tensor of time differences.

        Returns:
        - torch.Tensor: Spatial-temporal differences.
        - torch.Tensor: Covariance values.
        """
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params
        
        tmp1 = lat_diff - advec_lat * time_diff
        tmp2 = lon_diff - advec_lon * time_diff
        tmp3 = beta * time_diff
        
        d = tmp1**2/range_lat**2 + tmp2**2/range_lon**2 + tmp3**2
        
        d = d.clone().detach()

         # since d is tensor, kv() returns tensor
        tmp = kv(self.smooth, np.sqrt(d)).clone().detach()
        out_tmp = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                (torch.sqrt(d))**self.smooth * tmp)
        
        out = (nugget + sigmasq) - out_tmp
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
                ax.set_xscale('symlog', linthresh=0.95)

                # Ticks must also be sorted (lat_lags is now sorted)
                ax.set_xticks(lat_lags)
                
                # Format ticks
                ticks = [str(round(x, 2)) for x in lat_lags]
                ax.set_xticklabels(ticks)
                
                ax.set_ylim(1e-4, 25)
                plt.setp(ax.get_xticklabels(), rotation=65, ha='right')

                ax.axvline(x=0, color='red', linestyle='--')
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



class CrossVariogram_emp_theory(CrossVariogram):
    def __init__(self, save_path, length_of_analysis, smooth):
        super().__init__(save_path, length_of_analysis)
        self.smooth = smooth
        
    def plot_cross_lon_emp_the(self, lon_lag_sem, days, deltas,df, cov_func):
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

        # colors = plt.cm.autumn(np.linspace(1, 0, 8))

        for index, day in enumerate(days):
            ax = axs[index // 2, index % 2]

            x_values = [lon for lat, lon in deltas]

            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]
                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            deltas2 = torch.linspace(-2,2, 40)
            params = list(df.iloc[day - 1][:-1])
            gamma_values = []
            d_values = []
            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d, gamma = cov_func(params, 0.00001, delta, 1)
        
                gamma_values.append(gamma.item()**2)  # Convert tensor to scalar for plotting
                d_values.append(d.item())
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color= (8/255, 69/255, 0), linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))


            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('symlog', linthresh=0.95)
            # ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 2)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()

        plt.tight_layout()
        # plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_longitude_days{days[0]}_{days[-1]}.png')
        save_path = Path(self.save_path) / f'cross_emp_the_longitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        plt.show()


    def plot_cross_lat_emp_the(self, lon_lag_sem, days, deltas,df, cov_func):
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
        
            x_values = [lat for lat, lon in deltas]

            # for j, (lat, lon) in enumerate(deltas):
            #    x_values.append(lat)  # Use negative index for negative lags
            # weight = [-0.55, -0.5, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 0.55]
            # weight2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            # Plotting for each orbit
            for i in range(7):
                y_values = [y**2 for y in lon_lag_sem[day][i]]

                ax.plot(x_values, y_values, color=colors[i], label=f'Empirical CV Hour {i + 1} to {i+2}')

                ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))
                
            deltas2 = torch.linspace(-2,2, 40)
            params = list(df.iloc[day - 1][:-1])

            gamma_values = []
            d_values = []
            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d, gamma =  cov_func(params,delta, 0.00001, 1)
        
                gamma_values.append(gamma.item()**2)  # Convert tensor to scalar for plotting
                d_values.append(d.item())
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color= (8/255, 69/255, 0), linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(self.squared_formatter))


            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-variogram Values', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d} ({self.length_of_analysis})', fontsize=14)
            ax.set_xscale('symlog', linthresh=0.95)
            #ax.set_xscale('linear')
            # ax.set_yscale('linear')
            ax.set_xticks(x_values)
            ticks = [str(round(x, 2)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 670)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--') # , label='x=0'

            ax.legend()

        plt.tight_layout()
        save_path = Path(self.save_path) / f'cross_emp_the_latitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        # plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_latitude_days{days[0]}_{days[-1]}.png')
        plt.show()
    
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