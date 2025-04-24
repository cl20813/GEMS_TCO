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

from GEMS_TCO.kernels import spatio_temporal_kernels

class diagnosis(spatio_temporal_kernels):
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

    def cross_lon_lat(self, deltas, map, days, tolerance):
        lon_lag_sem = {}
        lon_lag_sem[0] = deltas
        for index, day in enumerate(days):
            lon_lag_sem[day] = [[0]*len(deltas) for _ in range(7)]
            t = day - 1
            ori_semi_var_timeseries = [[0] * len(deltas) for _ in range(7)]
            key_list = sorted(map)
            
            for i in range(8 * t, 8 * t + 7):  # change 7 to 6
                cur_data = map[key_list[i]]
                next_data = map[key_list[i+1]]
                coordinates = cur_data[:, :2]
                cur_values = cur_data[:, 2] - torch.mean(cur_data[:, 2])
                next_values = next_data[:, 2] - torch.mean(next_data[:, 2])
                lat_diffs = coordinates[:, None, 0] - coordinates[None, :, 0]
                lon_diffs = coordinates[:, None, 1] - coordinates[None, :, 1]

                for j, (delta_lat, delta_lon) in enumerate(deltas):
                    valid_pairs = np.where(
                        (np.abs(lat_diffs - delta_lat) <= tolerance) & 
                        (np.abs(lon_diffs - delta_lon) <= tolerance)
                    )
            
                    if len(valid_pairs[0]) == 0:
                        print(f"No valid pairs found for t{j+1:02d}_{i+1} at delta ({delta_lat}, {delta_lon})")
                        ori_semi_var_timeseries[i % 8][j] = np.nan
                        continue
                    semivariances = 0.5 * torch.mean((cur_values[valid_pairs[1]] - next_values[valid_pairs[0]]) ** 2)
                    lon_lag_sem[day][i % 8][j] = semivariances.item()
                    ori_semi_var_timeseries[i % 8][j] = semivariances.item()
        return lon_lag_sem
    
    def cross_directional_sem(self,deltas, map,  days, tolerance, direction1, direction2):
        directional_sem = {}
        directional_sem[0] = deltas

        for index, day in enumerate(days):
            directional_sem[day] = [[0]*len(deltas) for _ in range(7)]
            
            t = day - 1
            ori_semi_var_timeseries = [[0] * len(deltas) for _ in range(7)]
            key_list = sorted(map)

            for i in range(8 * t, 8 * t + 7):         # change 7 to 6
                cur_data = map[key_list[i]]
                next_data = map[key_list[i+1]]
                coordinates = cur_data[:,:2]   # latitude and longitude
                cur_values = cur_data[:,2]- torch.mean(cur_data[:,2])
                next_values = (next_data[:,2]) - torch.mean( (next_data[:,2]) )

                # Calculate pairwise differences in both latitude and longitude
                lat_diffs = coordinates[:, None, 0] - coordinates[None, :, 0]
                lon_diffs = coordinates[:, None, 1] - coordinates[None, :, 1]

                # Calculate the pairwise distances between all points

                angle = np.arctan2(lat_diffs, lon_diffs) 
                direction1_filter = np.abs(angle - direction1) <= np.pi/8 
                direction2_filter = np.abs(angle -direction2) <= np.pi/8

                for j, (distance) in enumerate(deltas):


                    if j<= len(deltas)/2:
                        direction_filtered = direction1_filter
                        direction = direction1
                    else:
                        direction_filtered = direction2_filter
                        direction = direction2
                    
                    # Apply the boolean mask directly to the 2D arrays
                    filtered_lat_diffs = lat_diffs*direction_filtered
                    filtered_lon_diffs = lon_diffs*direction_filtered

            
                    # Check if we have valid filtered lat/lon diffs
                    # print(filtered_lat_diffs,filtered_lat_diffs.shape)  # Check the filtered lat diffs before the comparison

                    if distance==0:
                        valid_pairs = np.where(
                            (np.abs(np.sqrt(lat_diffs**2 + lon_diffs**2) - distance) <= tolerance)
                        )
                    else:
                        valid_pairs = np.where(
                            (filtered_lat_diffs != 0) &  # allow earlier filter
                            (np.abs(np.sqrt(filtered_lat_diffs**2 + filtered_lon_diffs**2) - distance) <= tolerance)
                        )

                    if len(valid_pairs[0]) == 0:
                        print(f"No valid pairs found for t{j+1:02d}_{i+1} at lag ({distance},  direction {direction})")
                        ori_semi_var_timeseries[i % 8][j] = np.nan
                        continue

                    # Compute the semivariance for those valid pairs
                    semivariances = 0.5 * torch.mean((cur_values[valid_pairs[0]] - next_values[valid_pairs[1]]) ** 2)
                    
                    # Normalize the semivariance
                    # variance_of_data = np.var(values)
                    # normalized_semivariance = semivariances / variance_of_data

                    directional_sem[day][i % 8][j] = semivariances.item()
                    # Append the normalized semivariance to the timeseries
                    ori_semi_var_timeseries[i % 8][j] = semivariances.item()
                
        return directional_sem


    def theoretical_gamma_ani_st(self,params, lat_diff, lon_diff, time_diff):
        # Unpack parameters
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params
        
        # Calculate the spatial-temporal differences
        tmp1 = lat_diff - advec_lat * time_diff
        tmp2 = lon_diff - advec_lon * time_diff
        tmp3 = beta * time_diff
        
        d = tmp1**2/range_lat**2 + tmp2**2/range_lon**2 + tmp3**2
        
        # Convert d into a tensor (if it's not already) and compute the semivariogram
        d = d.clone().detach()

        out = nugget + sigmasq * (1 - torch.exp(- torch.sqrt(d) ) )
        
        return torch.sqrt(d), out

    def squared_formatter(self,x, pos):
        return f'{np.sqrt(x):.2f}'

    def plot_lon_sem(self, lon_lag_sem, days, deltas):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            # Separate positive and negative lags and assign appropriate indices
            x_values = []
            for j, (lat, lon) in enumerate(deltas):
                x_values.append(lon)  # Use negative index for negative lags
            # weight = [-0.55, -0.5, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 0.55]
            # weight2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            # Plotting for each orbit
            for i in range(7):
                for j, (x, y) in enumerate(zip(x_values, lon_lag_sem[day][i])):
                    ax.scatter(x, y, marker='o', s=9, color='black')
                    # ax.text(x + weight2[i] * 1.5, y + weight[j] * 0.5, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom')

                # Apply offset using transforms
                    offset = transforms.ScaledTranslation(0.04 * i, 0.04 * j, fig.dpi_scale_trans)
                    
                    trans = ax.transData + offset
                    ax.text(x, y, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom', transform=trans)

            ax.grid(True)
            ax.set_xlabel('Longitude Lags', fontsize=12)
            ax.set_ylabel('Cross-Variogram Value', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d}', fontsize=14)
            ax.set_xscale('linear')  # Linear scale for x-axis
            ax.set_yscale('linear')  # Linear scale for y-axis
            ax.set_xticks(x_values)
            ticks = [ str( round(x,1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 30)
            # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--', label='x=0')

        plt.tight_layout()
        plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_longitude_days{days[0]}_{days[-1]}.png')
        plt.show()

    def plot_lat_sem(self, lon_lag_sem, days, deltas):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        for index, day in enumerate(days):
            # t = day - 1
            # key_list = sorted(map)

            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            # Separate positive and negative lags and assign appropriate indices
            x_values = []
            for j, (lat, lon) in enumerate(deltas):
                x_values.append(lat)  # Use negative index for negative lags
            # weight = [-0.55, -0.5, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 0.55]
            # weight2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            # Plotting for each orbit
            for i in range(7):
                for j, (x, y) in enumerate(zip(x_values, lon_lag_sem[day][i])):
                    ax.scatter(x, y, marker='o', s=9, color='black')
                    # ax.text(x + weight2[i] * 1.5, y + weight[j] * 0.5, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom')

                    offset = transforms.ScaledTranslation(0.04 * i, 0.04 * j, fig.dpi_scale_trans)
                    
                    trans = ax.transData + offset
                    ax.text(x, y, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom', transform=trans)


            ax.grid(True)
            ax.set_xlabel('Latitude Lags', fontsize=12)
            ax.set_ylabel('Cross-Variogram Value', fontsize=12)
            ax.set_title(f'Cross-Variogram on 2024-07-{day:02d}', fontsize=14)
            ax.set_xscale('linear')  # Linear scale for x-axis
            ax.set_yscale('linear')  # Linear scale for y-axis
            ax.set_xticks(x_values)
            ticks = [ str( round(x,1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 60)
        # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--', label='x=0')

        plt.tight_layout()
        plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_latitude_days{days[0]}_{days[-1]}.png')
        plt.show()


    def plot_directional_sem(self,x_values,direictional_sem, days, direction1, direction2):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        for index, day in enumerate(days):
            # Create a 2x2 plot
            ax = axs[index // 2, index % 2]

            # weight = [-0.55, -0.5, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 0.55]
            # weight2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            # Plotting for each orbit
            for i in range(7):
                for j, (x, y) in enumerate(zip(x_values, direictional_sem[day][i])):
                    ax.scatter(x, y, marker='o', s=9, color='black')
                   # ax.text(x + weight2[i] * 1.5, y + weight[j] * 0.5, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom')
                    offset = transforms.ScaledTranslation(0.04 * i, 0.04 * j, fig.dpi_scale_trans)
                    
                    
                    trans = ax.transData + offset
                    ax.text(x, y, str(i + 1), fontsize=9, color='blue', ha='center', va='bottom', transform=trans)

            ax.grid(True)
            ax.set_xlabel('Euclidean disstance', fontsize=12)
            ax.set_ylabel('Cross-Variogram Value', fontsize=12)
            ax.set_title(f'Directional Cross-Variogram {direction1*(180/np.pi)}_{direction2*(180/np.pi)} on 2024-07-{day:02d}', fontsize=14)
            ax.set_xscale('linear')  # Linear scale for x-axis
            ax.set_yscale('linear')  # Linear scale for y-axis
            ax.set_xticks(x_values)
            ticks = [ str( round(x,1)) for x in x_values]
            ax.set_xticklabels(ticks)
            ax.set_ylim(1e-4, 30)


        # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

            # Add vertical red line at x=0
            ax.axvline(x=0, color='red', linestyle='--', label='x=0')

        plt.tight_layout()
    
            # Save the plot to the specified directory
        plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_{direction1*(180/np.pi)}_{direction2*(180/np.pi)}_days{days[0]}_{days[-1]}.png')
        plt.show()





class CrossVariogram:
    def __init__(self, save_path, length_of_analysis):
        self.save_path = save_path
        self.length_of_analysis = length_of_analysis
        

    def cross_lon_lat(self, deltas, map, days, tolerance):
        lon_lag_sem = {}
        lon_lag_sem[0] = deltas
        for index, day in enumerate(days):
            lon_lag_sem[day] = [[0]*len(deltas) for _ in range(7)]
            t = day - 1
            ori_semi_var_timeseries = [[0] * len(deltas) for _ in range(7)]
            key_list = sorted(map)
            
            for i in range(8 * t, 8 * t + 7):  # change 7 to 6
                cur_data = map[key_list[i]]
                next_data = map[key_list[i+1]]
                coordinates = cur_data[:, :2]
                cur_values = cur_data[:, 2] - torch.mean(cur_data[:, 2])
                next_values = next_data[:, 2] - torch.mean(next_data[:, 2])
                lat_diffs = coordinates[:, None, 0] - coordinates[None, :, 0]
                lon_diffs = coordinates[:, None, 1] - coordinates[None, :, 1]

                for j, (delta_lat, delta_lon) in enumerate(deltas):
                    valid_pairs = np.where(
                        (np.abs(lat_diffs - delta_lat) <= tolerance) & 
                        (np.abs(lon_diffs - delta_lon) <= tolerance)
                    )
            
                    if len(valid_pairs[0]) == 0:
                        print(f"No valid pairs found for t{j+1:02d}_{i+1} at delta ({delta_lat}, {delta_lon})")
                        ori_semi_var_timeseries[i % 8][j] = np.nan
                        continue
                    semivariances = 0.5 * torch.mean((cur_values[valid_pairs[1]] - next_values[valid_pairs[0]]) ** 2)
                    lon_lag_sem[day][i % 8][j] = semivariances.item()
                    ori_semi_var_timeseries[i % 8][j] = semivariances.item()
        return lon_lag_sem


    def theoretical_gamma_ani_st(self,params, lat_diff, lon_diff, time_diff):
        # Unpack parameters
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params
        
        # Calculate the spatial-temporal differences
        tmp1 = lat_diff - advec_lat * time_diff
        tmp2 = lon_diff - advec_lon * time_diff
        tmp3 = beta * time_diff
        
        d = tmp1**2/range_lat**2 + tmp2**2/range_lon**2 + tmp3**2
        
        # Convert d into a tensor (if it's not already) and compute the semivariogram
        d = d.clone().detach()

        out = nugget + sigmasq * (1 - torch.exp(- torch.sqrt(d) ) )
        
        return torch.sqrt(d), out

    def squared_formatter(self,x, pos):
        return f'{np.sqrt(x):.2f}'
    
class CrossVariogram_emp_theory(CrossVariogram):
    def __init__(self, save_path, length_of_analysis):
        super().__init__(save_path, length_of_analysis)
        
    def plot_lon_emp_the(self, lon_lag_sem, days, deltas,df):
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
           
            # params = [21.28, 9.92, 3.335, -1.2099, -0.1238,-0.0151, 3.4034]

            gamma_values = []
            d_values = []
            for delta in deltas2:
            # Calculate theoretical semivariogram for this distance and time lag
                d, gamma = self.theoretical_gamma_ani_st(params, 0.00001, delta, 1)
        
                gamma_values.append(gamma.item()**2)  # Convert tensor to scalar for plotting
                d_values.append(d.item())
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color= (8/255, 69/255, 0), linestyle='--', alpha=0.7)
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
        # plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_longitude_days{days[0]}_{days[-1]}.png')
        save_path = Path(self.save_path) / f'dir_sem_longitude_days{days[0]}_{days[-1]}.png'
        plt.savefig(save_path)
        plt.show()


    def plot_lat_emp_the(self, lon_lag_sem, days, deltas,df):
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
                d, gamma = self.theoretical_gamma_ani_st(params,delta, 0.00001, 1)
        
                gamma_values.append(gamma.item()**2)  # Convert tensor to scalar for plotting
                d_values.append(d.item())
            
            ax.plot(deltas2.numpy(), gamma_values, label=f"Fitted CV, time lag {1}", color= (8/255, 69/255, 0), linestyle='--', alpha=0.7)
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
        save_path = Path(self.save_path) / f'dir_sem_latitude_days{days[0]}_{days[-1]}.png'
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