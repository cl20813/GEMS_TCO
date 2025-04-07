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

# Type hints
from typing import Callable, Union, Tuple

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
    def __init__(self):
        pass

    def cross_lon_lat(self,deltas, map,  days, tolerance):
        lon_lag_sem = {}
        lon_lag_sem[0] = deltas
        for index, day in enumerate(days):
            lon_lag_sem[day] = [[0]*len(deltas) for _ in range(7)]
            t = day - 1
            ori_semi_var_timeseries = [[0] * len(deltas) for _ in range(7)]
            key_list = sorted(map)
            
            for i in range(8 * t, 8 * t + 7):         # change 7 to 6
                cur_data = map[key_list[i]]
                next_data = map[key_list[i+1]]
                # coordinates = np.array(cur_data[['Latitude', 'Longitude']])
                # cur_values = np.array(cur_data['ColumnAmountO3']) - np.mean(np.array(cur_data['ColumnAmountO3']))
                coordinates = cur_data[:,:2]
                cur_values = cur_data[:,2]- torch.mean(cur_data[:,2])

                # next_values = np.array(next_data['ColumnAmountO3']) - torch.mean( np.array(next_data['ColumnAmountO3'])  )
                next_values = (next_data[:,2]) - torch.mean( (next_data[:,2]) )
                # Calculate pairwise differences in both latitude and longitude
                lat_diffs = coordinates[:, None, 0] - coordinates[None, :, 0]
                lon_diffs = coordinates[:, None, 1] - coordinates[None, :, 1]

                # Calculate the pairwise distances between all points
            
                for j, (delta_lat, delta_lon) in enumerate(deltas):

                    valid_pairs = np.where(
                        (np.abs(lat_diffs - delta_lat) <= tolerance) & 
                        (np.abs(lon_diffs - delta_lon) <= tolerance)
                    )
            
                    if len(valid_pairs[0]) == 0:
                        print(f"No valid pairs found for t{j+1:02d}_{i+1} at delta ({delta_lat}, {delta_lon})")
                        ori_semi_var_timeseries[i % 8][j] = np.nan
                        continue
                    # Compute the semivariance for those valid pairs
                    semivariances = 0.5 * torch.mean(( cur_values[valid_pairs[1]] - next_values[valid_pairs[0]]  ) ** 2)

                    # Normalize the semivariance
                    #variance_of_data = torch.var(values)
                    #normalized_semivariance = semivariances 

                    # Append the normalized semivariance to the timeseries
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
            ax.set_ylim(1e-4, 30)
        # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

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

        plt.tight_layout()

            # Save the plot to the specified directory
        plt.savefig(f'/Users/joonwonlee/Documents/GEMS_TCO-1/plots/directional_semivariograms/dir_sem_{direction1*(180/np.pi)}_{direction2*(180/np.pi)}_days{days[0]}_{days[-1]}.png')

        plt.show()