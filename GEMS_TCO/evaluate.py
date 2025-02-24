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