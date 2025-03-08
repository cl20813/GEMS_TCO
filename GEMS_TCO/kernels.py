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
from sklearn.preprocessing import MinMaxScaler

# Special functions and optimizations
from scipy.spatial.distance import cdist  # For space and time distance
from scipy.special import gamma, kv  # Bessel function and gamma function
from scipy.optimize import minimize
from scipy.optimize import basinhopping, minimize
from scipy.stats import norm,uniform
from scipy.stats import t
import torch

# Type hints
from typing import Callable, Union, Tuple




# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

# Custom imports
# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'


class spatio_temporal_kernels:               #sigmasq range advec beta  nugget
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        self.smooth = smooth
        self.input_map = input_map
        self.key_list = sorted(input_map)
        self.number_of_timestamps = len(self.key_list)

        sample_df = input_map[self.key_list[0]]

        self.size_per_hour = len(sample_df)
        self.mm_cond_number = mm_cond_number
        nns_map = list(nns_map) # nns_map is ndarray this allows to have sub array of diffrent lengths
        for i in range(len(nns_map)):  
            # Select elements up to mm_cond_number and remove -1
            tmp = np.delete(nns_map[i][:self.mm_cond_number], np.where(nns_map[i][:self.mm_cond_number] == -1))
            if tmp.size>0:
                nns_map[i] = tmp
            else:
                nns_map[i] = []
        self.nns_map = nns_map
    
        ''' 
        # gneiting model (gneting 2002)
        https://www.tandfonline.com/doi/abs/10.1198/016214502760047113

        See the equation (14) on page 5/12.

        $C(s,t) = \frac{\sigma^2}{ (a|t|^{2\alpha}+1 )^{\tau}} \exp(  \frac{ -c||s||^{2\gamma}}{(a|t|^{2\alpha}+1 )^{\beta \gamma}})$, 

        where 
        $s \in R^2$ is spatial distance and $t \in R^1$ is temporal distance.   
        a: scaling parameter of time, non-negative   
        c: scaling parameter of space, non-negative   
        $\alpha, \gamma$: smooth parameter of time, and space. both $ \alpha, \gamma \in (0,1]$      
        $\beta, \tau$: space and time interaction parameters. $\tau >=d/2 = 1$, $\beta \in [0,1]$.  
        '''

    def gneiting_cov_yx(self, params: Tuple[float, float, float, float, float, float, float], y: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, c, alpha, gamma, tau, beta, sigma  = params                 ### x for just consistency with other functions
        nugget = 0.1

        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")
     
        # Extract values
        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3] # hour

        # Efficient distance computation using cdist (operation is vectorized)
        coords1 = np.stack((x1, y1), axis=-1)
        coords2 = np.stack((x2, y2), axis=-1)

        s_dist = cdist(coords1, coords2, 'euclidean')
        t_dist = cdist(t1[:, None], t2[:, None], 'euclidean')  # Ensure t is a 2D array for cdist
        
        # Compute the covariance matrix
        tmp1 = sigma**2 / (a * t_dist**(2 * alpha) + 1)**tau
        tmp2 = np.exp(-c * s_dist**(2 * gamma) / (a * t_dist**(2 * alpha) + 1)**(beta * gamma))
        out = tmp1 * tmp2

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget
        return out

    # Custom distance function for cdist
    def custom_distance_matrix(self,U, V, sqrt_range_mat):
        diff = U[:, :2, np.newaxis] - V[:, :2].T
        d = np.tensordot(sqrt_range_mat, diff, axes=(1, 1))

    
        spatial_diff = np.linalg.norm(d, axis=0)
        temporal_diff = np.abs(U[:, 2, np.newaxis] - V[:, 2].T)
        return np.sqrt(spatial_diff**2 + temporal_diff**2)
    
    def matern_cov_yx_v05(self,params: Tuple[float,float,float,float,float,float], y: np.ndarray, x: np.ndarray) -> np.ndarray:
    
        sigmasq, range_lat, range_lon, advec, beta, nugget  = params
        # Validate inputs
        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")
        
        # Extract values
        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3] # hour

        spat_coord1 = np.stack((x1- advec*t1, y1 - advec*t1), axis=-1)
        spat_coord2 = np.stack((x2- advec*t2, y2 - advec*t2), axis=-1)

        coords1 = np.hstack ((spat_coord1, (beta * t1).reshape(-1,1) ))
        coords2 = np.hstack ((spat_coord2, (beta * t2).reshape(-1,1) ))

        sqrt_range_mat = np.diag([ 1/range_lat**0.5, 1/range_lon**0.5])
        self.sqrt_range_mat = sqrt_range_mat


        # distance = cdist(coords1,coords2, metric = 'self.custom_distance_matrix')
        distance = self.custom_distance_matrix(coords1, coords2, self.sqrt_range_mat)

        # Initialize the covariance matrix with zeros
        out = distance
        
        # Compute the covariance for non-zero distances
        # Compute the covariance for non-zero distances
        non_zero_indices = distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = sigmasq* np.exp(-distance[non_zero_indices])
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget

        return out
    
    def matern_cov_yx(self,params: Tuple[float,float,float,float,float,float], y: np.ndarray, x: np.ndarray) -> np.ndarray:
    
        sigmasq, range_lat, range_lon, advec, beta, nugget  = params
        # Validate inputs
        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")
        # Extract values
        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3] # hour

        spat_coord1 = np.stack((x1- advec*t1, y1 - advec*t1), axis=-1)
        spat_coord2 = np.stack((x2- advec*t2, y2 - advec*t2), axis=-1)

        coords1 = np.hstack ((spat_coord1, (beta * t1).reshape(-1,1) ))
        coords2 = np.hstack ((spat_coord2, (beta * t2).reshape(-1,1) ))

        sqrt_range_mat = np.diag([ 1/np.sqrt(range_lat), 1/np.sqrt(range_lon)])
        self.sqrt_range_mat = sqrt_range_mat

        distance = self.custom_distance_matrix(coords1, coords2, self.sqrt_range_mat)
        # Initialize the covariance matrix with zeros
        out = distance
        
        # Compute the covariance for non-zero distances
        non_zero_indices = distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                                    (distance[non_zero_indices] )**self.smooth *
                                    kv(self.smooth, distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget

        return out

    def matern_cov_yx_test1(self, params: Tuple[float, float, float, float, float,float], y: np.ndarray, x: np.ndarray) -> np.ndarray:
        sigmasq, range_lat, range_lon, sigmasq2, range_tmp, nugget = params
        # Validate inputs
        if y is None or x is None:
            raise ValueError("Both y and x must be provided.")
        # Extract values
        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 2]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 2]  # hour

        spat_coord1 = np.stack((x1, y1), axis=-1)
        spat_coord2 = np.stack((x2, y2), axis=-1)
        # Scale spatial coordinates by the square root of the range parameters
        sqrt_range_mat = np.diag([1/ np.sqrt(range_lat), 1/np.sqrt(range_lon)])
        scaled_spat_coord1 = np.dot(spat_coord1, sqrt_range_mat)
        scaled_spat_coord2 = np.dot(spat_coord2, sqrt_range_mat)

        # Calculate spatial distances
        spat_distance = cdist(scaled_spat_coord1, scaled_spat_coord2, metric='euclidean')
        temp_coord1 = t1.reshape(-1, 1)
        temp_coord2 = t2.reshape(-1, 1)
        tmp_distance = cdist(temp_coord1, temp_coord2, metric='euclidean')
        temp_distance = tmp_distance / range_tmp
      
        # Initialize the covariance matrix with zeros
        out = np.zeros_like(spat_distance)

        non_zero_indices = spat_distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = (sigmasq * (2**(1 - self.smooth)) / gamma(self.smooth) *
                                    (spat_distance[non_zero_indices] )**self.smooth *
                                    kv(self.smooth, spat_distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # it seems vecchia is catching the behavior when smooth = 0.5  if 1 full likelihood better but worse approximation
        # and   15 1.25 1.25 12 1 0.2 guaranteed to be better. 
        
        smooth_time = 0.5
        non_zero_indices = temp_distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] *= (sigmasq2 * (2**(1 -smooth_time)) / gamma(smooth_time) *
                                    (temp_distance[non_zero_indices])*smooth_time *
                                    kv(smooth_time, temp_distance[non_zero_indices] ))
        out[~non_zero_indices] *= sigmasq2

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget
        

        return out
        
class likelihood_function_testing(spatio_temporal_kernels):
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        super().__init__(smooth, input_map, nns_map, mm_cond_number)
        # Any additional initialization for dignosis class can go here
     

    def matern_cov_yx(self, params: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        sigmasq, range_lat, range_lon, advec, beta, nugget = params
        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")

        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3]

        spat_coord1 = torch.stack((x1 - advec * t1, y1 - advec * t1), dim=-1)
        spat_coord2 = torch.stack((x2 - advec * t2, y2 - advec * t2), dim=-1)

        coords1 = torch.cat((spat_coord1, (beta * t1).reshape(-1, 1)), dim=1)
        coords2 = torch.cat((spat_coord2, (beta * t2).reshape(-1, 1)), dim=1)

        sqrt_range_mat = torch.diag(torch.tensor([1 / torch.sqrt(range_lat), 1 / torch.sqrt(range_lon)], dtype=torch.float32))

        def custom_distance_matrix(U, V, sqrt_range_mat):
            diff = U[:, :2].unsqueeze(1) - V[:, :2].unsqueeze(0)
            d = torch.tensordot(sqrt_range_mat, diff, dims=([1], [2]))
            spatial_diff = torch.norm(d, dim=0)
            temporal_diff = torch.abs(U[:, 2].unsqueeze(1) - V[:, 2].unsqueeze(0))
            return torch.sqrt(spatial_diff**2 + temporal_diff**2)

        distance = custom_distance_matrix(coords1, coords2, sqrt_range_mat)
        out = torch.zeros_like(distance)

        non_zero_indices = distance != 0
        if torch.any(non_zero_indices):
            out[non_zero_indices] = sigmasq * torch.exp(-distance[non_zero_indices])
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += torch.eye(out.shape[0]) * nugget

        return out
                
    def full_likelihood(self, params: torch.Tensor, input_np: torch.Tensor, y: torch.Tensor, covariance_function) -> torch.Tensor:
        input_arr = input_np[:, :4]
        y_arr = y

        # Compute the covariance matrix
        cov_matrix = covariance_function(params=params, y=input_arr, x=input_arr)
        
        # Compute the log determinant of the covariance matrix
        sign, log_det = torch.slogdet(cov_matrix)
        # if sign <= 0:
        #     raise ValueError("Covariance matrix is not positive definite")
        
        # Extract locations
        locs = input_arr[:, :2]

        # Compute beta
        tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
        tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_arr))
        beta = torch.linalg.solve(tmp1, tmp2)

        # Compute the mean
        mu = torch.matmul(locs, beta)
        y_mu = y_arr - mu

        # Compute the quadratic form
        quad_form = torch.matmul(y_mu, torch.linalg.solve(cov_matrix, y_mu))

        # Compute the negative log likelihood
        neg_log_lik = 0.5 * (log_det + quad_form)

        return neg_log_lik
    
    def vecchia_like_local(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        self.cov_map = defaultdict(list)
        neg_log_lik = 0.0
        
        for time_idx in range(self.number_of_timestamps):
            current_np = self.input_map[self.key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            cur_heads = current_np[:21, :]
            neg_log_lik += self.full_likelihood(params, cur_heads, cur_heads[:, 2], covariance_function)

            for index in range(21, self.size_per_hour):
                current_row = current_np[index].reshape(1, -1)
                current_y = current_row[0, 2]

                # Construct conditioning set
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    data_list.append(current_np[past])

                if time_idx > 1:
                    cov_matrix = self.cov_map[index]['cov_matrix']
                    tmp_for_beta = self.cov_map[index]['tmp_for_beta']
                    cov_xx_inv = self.cov_map[index]['cov_xx_inv']
                    L_inv = self.cov_map[index]['L_inv']
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx']
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs = self.cov_map[index]['locs']
                    
                    last_hour_np = self.input_map[self.key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                    if data_list:
                        conditioning_data = torch.vstack(data_list)
                    else:
                        conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float32)

                    np_arr = torch.vstack((current_row, conditioning_data))
                    y_and_neighbors = np_arr[:, 2]

                    cov_yx = cov_matrix[0, 1:]

                    tmp2 = torch.matmul(torch.matmul(L_inv, locs).T, torch.matmul(L_inv, y_and_neighbors))
                    beta = torch.linalg.solve(tmp_for_beta, tmp2)

                    mu = torch.matmul(locs, beta)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]
                    
                    # Mean and variance of y|x
                    cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                    alpha = current_y - cond_mean
                    quad_form = alpha**2 * (1 / cov_ygivenx)
                    neg_log_lik += 0.5 * (log_det + quad_form)

                    continue

                if time_idx > 0:
                    last_hour_np = self.input_map[self.key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float32)

                np_arr = torch.vstack((current_row, conditioning_data))
                y_and_neighbors = np_arr[:, 2]
                locs = np_arr[:, :2]

                cov_matrix = covariance_function(params=params, y=np_arr, x=np_arr)
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                L = torch.linalg.cholesky(cov_matrix)
                L11 = L[:1, :1]
                L12 = torch.zeros(L[:1, 1:].shape)
                L21 = L[1:, :1]
                L22 = L[1:, 1:]
                L11_inv = torch.linalg.inv(L11)
                L22_inv = torch.linalg.inv(L22)

                L_inv = torch.block_diag(L11_inv, L22_inv)
                L_inv[1:, :1] = -torch.matmul(L22_inv, L21) @ L11_inv

                cov_yx = cov_matrix[0, 1:]

                tmp1 = torch.matmul(L_inv, locs)
                tmp2 = torch.matmul(torch.matmul(L_inv, locs).T, torch.matmul(L_inv, y_and_neighbors))
                tmp_for_beta = torch.matmul(tmp1.T, tmp1)
                beta = torch.linalg.solve(tmp_for_beta, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx.T, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx.T, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)

                if time_idx == 1:
                    self.cov_map[index] = {
                        'tmp_for_beta': tmp_for_beta,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
                        'L_inv': L_inv,
                        'cov_ygivenx': cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs': locs
                    }

        return neg_log_lik    


class likelihood_function(spatio_temporal_kernels):
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        super().__init__(smooth, input_map, nns_map, mm_cond_number)
        # Any additional initialization for dignosis class can go here

    def full_likelihood(self, params, input_np, y, covariance_function):
  
        # Compute the covariance matrix from the matern function
            # Convert DataFrame to NumPy array with float64 dtype
               
        input_arr = input_np[:,:4]
        y_arr = y
    
        cov_matrix = covariance_function(params=params, y = input_arr, x =input_arr)
        
        sign, log_det = np.linalg.slogdet(cov_matrix)

        # Compute the Cholesky decomposition
        # L = np.linalg.cholesky(cov_matrix)
        # Solve for the log determinant
        # log_det = 2 * np.sum(np.log(np.diagonal(L)))
        locs = input_arr[:,:2]
       
        tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
        tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y_arr))
        beta = np.linalg.solve(tmp1, tmp2)
      
        mu = np.dot(locs, beta)
        y_mu = y_arr - mu
    
        # alpha = np.linalg.solve(L, y_mu)
        # quad_form = np.dot(alpha.T, alpha)
        quad_form = np.dot( y_mu, np.linalg.solve(cov_matrix, y_mu))
        # Compute the negative log-likelihood
        n = len(y)
        # neg_log_lik = 0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)
        neg_log_lik = 0.5 * ( log_det + quad_form)
        return neg_log_lik
    
    def vecchia_like_using_cholesky(self, params, covariance_function):
        self.cov_map = defaultdict(list)
        neg_log_lik = 0
        
        for time_idx in range(self.number_of_timestamps):
            current_np = self.input_map[self.key_list[time_idx]]

            # use below when working on local computer to avoid singular matrix
            #cur_heads = current_np[:31,:]
            #neg_log_lik += self.full_likelihood(params,cur_heads, cur_heads[:,2],covariance_function)

            for index in range(0, self.size_per_hour):

                current_row = current_np[index]
      
                current_row = current_row.reshape(1,-1)
                current_y = current_row[0][2]

                # construct conditioning set
                
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    data_list.append( current_np[past])

                if time_idx > 1:
                    cov_matrix = self.cov_map[index]['cov_matrix']
                    tmp_for_beta = self.cov_map[index]['tmp_for_beta']
                    cov_xx_inv = self.cov_map[index]['cov_xx_inv']
                    L_inv = self.cov_map[index]['L_inv']
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx'] 
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs  = self.cov_map[index]['locs']
                   
                    last_hour_np = self.input_map[self.key_list[time_idx-1]]
                   
                    past_conditioning_data = last_hour_np[ (past+[index]),: ]
                    data_list.append( past_conditioning_data)

                    if data_list:
                        conditioning_data = np.vstack(data_list)
                    else:
                        conditioning_data = np.array([]).reshape(0, current_row.shape[1])

                    np_arr = np.vstack( (current_row, conditioning_data) )
                    y_and_neighbors = np_arr[:,2]
                    # locs = np_arr[:,:2]

                    cov_yx = cov_matrix[0,1:]

                    tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                    beta = np.linalg.solve(tmp_for_beta , tmp2)

                    mu = np.dot(locs, beta)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]
                    
                    # mean and variance of y|x
                         
                    cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                    alpha = current_y - cond_mean
                    quad_form = alpha**2 *(1/cov_ygivenx)
                   
                    # neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
                    neg_log_lik += 0.5 * ( log_det + quad_form)

                    continue

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
                L = np.linalg.cholesky(cov_matrix)
                L11 = L[:1,:1]
                L12 = np.zeros(L[:1,1:].shape)
                L21 = L[1:,:1]
                L22 = L[1:,1:]
                L11_inv = np.linalg.inv(L11)
                L22_inv = np.linalg.inv(L22)

                L_inv = np.block([
                    [L11_inv, L12],
                    [- np.dot( np.dot(L22_inv,L21), L11_inv), L22_inv]
                ])

                cov_yx = cov_matrix[0,1:]
                
                tmp1 = np.dot(L_inv,locs)
                tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                tmp_for_beta= np.dot(tmp1.T,tmp1)
                beta = np.linalg.solve(tmp_for_beta , tmp2)

                mu = np.dot(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # mean and variance of y|x
                sigma = cov_matrix[0][0]

                # cov_xx = np.dot(L21,L21.T) +np.dot(L22,L22.T) 
                cov_xx = cov_matrix[1:,1:]
                cov_xx_inv = np.linalg.inv(cov_xx)
                
                cov_ygivenx = sigma - np.dot(cov_yx.T, np.dot(cov_xx_inv, cov_yx))
               
                cond_mean_tmp = np.dot(cov_yx.T, cov_xx_inv)
                cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                alpha = current_y - cond_mean
                quad_form = alpha**2 *(1/cov_ygivenx)
                log_det = np.log(cov_ygivenx)
               
                # neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
                neg_log_lik += 0.5 * ( log_det + quad_form)

                if time_idx == 1:
                    self.cov_map[index] = {
                        'tmp_for_beta': tmp_for_beta,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
                        'L_inv':L_inv,
                        'cov_ygivenx':cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs':locs
                    }

        return neg_log_lik   


    def vecchia_like_local(self, params, covariance_function):
        self.cov_map = defaultdict(list)
        neg_log_lik = 0
        
        for time_idx in range(self.number_of_timestamps):
            current_np = self.input_map[self.key_list[time_idx]]

            # use below when working on local computer to avoid singular matrix
            cur_heads = current_np[:21,:]
            neg_log_lik += self.full_likelihood(params,cur_heads, cur_heads[:,2],covariance_function)

            for index in range(21, self.size_per_hour):

                current_row = current_np[index]
      
                current_row = current_row.reshape(1,-1)
                current_y = current_row[0][2]

                # construct conditioning set
                
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    data_list.append( current_np[past])

                if time_idx > 1:
                    cov_matrix = self.cov_map[index]['cov_matrix']
                    tmp_for_beta = self.cov_map[index]['tmp_for_beta']
                    cov_xx_inv = self.cov_map[index]['cov_xx_inv']
                    L_inv = self.cov_map[index]['L_inv']
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx'] 
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs  = self.cov_map[index]['locs']
                   
                    last_hour_np = self.input_map[self.key_list[time_idx-1]]
                   
                    past_conditioning_data = last_hour_np[ (past+[index]),: ]
                    data_list.append( past_conditioning_data)

                    if data_list:
                        conditioning_data = np.vstack(data_list)
                    else:
                        conditioning_data = np.array([]).reshape(0, current_row.shape[1])

                    np_arr = np.vstack( (current_row, conditioning_data) )
                    y_and_neighbors = np_arr[:,2]
                    # locs = np_arr[:,:2]

                    cov_yx = cov_matrix[0,1:]

                    tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                    beta = np.linalg.solve(tmp_for_beta , tmp2)

                    mu = np.dot(locs, beta)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]
                    
                    # mean and variance of y|x
                         
                    cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                    alpha = current_y - cond_mean
                    quad_form = alpha**2 *(1/cov_ygivenx)
                   
                    # neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
                    neg_log_lik += 0.5 * ( log_det + quad_form)

                    continue

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
               
                L = np.linalg.cholesky(cov_matrix)
                L11 = L[:1,:1]
                L12 = np.zeros(L[:1,1:].shape)
                L21 = L[1:,:1]
                L22 = L[1:,1:]
                L11_inv = np.linalg.inv(L11)
                L22_inv = np.linalg.inv(L22)

                L_inv = np.block([
                    [L11_inv, L12],
                    [- np.dot( np.dot(L22_inv,L21), L11_inv), L22_inv]
                ])

                cov_yx = cov_matrix[0,1:]
                
                tmp1 = np.dot(L_inv,locs)
                tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                tmp_for_beta= np.dot(tmp1.T,tmp1)
                beta = np.linalg.solve(tmp_for_beta , tmp2)

                mu = np.dot(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # mean and variance of y|x
                sigma = cov_matrix[0][0]

                # cov_xx = np.dot(L21,L21.T) +np.dot(L22,L22.T) 
                cov_xx = cov_matrix[1:,1:]
                cov_xx_inv = np.linalg.inv(cov_xx)
                
                cov_ygivenx = sigma - np.dot(cov_yx.T, np.dot(cov_xx_inv, cov_yx))
               
                cond_mean_tmp = np.dot(cov_yx.T, cov_xx_inv)
                cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                alpha = current_y - cond_mean
                quad_form = alpha**2 *(1/cov_ygivenx)
                log_det = np.log(cov_ygivenx)
               
                # neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
                neg_log_lik += 0.5 * ( log_det + quad_form)

                if time_idx == 1:
                    self.cov_map[index] = {
                        'tmp_for_beta': tmp_for_beta,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
                        'L_inv':L_inv,
                        'cov_ygivenx':cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs':locs
                    }
            
        return neg_log_lik   
    

    def full_likelihood_bayesian(self, params, input_np, y, covariance_function):
  
        # Compute the covariance matrix from the matern function
            # Convert DataFrame to NumPy array with float64 dtype
               
        
        input_arr = input_np[:,:4]
        y_arr = y
    
        cov_matrix = covariance_function(params=params, y = input_arr, x =input_arr)
        
        sign, log_det = np.linalg.slogdet(cov_matrix)

        # Compute the Cholesky decomposition
        # L = np.linalg.cholesky(cov_matrix)
        # Solve for the log determinant
        # log_det = 2 * np.sum(np.log(np.diagonal(L)))
        locs = input_arr[:,:2]
       
        tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
        tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y_arr))
        beta = np.linalg.solve(tmp1, tmp2)
      
        mu = np.dot(locs, beta)
        y_mu = y_arr - mu
    
        # alpha = np.linalg.solve(L, y_mu)
        # quad_form = np.dot(alpha.T, alpha)
        quad_form = np.dot( y_mu, np.linalg.solve(cov_matrix, y_mu))
        # Compute the negative log-likelihood
        n = len(y)
        neg_log_lik = 0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)

        # np.random.seed(42)
        priors = [
            norm(loc=15, scale=25),  # Prior for parameter sigmasq
            uniform(loc= 0.001, scale= 30),  # Prior for parameter range_lat
            uniform(loc= 0.001, scale= 30),  # Prior for parameter range_lon
            norm(loc=0, scale=0.01),  # Prior for parameter advection
            norm(loc=0.001, scale= 1),   # Prior for parameter beta
            uniform(loc=0.0001, scale=8)   # Prior for parameter nugget
            
        ]
        idx_params = [0, 1,2, 3,4,5]
        # Add prior terms for the parameters
        prior_terms = 0
        for i in idx_params:
            prior = priors[i]
            param_value = params[i]
            logpdf_value = prior.logpdf(param_value)
            # print(f"Parameter {i}: {param_value}, logpdf: {logpdf_value}")
            if np.isnan(logpdf_value) or np.isinf(logpdf_value):
                raise ValueError(f"Invalid logpdf value for parameter {i}: {logpdf_value}")
            prior_terms += logpdf_value
        
        neg_log_lik -= prior_terms
        # neg_log_lik = lr * neg_log_lik + (1-lr) * prior_terms            
        return neg_log_lik   
      

    def vecchia_like_using_cholesky_bayesian(self, params, covariance_function):
        self.cov_map = defaultdict(list)
        neg_log_lik = 0
        
        for time_idx in range(self.number_of_timestamps):
            current_np = self.input_map[self.key_list[time_idx]]

            # use below when working on local computer to avoid singular matrix
            #cur_heads = current_np[:31,:]
            #neg_log_lik += self.full_likelihood(params,cur_heads, cur_heads[:,2],covariance_function)

            for index in range(0, self.size_per_hour):

                current_row = current_np[index]
      
                current_row = current_row.reshape(1,-1)
                current_y = current_row[0][2]

                # construct conditioning set
                
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    data_list.append( current_np[past])

                if time_idx > 1:
                    cov_matrix = self.cov_map[index]['cov_matrix']
                    tmp_for_beta = self.cov_map[index]['tmp_for_beta']
                    cov_xx_inv = self.cov_map[index]['cov_xx_inv']
                    L_inv = self.cov_map[index]['L_inv']
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx'] 
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs  = self.cov_map[index]['locs']
                    

                    last_hour_np = self.input_map[self.key_list[time_idx-1]]
                   
                    past_conditioning_data = last_hour_np[ (past+[index]),: ]
                    data_list.append( past_conditioning_data)

                    if data_list:
                        conditioning_data = np.vstack(data_list)
                    else:
                        conditioning_data = np.array([]).reshape(0, current_row.shape[1])

                    np_arr = np.vstack( (current_row, conditioning_data) )
                    y_and_neighbors = np_arr[:,2]
                    # locs = np_arr[:,:2]

                    cov_yx = cov_matrix[0,1:]

                    tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                    beta = np.linalg.solve(tmp_for_beta , tmp2)

                    mu = np.dot(locs, beta)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]
                    
                    # mean and variance of y|x
                         
                    cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                    alpha = current_y - cond_mean
                    quad_form = alpha**2 *(1/cov_ygivenx)
                   
                    neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
                  

                    continue

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
                L = np.linalg.cholesky(cov_matrix)
                L11 = L[:1,:1]
                L12 = np.zeros(L[:1,1:].shape)
                L21 = L[1:,:1]
                L22 = L[1:,1:]
                L11_inv = np.linalg.inv(L11)
                L22_inv = np.linalg.inv(L22)

                L_inv = np.block([
                    [L11_inv, L12],
                    [- np.dot( np.dot(L22_inv,L21), L11_inv), L22_inv]
                ])

                cov_yx = cov_matrix[0,1:]
                
                tmp1 = np.dot(L_inv,locs)
                tmp2 = np.dot( np.dot(L_inv, locs).T, np.dot(L_inv, y_and_neighbors))
                tmp_for_beta= np.dot(tmp1.T,tmp1)
                beta = np.linalg.solve(tmp_for_beta , tmp2)

                mu = np.dot(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # mean and variance of y|x
                sigma = cov_matrix[0][0]

                # cov_xx = np.dot(L21,L21.T) +np.dot(L22,L22.T) 
                cov_xx = cov_matrix[1:,1:]
                cov_xx_inv = np.linalg.inv(cov_xx)
                
                cov_ygivenx = sigma - np.dot(cov_yx.T, np.dot(cov_xx_inv, cov_yx))
               
                cond_mean_tmp = np.dot(cov_yx.T, cov_xx_inv)
                cond_mean = mu_current + np.dot(cond_mean_tmp, (y_and_neighbors[1:]-mu_neighbors) )  # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            
                alpha = current_y - cond_mean
                quad_form = alpha**2 *(1/cov_ygivenx)
                log_det = np.log(cov_ygivenx)
               
                neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)

                
                if time_idx == 1:
                    self.cov_map[index] = {
                        'tmp_for_beta': tmp_for_beta,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
                        'L_inv':L_inv,
                        'cov_ygivenx':cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs':locs
                    }
        # np.random.seed(42)
        priors = [
            norm(loc=15, scale=25),  # Prior for parameter sigmasq
            uniform(loc= 0.001, scale= 30),  # Prior for parameter range_lat
            uniform(loc= 0.001, scale= 30),  # Prior for parameter range_lon
            norm(loc=0, scale=0.01),  # Prior for parameter advection
            norm(loc=0.001, scale= 1),   # Prior for parameter beta
            uniform(loc=0.0001, scale=8)   # Prior for parameter nugget
            
        ]
        idx_params = [0, 1,2, 3,4,5]
        # Add prior terms for the parameters
        prior_terms = 0
        for i in idx_params:
            prior = priors[i]
            param_value = params[i]
            logpdf_value = prior.logpdf(param_value)
            # print(f"Parameter {i}: {param_value}, logpdf: {logpdf_value}")
            if np.isnan(logpdf_value) or np.isinf(logpdf_value):
                raise ValueError(f"Invalid logpdf value for parameter {i}: {logpdf_value}")
            prior_terms += logpdf_value

        # Combine the negative log-likelihood and prior terms with scaling factors
        lr = 0.85
        neg_log_lik -= prior_terms
        # neg_log_lik = lr * neg_log_lik + (1-lr) * prior_terms            
        return neg_log_lik   
    
class model_fitting(likelihood_function):
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        super().__init__(smooth, input_map, nns_map, mm_cond_number)
        # Any additional initialization for dignosis class can go here

    def mle_parallel_vecc(self, bounds, params,covariance_function, vecch_fun ):
        iteration_count = 0 
        def callback(xk):
            nonlocal iteration_count
            iteration_count += 1
        try:
            logging.info(f"fit_st_1_27")
            print(f"fit_st_1_27")  # Debugging line
        
            result = minimize(
                vecch_fun, 
                params, 
                args = (covariance_function),
                # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B',
                callback= callback,
                options={'maxiter': 60}
            )
            jitter = result.x
            logging.info(f"Estimated parameters : {jitter}, when cond {self.mm_cond_number}, bounds={bounds}, smooth={self.smooth}")
            logging.info(f"Total iterations: {iteration_count}")
            print(f"Total iterations: {iteration_count}")

            return f"Estimated parameters : {jitter}, when cond {self.mm_cond_number}, bounds={bounds}, smooth={self.smooth}"
        except Exception as e:
            error_message = f"Error occurred: {str(e)}"
            print(error_message)
            logging.error(error_message)

    def mle_parallel_full(self, bounds, params , input_np, y,covariance_function, full_fun):
        iteration_count = 0 
        def callback(xk):
            nonlocal iteration_count
            iteration_count += 1

        try:
            logging.info(f"fit_st_1_27")
            print(f"fit_st_1_27")  # Debugging line
        
            result = minimize(
                full_fun, 
                params, 
                args = (input_np, y,covariance_function),
                # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B',
                callback= callback
            )
            jitter = result.x
            logging.info(f"Estimated parameters : {jitter}, when cond {self.mm_cond_number}, bounds={bounds}, smooth={self.smooth}")
            logging.info(f"Total iterations: {iteration_count}")
            print(f"Total iterations: {iteration_count}")

            return f"Estimated parameters : {jitter}, when cond {self.mm_cond_number}, bounds={bounds}, smooth={self.smooth}"
        except Exception as e:
            error_message = f"Error occurred: {str(e)}"
            print(error_message)
            logging.error(error_message) 


class space_smooth_experiment:               #sigmasq range advec beta  nugget
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        self.smooth = smooth
        self.input_map = input_map
        self.key_list = sorted(input_map)
        self.number_of_timestamps = len(self.key_list)

        sample_df = input_map[self.key_list[0]]

        self.size_per_hour = len(sample_df)
        self.mm_cond_number = mm_cond_number
        nns_map = list(nns_map) # nns_map is ndarray this allows to have sub array of diffrent lengths
        for i in range(len(nns_map)):  
            # Select elements up to mm_cond_number and remove -1
            tmp = np.delete(nns_map[i][:self.mm_cond_number], np.where(nns_map[i][:self.mm_cond_number] == -1))
            if tmp.size>0:
                nns_map[i] = tmp
            else:
                nns_map[i] = []
        self.nns_map = nns_map

    # Custom distance function for cdist
    def custom_distance(self,u, v):
        d = np.dot(self.sqrt_range_mat, u[:2] - v[:2] ) # Distance between x1,x2 (2D)
        spatial_diff = np.linalg.norm(d)  # Distance between x1,x2 (2D)
        temporal_diff = np.abs(u[2] - v[2])           # Distance between y1 and y2
        return np.sqrt(spatial_diff**2 + temporal_diff**2)
    
    def matern_cov_yx(self, sigmasq, params, y_df, x_df) -> pd.DataFrame:
    
        range_lat, range_lon, advec, beta, nugget = params[1:]
             
        # Validate inputs
        if y_df is None or x_df is None:
            raise ValueError("Both y and x_df must be provided.")
        # Extract values
        x1 = x_df['Longitude'].values
        y1 = x_df['Latitude'].values
        t1 = x_df['Hours_elapsed'].values
 
        x2 = y_df['Longitude'].values
        y2 = y_df['Latitude'].values
        t2 = y_df['Hours_elapsed'].values

        spat_coord1 = np.stack((x1- advec*t1, y1 - advec*t1), axis=-1)
        spat_coord2 = np.stack((x2- advec*t2, y2 - advec*t2), axis=-1)

        coords1 = np.hstack ((spat_coord1, (beta * t1).reshape(-1,1) ))
        coords2 = np.hstack ((spat_coord2, (beta * t2).reshape(-1,1) ))

        sqrt_range_mat = np.diag([ 1/range_lon**0.5, 1/range_lat**0.5])
        self.sqrt_range_mat = sqrt_range_mat

        distance = cdist(coords1,coords2, metric = self.custom_distance)

        # Initialize the covariance matrix with zeros
        out = distance
        
        # Compute the covariance for non-zero distances
        non_zero_indices = distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                                    (distance[non_zero_indices] )**self.smooth *
                                    kv(self.smooth, distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget

        return pd.DataFrame(out)

    def vecchia_likelihood(self, sigmasq, params ,input_df, mm_cond_number, nns_map):

        # reordered_df['ColumnAmountO3'] = reordered_df['ColumnAmountO3']-np.mean(reordered_df['ColumnAmountO3'])
        neg_log_lik = 0
        ## likelihood for the first 30 observations
        # smallset = input_df.iloc[:31,:]
        # neg_log_lik += self.full_likelihood(params, smallset, smallset['ColumnAmountO3'])

        for i in range(0,len(input_df)):
            # current_data and conditioning data
            current_data = input_df.iloc[i:i+1,:]
            current_y = current_data['ColumnAmountO3'].values[0]
            mm_past = nns_map[i,:mm_cond_number]
            mm_past = mm_past[mm_past!=-1]
            # mm_past = np.arange(i)

            conditioning_data = input_df.loc[mm_past,: ]
            df = pd.concat( (current_data, conditioning_data), axis=0)
            y_and_neighbors = df['ColumnAmountO3'].values
            cov_matrix = self.matern_cov_yx(sigmasq, params, y_df= df, x_df=df)
 
            cov_xx = cov_matrix.iloc[1:,1:].reset_index(drop=True)
            cov_yx = cov_matrix.iloc[0,1:]

            # get mean
            locs = np.array(df[['Latitude','Longitude']])

            tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
            tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y_and_neighbors))
            beta = np.linalg.solve(tmp1, tmp2)

            mu = np.dot(locs, beta)
            mu_current = mu[0]
            mu_neighbors = mu[1:]
            
            # mean and variance of y|x
            sigma = cov_matrix.iloc[0,0]
            cov_ygivenx = sigma - np.dot(cov_yx.T,np.linalg.solve(cov_xx, cov_yx))
            cond_mean = mu_current + np.dot(cov_yx.T, np.linalg.solve( cov_xx, (y_and_neighbors[1:]-mu_neighbors) ))   # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            # print(f'cond_mean{mean_z}')

            alpha = current_y - cond_mean
            quad_form = alpha**2 *(1/cov_ygivenx)
            log_det = np.log(cov_ygivenx)
            # Compute the negative log-likelihood

            neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
        
        return neg_log_lik


    def mle_parallel(self, key, bounds, sigmasq, initial_params, input_df, mm_cond_number, nns_map):
        try:
            logging.info(f"fit_space_sigma, time: {key}")
            print(f"fit_space_sigma, time: {key}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                sigmasq, 
                args=(initial_params,input_df, mm_cond_number, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"Estimated sigma on {key}, is : {jitter}, when cond {mm_cond_number}, bounds={bounds}")
        
            return f"Estimated sigma on {key}, is : {jitter}, when cond {mm_cond_number},  bounds={bounds}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
        


