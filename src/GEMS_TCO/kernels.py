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
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from scipy.interpolate import splrep, splev

# Fit your "spline" by just storing the x and y
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from typing import Dict, Any, Callable, List, Tuple


import copy    
import logging     # for logging
# Add your custom path

import warnings

# --- MODIFIED 'optimizer_fun' IN 'model_fitting' CLASS ---
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # Need to import this
# ... (other imports) ...

sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

class spatio_temporal_kernels:               #sigmasq range advec beta  nugget
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
        # self.smooth = torch.tensor(smooth,dtype=torch.float64 )

        # Store device for creating new tensors
        self.device = aggregated_data.device

        self.smooth = smooth
        # Store smooth as a tensor on the correct device
        self.smooth_tensor = torch.tensor(self.smooth, dtype=torch.float64, device=self.device)
        # Pre-compute the Matern constant on the correct device
        gamma_val = torch.tensor(gamma(self.smooth), dtype=torch.float64, device=self.device)
        # Pre-compute the Matern constant on the correct device
        self.matern_const = ( (2**(1-self.smooth)) / gamma_val )

        self.smooth = smooth

        self.input_map = input_map
        self.aggregated_data = aggregated_data[:,:4]
        self.aggregated_response = aggregated_data[:,2]
        self.aggregated_locs = aggregated_data[:,:2]

        self.key_list = list(input_map.keys())
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
    
    ## The torch.sqrt() is moved to the covariance function to track gradients of beta and avec
    def custom_distance_matrix(self, U:torch.Tensor, V:torch.Tensor):
        # Efficient distance computation with broadcasting
        spatial_diff = torch.norm(U[:, :2].unsqueeze(1) - V[:, :2].unsqueeze(0), dim=2)
        temporal_diff = torch.abs(U[:, 2].unsqueeze(1) - V[:, 2].unsqueeze(0))
        distance = (spatial_diff**2 + temporal_diff**2)  # move torch.sqrt to covariance function to track gradients of beta and avec
        return distance
    
    def precompute_coords_anisotropy(self, params:torch.Tensor, y: torch.Tensor, x: torch.Tensor)-> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params

        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")
        x1, y1, t1 = x[:, 0], x[:, 1], x[:, 3]
        x2, y2, t2 = y[:, 0], y[:, 1], y[:, 3]
        spat_coord1 = torch.stack(( (x1 - advec_lat * t1)/range_lat, (y1 - advec_lon * t1)/range_lon ), dim=-1)
        spat_coord2 = torch.stack(( (x2 - advec_lat * t2)/range_lat, (y2 - advec_lon * t2)/range_lon ), dim=-1)
        U = torch.cat((spat_coord1, (beta * t1).reshape(-1, 1)), dim=1)
        V = torch.cat((spat_coord2, (beta * t2).reshape(-1, 1)), dim=1)
        distance = self.custom_distance_matrix(U,V)
        return distance # This is the optimized version, fixing the redundancy
    


        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")

        x1, y1, t1 = x[:, 0], x[:, 1], x[:, 3]
        x2, y2, t2 = y[:, 0], y[:, 1], y[:, 3]

        # spat_coord1 = torch.stack((self.x1 , self.y1 - advec * self.t1), dim=-1)
        spat_coord1 = torch.stack(( (x1 - advec_lat * t1)/range_lat, (y1 - advec_lon * t1)/range_lon ), dim=-1)
        spat_coord2 = torch.stack(( (x2 - advec_lat * t2)/range_lat, (y2 - advec_lon * t2)/range_lon ), dim=-1)

        U = torch.cat((spat_coord1, (beta * t1).reshape(-1, 1)), dim=1)
        V = torch.cat((spat_coord2, (beta * t2).reshape(-1, 1)), dim=1)

        distance = self.custom_distance_matrix(U,V)
        non_zero_indices = distance != 0
        return distance, non_zero_indices

 
    def matern_cov_anisotropy_v05(self,params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params
        
        distance = self.precompute_coords_anisotropy(params, x,y)

        out = torch.zeros_like(distance)

        non_zero_indices = distance != 0
        if torch.any(non_zero_indices):
            out[non_zero_indices] = sigmasq * torch.exp(- torch.sqrt(distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += torch.eye(out.shape[0], dtype=torch.float64) * nugget 
        return out

    def matern_cov_anisotropy_v15(self,params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params
        
        distance = self.precompute_coords_anisotropy(params, x,y)

        out = torch.zeros_like(distance)

        non_zero_indices = distance != 0
        if torch.any(non_zero_indices):
            out[non_zero_indices] = sigmasq * (1+ torch.sqrt(distance[non_zero_indices])) * torch.exp(- torch.sqrt(distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += torch.eye(out.shape[0], dtype=torch.float64) * nugget 
        return out
    def matern_cov_anisotropy_kv(self,params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params

        distance, non_zero_indices = self.precompute_coords_anisotropy(params, x,y)
        out = torch.zeros_like(distance, dtype= torch.float64)
        
        # Compute the covariance for non-zero distances
        non_zero_indices = distance != 0
        if torch.any(non_zero_indices):
            tmp = torch.tensor( kv(self.smooth, np.sqrt(distance[non_zero_indices].detach().numpy())), dtype=torch.float64)
            out[non_zero_indices] = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                                    (torch.sqrt(distance[non_zero_indices]) )**self.smooth *
                                    tmp)
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += torch.eye(out.shape[0], dtype=torch.float64) * nugget
        return out


    def matern_cov_anisotropy_spline(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params

        # Precompute anisotropic distances
        distance, non_zero_indices = self.precompute_coords_anisotropy(params, x, y)

        coarse_factor = 4  # Increase coarse_factor to reduce the number of points
        flat_distances = distance.flatten()

        x2 = x[::(coarse_factor)]
        y2 = y[::(coarse_factor)]
        print( x.shape, y.shape)
        print(f' x2 shape {x2.shape} y2 shape {y2.shape}')

        # Generate coarse distances for spline fitting
        fit_distances = torch.linspace(0.001, distance.max(), len(flat_distances) // coarse_factor**2)

        print(len(fit_distances))
        # Convert fit_distances to NumPy array for spline fitting
        fit_distances_np = fit_distances.detach().numpy()

        # Compute exact values using coarse distances
        exact_values_np = self.matern_cov_anisotropy_kv(params, x2, y2).detach().numpy()
        exact_values_np = exact_values_np.flatten()
        print(exact_values_np.shape)
        # Sort fit_distances_np and reorder exact_values_np to match
        sorted_indices = np.argsort(fit_distances_np)
        fit_distances_np = fit_distances_np[sorted_indices]
        exact_values_np = exact_values_np[sorted_indices]

        # Fit a cubic spline to the exact values
        spline_params = splrep(fit_distances_np, exact_values_np, k=3)

        # Evaluate spline at dense distances
        flat_distances_np = flat_distances.detach().numpy()
        spline_values_np = splev(flat_distances_np, spline_params)

        # Convert spline values back to PyTorch tensor and reshape
        spline_values = torch.tensor(spline_values_np, dtype=torch.float64, requires_grad=True).reshape(distance.shape)

        return spline_values


class likelihood_function(spatio_temporal_kernels):
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map: Dict[str, Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.nheads = nheads

    '''         
=======
              
>>>>>>> 98fc71c474ddced6792e89e9ab27c07529da5b48
=======
              
>>>>>>> 0a418ac421c02a3cd32b6e4c97b2bdc92cdb79b7
    def full_likelihood(self,params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
   
        cov_matrix = covariance_function(params=params, y= input_data, x= input_data)
        sign, log_det = torch.slogdet(cov_matrix)
        # if sign <= 0:
        #     raise ValueError("Covariance matrix is not positive definite")

        locs = input_data[:,:2]
        response = y
        # Compute beta
        tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
        tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, response))
        beta = torch.linalg.solve(tmp1, tmp2)

        # Compute the mean
        mu = torch.matmul(locs, beta)
        y_mu = response - mu

        # Compute the quadratic form
        quad_form = torch.matmul(y_mu, torch.linalg.solve(cov_matrix, y_mu))

        # Compute the negative log likelihood
        neg_log_lik = 0.5 * (log_det + quad_form)
        return  neg_log_lik 
<<<<<<< HEAD
<<<<<<< HEAD
    ''' 

    def full_likelihood(self,params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
        """
        Optimized likelihood function using Cholesky decomposition.
        """
        cov_matrix = covariance_function(params=params, y= input_data, x= input_data)
        
        # --- OPTIMIZATION ---
        # 1. Perform Cholesky decomposition ONCE. This is O(N^3)
        try:
            L = torch.linalg.cholesky(cov_matrix)
        except torch.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed. Matrix may not be positive definite.")
            # Fallback to original (slower) method to get a proper error or inf
            return self.full_likelihood_original(params, input_data, y, covariance_function)

        # 2. Get log-determinant from Cholesky factor (fast and stable)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        locs = input_data[:,:2] # (N, 2)
        
        # Ensure y is a column vector (N, 1) for matrix operations
        if y.dim() == 1:
            y_col = y.unsqueeze(-1)
        else:
            y_col = y

        # 3. Solve for C_inv_X and C_inv_y using efficient O(N^2) triangular solves
        C_inv_X = torch.cholesky_solve(locs, L, upper=False)  # Solves C*z = X
        C_inv_y = torch.cholesky_solve(y_col, L, upper=False) # Solves C*z = y

        # 4. Compute beta
        tmp1 = torch.matmul(locs.T, C_inv_X)       # (2, N) @ (N, 2) = (2, 2)
        tmp2 = torch.matmul(locs.T, C_inv_y)       # (2, N) @ (N, 1) = (2, 1)
        
        try:
            beta = torch.linalg.solve(tmp1, tmp2)      # Solves small (2, 2) system
        except torch.linalg.LinAlgError:
            print("Warning: Could not solve for beta. X^T C_inv X may be singular.")
            return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

        # 5. Compute the mean
        mu = torch.matmul(locs, beta)              # (N, 2) @ (2, 1) = (N, 1)
        y_mu = y_col - mu                          # (N, 1)

        # 6. Compute the quadratic form using another efficient O(N^2) solve
        C_inv_y_mu = torch.cholesky_solve(y_mu, L, upper=False)
        quad_form = torch.matmul(y_mu.T, C_inv_y_mu) # (1, N) @ (N, 1) = (1, 1)

        # 7. Compute the negative log likelihood
        neg_log_lik = 0.5 * (log_det + quad_form.squeeze())
        return  neg_log_lik 
    

class spline(spatio_temporal_kernels):
    '''
    fit_cublic_spline() for each data shares the common locations. Even though the
    'distances' matrix is a function of parameters, we can make a common upper bound
    by putting range parameters 0.5, advections 0, beta 2.
    and we fit cubic_spline() for fixed smooth Matern model with range=1 and sigmasq=1.
    Essentially, we are approximating simple Matern model for v=1.
    
    Any change in parameters will be reflected through "distances" matrix. So,
    we define "distances" matrix for each epoch.
    
    '''
    def __init__(self, epsilon:float, coarse_factor:int, nheads:int, smooth:float, input_map: Dict[str, Any], aggregated_data:torch.Tensor, nns_map: np.ndarray, mm_cond_number:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.smooth = torch.tensor(smooth, dtype= torch.float64)
        
        self.epsilon = epsilon  # starting point for the spline fitting
        sample_params = [25, 0.5, 0.5, 0, 0, 2, 5] # just random nuumber to initialize spline
        sample_params = torch.tensor(sample_params, dtype=torch.float64, requires_grad=True)
        
        self.coarse_factor = coarse_factor
        self.nheads = nheads 

        """
        Initialize the class with given parameters.
        Args:
            coarse_factor (int): Factor used for coarse-graining.
            smooth (float): Smooth parameter in Matern model.
            input_map (Dict[str, Any]): Dictionary containing input mappings.
            aggregated_data (torch.Tensor): Tensor containing aggregated data.
            nns_map (Dict[str, Any]): 2-d nd.array containing nearest neighbors mappings.
            mm_cond_number (int): Condition number for Vecchia approximation
        """


    def fit_cubic_spline(self, target_distances, coarse_factor:int=4):

        """
        Fit a natural cubic spline coefficients.

        Args:
            params (tuple): Parameters for the spline fitting.

        Returns:
            NaturalCubicSpline: The fitted spline object with coefficients.
        """

        def flat_distance_matrix(distances: torch.Tensor) -> torch.Tensor:
            n = distances.size(0)
            indices = torch.triu_indices(n, n, offset=1)
            upper_tri = distances[indices[0], indices[1]]
            unique_sorted = torch.unique(upper_tri, sorted=True)
            flat_distances = torch.cat([torch.tensor([0.0], device=unique_sorted.device), unique_sorted])
            max_distance = torch.max(flat_distances).clone().detach()
            len_distance_arr = len(flat_distances)
            
            return max_distance, len_distance_arr
        
        max_distance, len_distance_arr = flat_distance_matrix(target_distances)

        # fit_distances should be 1 d array to be used in natural_cubic_spline_coeffs
        fit_distances = torch.linspace(0, max_distance*1.5 + 1e-6 , len_distance_arr//coarse_factor)
        non_zero_indices = fit_distances != 0
        out = torch.zeros_like(fit_distances, dtype= torch.float64)
        # print(max_distance)
        if torch.any(non_zero_indices):
            tmp = kv(self.smooth, torch.sqrt(fit_distances[non_zero_indices])).double().clone()
            out[non_zero_indices] = (1 * (2**(1-self.smooth)) / gamma(self.smooth) *
                                    (torch.sqrt(fit_distances[non_zero_indices]) ) ** self.smooth *
                                    tmp)
        out[~non_zero_indices] = 1

        # Compute spline coefficients. If input is tensor, so is output.
        # natural_cubic_spline_coeffs(t,x), t should be 1-d array (n,) and x should be (n,channels)
        # where channels reoresent number of features. out.unsquueze(1) makes (n,1).
        coeffs = natural_cubic_spline_coeffs(fit_distances, out.unsqueeze(1))
        # Create spline object
        spline = NaturalCubicSpline(coeffs)
        return spline


    def interpolate_cubic_spline(self, params:torch.Tensor, target_distances:torch.Tensor, spline_object) -> torch.Tensor:

        """
        Interpolate using the fitted cubic spline.
        Args:
            params (tuple): Parameters for the interpolation.
            target_distances (torch.Tensor): Distances to interpolate.
            spline_object (NaturalCubicSpline): The fitted spline object.

        Returns:
            torch.Tensor: Interpolated values.
        """
    
        sigmasq, _, _, _, _, _, nugget = params
        n = target_distances.size(0)
        indices = torch.triu_indices(n, n, offset=0)  # offset=0 to include diagonal

        # Evaluate spline only on upper triangle
        cov_upper = spline_object.evaluate(target_distances[indices[0], indices[1]])

        # Create empty matrix and fill upper triangle
        cov_matrix = torch.zeros_like(target_distances)

        # spline_object.evaluate return [N,1] 
        #print(cov_matrix.shape, cov_upper.shape, indices.shape, indices[0].shape)
        #print(indices)
        
        cov_matrix[indices[0], indices[1]] = cov_upper.view(-1)

        # Mirror to lower triangle
        cov_matrix = cov_matrix + cov_matrix.T - torch.diag(torch.diag(cov_matrix))

        # Apply scaling and nugget
        cov_matrix = cov_matrix * sigmasq
        cov_matrix = cov_matrix + torch.eye(n, dtype=torch.float64, device=cov_matrix.device) * nugget

        ''' 
        Before May26
        sigmasq, _, _, _, _, _, nugget = params
        cov_1d = spline_object.evaluate(target_distances)
        cov_matrix = cov_1d.reshape(target_distances.shape)
        cov_matrix = cov_matrix * sigmasq
        cov_matrix = cov_matrix + torch.eye(cov_matrix.shape[0], dtype=torch.float64) * nugget 
        '''
        return cov_matrix


    def full_likelihood_using_spline(self, params:torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, target_distances:torch.Tensor, spline_object):
    
        cov_matrix = self.interpolate_cubic_spline(params, target_distances, spline_object)

        sign, log_det = torch.slogdet(cov_matrix)

        # if sign <= 0:
        #     raise ValueError("Covariance matrix is not positive definite")
        # Compute beta

        locs = input_data[:,:2]
        response = input_data[:,2]

        tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
        tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, response))
        beta = torch.linalg.solve(tmp1, tmp2)

        mu = torch.matmul(locs, beta)
        y_mu = response - mu
        quad_form = torch.matmul(y_mu, torch.linalg.solve(cov_matrix, y_mu))
        neg_log_lik = 0.5 * (log_det + quad_form)
        return  neg_log_lik


    def cov_structure_saver_using_spline(self, params: torch.Tensor) -> None:
        
        cov_map = defaultdict(lambda: defaultdict(dict))
        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        for time_idx in range(0,3):
            current_array = self.input_map[key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            for index in range(cut_line, self.size_per_hour):
                current_row = current_array[index].reshape(1, -1)
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors) 
                data_list = []

                if past:
                    data_list.append(current_array[past])

                if time_idx > 0:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    data_list.append(one_hour_lag[past + [index], :])

                if time_idx > 1:
                    two_hour_lag = self.input_map[key_list[time_idx -2]]
                    data_list.append(two_hour_lag [past + [index], :])
                
                conditioning_data = torch.vstack(data_list) if data_list else torch.empty((0, current_row.shape[1]), dtype=torch.float64)
                aggregated_arr = torch.vstack((current_row, conditioning_data))
                locs = aggregated_arr[:, :2]

                target_distances_for_cond, non_zero_indices = self.precompute_coords_anisotropy(params, aggregated_arr,aggregated_arr)

                # cond_spline_object = self.fit_cubic_spline(target_distances_for_cond, self.coarse_factor_cond )  # change here  
                cov_matrix = self.interpolate_cubic_spline(params, target_distances_for_cond, self.spline_object)

                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")

                cov_yx = cov_matrix[0, 1:]
                sign, log_det = torch.slogdet(cov_matrix)
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
            
                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)
                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                log_det = torch.log(cov_ygivenx)
            
                cov_map[(time_idx,index)] = {
                    'tmp1': tmp1.clone().detach(),
                    'cov_xx_inv': cov_xx_inv.clone().detach(),
                    'cov_matrix': cov_matrix.clone().detach(),
                    'cov_ygivenx': cov_ygivenx.clone().detach(),
                    'cond_mean_tmp': cond_mean_tmp.clone().detach(),
                    'log_det': log_det.clone().detach(),
                    'locs': locs.clone().detach()
                }
        return cov_map

    def vecchia_nll_using_spline(self, params: torch.Tensor, cov_map:Dict[str,Any]) -> torch.Tensor:

        cut_line= self.nheads
        key_list = list(self.input_map.keys())
        neg_log_lik = 0.0
        heads = self.input_map[key_list[0]][:cut_line,:]

        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[key_list[time_idx]][:cut_line,:]
            heads = torch.cat( (heads,tmp), dim=0)

        
        distances_heads, _ = self.precompute_coords_anisotropy(params, heads, heads)
        #spline_object_head = self.fit_cubic_spline( distances_heads, self.coarse_factor_head)  # change here
        neg_log_lik += self.full_likelihood_using_spline(params, heads[:,:4], heads[:,2], distances_heads, self.spline_object)
    
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[key_list[time_idx]]

            for index in range(cut_line, self.size_per_hour):
                current_row = current_np[index].reshape(1, -1)
                current_y = current_row[0, 2]

                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors) 
                data_list = []

                if past:
                    data_list.append(current_np[past])  
                if time_idx < 2:
                    cov_matrix = cov_map[(time_idx,index)]['cov_matrix']
                    tmp1 = cov_map[(time_idx,index)]['tmp1']
                    cov_xx_inv = cov_map[(time_idx,index)]['cov_xx_inv']
                    cov_ygivenx = cov_map[(time_idx,index)]['cov_ygivenx']
                    cond_mean_tmp = cov_map[(time_idx,index)]['cond_mean_tmp']
                    log_det = cov_map[(time_idx,index)]['log_det']
                    locs = cov_map[(time_idx,index)]['locs']
                else:
                    cov_matrix = cov_map[(2,index)]['cov_matrix']
                    tmp1 = cov_map[(2,index)]['tmp1']
                    cov_xx_inv = cov_map[(2,index)]['cov_xx_inv']
                    cov_ygivenx = cov_map[(2,index)]['cov_ygivenx']
                    cond_mean_tmp = cov_map[(2,index)]['cond_mean_tmp']
                    log_det = cov_map[(2,index)]['log_det']
                    locs = cov_map[(2,index)]['locs']

                if time_idx >= 1:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = one_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if time_idx > 1:
                    two_hour_lag = self.input_map[key_list[time_idx - 2]]
                    past_conditioning_data = two_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)
    
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                aggregated_arr = torch.vstack((current_row, conditioning_data))
                aggregated_y = aggregated_arr[:, 2]

                cov_yx = cov_matrix[0, 1:]
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, aggregated_y))
                beta = torch.linalg.solve(tmp1, tmp2)
                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # Mean and variance of y|x
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (aggregated_y[1:] - mu_neighbors))
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)

        return neg_log_lik
    
    def compute_full_nll(self, params:torch.Tensor, aggregated_data, distances:torch.Tensor, spline_object): 
        nll = self.full_likelihood_using_spline( params, aggregated_data[:,:4], aggregated_data[:,2], distances, spline_object)
        return nll

    def compute_vecchia_nll(self, params:torch.Tensor): 
        cov_map = self.cov_structure_saver_using_spline(params)
        nll = self.vecchia_nll_using_spline(params, cov_map)
        return nll
    
    def optimizer_fun(self, params:torch.Tensor, lr:float =0.01, betas: tuple=(0.9, 0.8), eps:float=1e-8, step_size:int=40, gamma:float=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler

    def run_full(self, params:torch.Tensor, aggregated_data, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, epochs:int=10 ):

        """
        Run the training loop for the full likelihood model.

        Args:
            params (torch.Tensor): Model parameters.
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            epochs (int): Number of epochs to train.

        Returns:
            list: Final parameters and loss.
            int: Number of epochs run.
        """

        prev_loss= float('inf')
        # 1e-3: Faster convergence, slightly lower accuracy than 1e-4
        tol = 1e-3  # Convergence tolerance

        distances, _ = self.precompute_coords_anisotropy(params, aggregated_data[:,:4], aggregated_data[:,:4])
        spline_object = self.fit_cubic_spline( distances, self.coarse_factor)  # change here
        
        for epoch in range(epochs):  
            optimizer.zero_grad()  # Zero the gradients 
            
            distances, _ = self.precompute_coords_anisotropy(params, aggregated_data[:,:4], aggregated_data[:,:4])
            #spline_object = self.fit_cubic_spline( distances, self.coarse_factor)  # change here

            loss = self.compute_full_nll(params, aggregated_data, distances, spline_object)
            loss.backward()  # Backpropagate the loss

            # Gradient and Parameter Logging for every 10th epoch
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Convergence Check
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n full Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        params = [torch.round(x*1000).detach().numpy()/1000 for x in params]
        loss = (torch.round(loss*1000)/1000).item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n full Parameters: {params}')
        return params + [loss], epoch

    def fit_vecchia(self, params:torch.Tensor, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, epochs:int=10 ):

        """
        Run the training loop for the full likelihood model.

        Args:
            params (torch.Tensor): Model parameters.
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            epochs (int): Number of epochs to train.

        Returns:
            list: Final p arameters and loss.
            int: Number of epochs run.
        """

        prev_loss= float('inf')
        # 1e-3: Faster convergence, slightly lower accuracy than 1e-4
        tol = 1e-3  # Convergence tolerance

        distances, _ = self.precompute_coords_anisotropy(params, self.aggregated_data[:,:4], self.aggregated_data[:,:4])
        self.spline_object = self.fit_cubic_spline( distances, self.coarse_factor)  # change here
        
        for epoch in range(epochs):  
            optimizer.zero_grad()  # Zero the gradients 
            # distance is a function of parameters
            # distances, non_zero_indices = self.precompute_coords_anisotropy(params, self.new_aggregated_data[:,:4], self.new_aggregated_data[:,:4])
            
            distances, _ = self.precompute_coords_anisotropy(params, self.aggregated_data[:,:4], self.aggregated_data[:,:4])
            #self.spline_object = copy.deepcopy(self.spline_object_base)

            loss = self.compute_vecchia_nll(params)
            loss.backward()  # Backpropagate the loss

            # Gradient and Parameter Logging for every 10th epoch
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Convergence Check
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        params = [torch.round(x*1000).detach().numpy()/1000 for x in params]
        loss = (torch.round(loss*1000)/1000).item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {params}')
        return params + [loss], epoch

############################################################

class vecchia_experiment(likelihood_function):
    def __init__(self, smooth:float, input_map :Dict[str,Any], aggregated_data: torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     

    def vecc_ghg_statistic(self,full_params, full_ll, vecc_ll, cov_function, aggregated_data, aggregated_y,):

        # Define the function to compute the loss
        def compute_loss_full(params):
            return full_ll(params, aggregated_data , aggregated_y, cov_function)

        def compute_loss_vecc(params):
            return vecc_ll(params, cov_function)
          
        grad_function = torch.func.grad(compute_loss_vecc)
        gradient = grad_function(full_params)

        # print(f'Gradient: {g1}')
        # Compute the Hessian matrix using torch.func.hessian
        try:
            hessian_matrix =  torch.func.hessian(compute_loss_full)(full_params)
            cond_number = torch.linalg.cond(hessian_matrix)
            # print(f'cond_number of hessian {cond_number}')

        except Exception as e:
            print(f'Error computing Hessian: {e}')

        statistic  = torch.matmul(gradient, torch.linalg.solve(hessian_matrix, gradient))
        # print(f' statistic is {statistic}')
        return statistic

    '''  
=======

>>>>>>> 98fc71c474ddced6792e89e9ab27c07529da5b48
=======

>>>>>>> 0a418ac421c02a3cd32b6e4c97b2bdc92cdb79b7
    def full_ghg_statistic(self,full_params, full_ll, cov_function, aggregated_data, aggregated_y,):

        # Define the function to compute the loss
        def compute_loss_full(params):
            return full_ll(params, aggregated_data , aggregated_y, cov_function)

        grad_function = torch.func.grad(compute_loss_full)
        gradient = grad_function(full_params)

        # print(f'Gradient: {g1}')

        # Compute the Hessian matrix using torch.func.hessian
        try:
            hessian_matrix =  torch.func.hessian(compute_loss_full)(full_params)
            cond_number = torch.linalg.cond(hessian_matrix)
            print(f'cond_number of hessian {cond_number}')
            
        except Exception as e:
            print(f'Error computing Hessian: {e}')

        return hessian_matrix, cond_number
<<<<<<< HEAD
<<<<<<< HEAD
        
        statistic  = torch.matmul(gradient, torch.linalg.solve(hessian_matrix, gradient))
        # print(f' statistic is {statistic}')
        return statistic
        
    '''


    def vecchia_grouping(self, params: torch.Tensor, covariance_function: Callable, cov_map:Dict[str,Any]) -> torch.Tensor:

        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        neg_log_lik = 0.0
        heads = self.input_map[key_list[0]][:cut_line,:]
        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[key_list[time_idx]][:cut_line,:]
            heads = torch.cat( (heads,tmp), dim=0)

        
        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)          
        
        st = 10
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            
            for index in range(cut_line, self.size_per_hour, st):
                current_row = current_np[index: (index+st),:]
                current_y = current_row[:, 2]
                
                # Construct conditioning set
                past = []
                for index in range(index,index+st):
                    mm_neighbors = self.nns_map[index]
                    past += list(mm_neighbors) 
                past = list(set(past))

                #print(len(past))
                # mm_neighbors = self.nns_map[index]
                # past = list(mm_neighbors) 
                data_list = []

                if past:
                    data_list.append(current_np[past])  

                tmp = list(range(index, index+st ))
                combined_indices = past + tmp

                if time_idx > 0:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    data_list.append(one_hour_lag[past, :])
                 
                #if time_idx > 1:
                #    two_hour_lag = self.input_map[key_list[time_idx - 2]]
                #    data_list.append(two_hour_lag[past, :])
                 
                conditioning_data = torch.vstack(data_list) if data_list else torch.empty((0, current_row.shape[1]), dtype=torch.float64)
                aggregated_arr = torch.vstack((current_row, conditioning_data))
                aggregated_y = aggregated_arr[:, 2]
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr)

                sign, log_det = torch.slogdet(cov_matrix)

                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, aggregated_y))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[:st]
                mu_neighbors = mu[st:]
            
                # Mean and variance of y|x
                cov_yx = cov_matrix[:st, st:]
                # cov_yx = cov_matrix[0, 1:]
        
                cov_yy = cov_matrix[:st, :st]
                cov_xx = cov_matrix[st:, st:]
                cov_xx_inv = torch.linalg.inv(cov_xx)
          
                cov_ygivenx = cov_yy - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx.T))
           
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)

                # Mean and variance of y|x
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (aggregated_y[st:] - mu_neighbors))
                alpha = current_y - cond_mean
            

                quad_form = torch.matmul( alpha, torch.linalg.solve(cov_ygivenx, alpha )  )
                
                if st==1:
                    log_det = torch.log(cov_ygivenx[0]).item()
                else: 
                    _, log_det = torch.slogdet( cov_ygivenx)
                    
                neg_log_lik += 0.5 * (log_det + quad_form)
           
        return neg_log_lik

    '''  
=======

>>>>>>> 98fc71c474ddced6792e89e9ab27c07529da5b48
=======

>>>>>>> 0a418ac421c02a3cd32b6e4c97b2bdc92cdb79b7
    def cov_structure_saver(self, params: torch.Tensor, covariance_function: Callable) -> None:
        
        cov_map = defaultdict(lambda: defaultdict(dict))
        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        for time_idx in range(0,3):
            current_np = self.input_map[key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            for index in range(cut_line, self.size_per_hour):
                current_row = current_np[index].reshape(1, -1)
             
                # Construct conditioning set
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors) 
                data_list = []

                if past:
                    data_list.append(current_np[past])

                if time_idx > 0:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    data_list.append(one_hour_lag[past + [index], :])

                if time_idx > 1:
                    two_hour_lag = self.input_map[key_list[time_idx -2]]
                    data_list.append(two_hour_lag [past + [index], :])

    
                conditioning_data = torch.vstack(data_list) if data_list else torch.empty((0, current_row.shape[1]), dtype=torch.float64)
                aggregated_arr = torch.vstack((current_row, conditioning_data))
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                
                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")

                cov_yx = cov_matrix[0, 1:]
                sign, log_det = torch.slogdet(cov_matrix)
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
            
                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)
          
                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                log_det = torch.log(cov_ygivenx)
            
                cov_map[(time_idx,index)] = {
                    'tmp1': tmp1.clone().detach(),
                    'cov_xx_inv': cov_xx_inv.clone().detach(),
                    'cov_matrix': cov_matrix.clone().detach(),
                    'cov_ygivenx': cov_ygivenx.clone().detach(),
                    'cond_mean_tmp': cond_mean_tmp.clone().detach(),
                    'log_det': log_det.clone().detach(),
                    'locs': locs.clone().detach()
                }
        return cov_map

    def vecchia_may9(self, params: torch.Tensor, covariance_function: Callable, cov_map:Dict[str,Any]) -> torch.Tensor:

        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        neg_log_lik = 0.0
        heads = self.input_map[key_list[0]][:cut_line,:]

        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[key_list[time_idx]][:cut_line,:]
            heads = torch.cat( (heads,tmp), dim=0)

        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)          
        
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            for index in range(cut_line, self.size_per_hour):
                current_row = current_np[index].reshape(1, -1)
                current_y = current_row[0, 2]

                # Construct conditioning set
                mm_neighbors = self.nns_map[index]
                past = list(mm_neighbors) 
                data_list = []

                if past:
                    data_list.append(current_np[past])  
                if time_idx < 2:
                    cov_matrix = cov_map[(time_idx,index)]['cov_matrix']
                    tmp1 = cov_map[(time_idx,index)]['tmp1']
                    cov_xx_inv = cov_map[(time_idx,index)]['cov_xx_inv']
                    cov_ygivenx = cov_map[(time_idx,index)]['cov_ygivenx']
                    cond_mean_tmp = cov_map[(time_idx,index)]['cond_mean_tmp']
                    log_det = cov_map[(time_idx,index)]['log_det']
                    locs = cov_map[(time_idx,index)]['locs']
                else:
                    cov_matrix = cov_map[(2,index)]['cov_matrix']
                    tmp1 = cov_map[(2,index)]['tmp1']
                    cov_xx_inv = cov_map[(2,index)]['cov_xx_inv']
                    cov_ygivenx = cov_map[(2,index)]['cov_ygivenx']
                    cond_mean_tmp = cov_map[(2,index)]['cond_mean_tmp']
                    log_det = cov_map[(2,index)]['log_det']
                    locs = cov_map[(2,index)]['locs']

                if time_idx >= 1:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = one_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if time_idx > 1:
                    two_hour_lag = self.input_map[key_list[time_idx - 2]]
                    past_conditioning_data = two_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)
    
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                aggregated_arr = torch.vstack((current_row, conditioning_data))
                aggregated_y = aggregated_arr[:, 2]

                cov_yx = cov_matrix[0, 1:]
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, aggregated_y))
                beta = torch.linalg.solve(tmp1, tmp2)
                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # Mean and variance of y|x
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (aggregated_y[1:] - mu_neighbors))
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)

            
                neg_log_lik += 0.5 * (log_det + quad_form)
        return neg_log_lik

    def vecchia_local_full_cond(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        neg_log_lik = 0.0

        print(self.aggregated_data.shape)
        for idx in range(0, len(self.aggregated_data)):
            current_row = self.aggregated_data[idx,:4]
            current_y = self.aggregated_data[idx,2]
            conditioning_data = self.aggregated_data[:idx,:4]

            torch_arr = torch.vstack((current_row, conditioning_data))
            y_and_neighbors = torch_arr[:, 2]
            locs = torch_arr[:, :2]

            cov_matrix = covariance_function(params=params, y= torch_arr, x= torch_arr)
            cov_yx = cov_matrix[0, 1:]
            cov_xx = cov_matrix[1:, 1:]
   
            tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
            tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_and_neighbors))   
            beta = torch.linalg.solve(tmp1, tmp2)

            mu = torch.matmul(locs, beta)
            mu_current = mu[0]
            mu_neighbors = mu[1:]

            sigma = cov_matrix[0, 0]
            cov_ygivenx = sigma - torch.matmul(cov_yx, torch.linalg.solve(cov_xx, cov_yx))
            cond_mean = mu_current + torch.matmul(cov_yx, torch.linalg.solve( cov_xx,(y_and_neighbors[1:] - mu_neighbors) ) )
            
            alpha = current_y - cond_mean
            quad_form = alpha**2 * (1 / cov_ygivenx)
            log_det = torch.log(cov_ygivenx)
     
            neg_log_lik += 0.5 * (log_det + quad_form) 
        return neg_log_lik
<<<<<<< HEAD
<<<<<<< HEAD
    '''
    def _build_conditioning_set(self, time_idx: int, index: int):
        """Helper function to build the conditioning set data."""
        current_np = self.input_map[self.key_list[time_idx]]
        current_row = current_np[index].reshape(1, -1)
        
        mm_neighbors = self.nns_map[index]
        past = list(mm_neighbors) 
        data_list = []

        if past:
            data_list.append(current_np[past])

        if time_idx > 0:
            one_hour_lag = self.input_map[self.key_list[time_idx - 1]]
            data_list.append(one_hour_lag[past + [index], :])

        if time_idx > 1:
            two_hour_lag = self.input_map[self.key_list[time_idx - 2]]
            data_list.append(two_hour_lag [past + [index], :])

        if data_list:
            conditioning_data = torch.vstack(data_list)
        else:
            conditioning_data = torch.empty((0, current_row.shape[1]), device=self.device, dtype=torch.float64)
            
        aggregated_arr = torch.vstack((current_row, conditioning_data))
        return aggregated_arr.to(dtype=torch.float64)
    
    def cov_structure_saver(self, params: torch.Tensor, covariance_function: Callable) -> Dict[str,Any]:
        """
        Optimized version.
        Uses Cholesky to pre-compute all expensive parts.
        Saves Cholesky factors for re-use.
        """
        cov_map = defaultdict(lambda: defaultdict(dict))
        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        # Hardcoded loop to 3
        for time_idx in range(0,3):
            # Handle if key_list is shorter than 3
            if time_idx >= len(key_list):
                break
            current_np = self.input_map[key_list[time_idx]]

            for index in range(cut_line, self.size_per_hour):
                
                aggregated_arr = self._build_conditioning_set(time_idx, index)
                locs = aggregated_arr[:, :2]
                aggregated_y = aggregated_arr[:, 2] # Get the Y-vector

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                
                # --- OPTIMIZATION 1: Cholesky for tmp1 ---
                try:
                    # Add jitter FOR Cholesky, but not to original cov_matrix
                    jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                    L_full = torch.linalg.cholesky(cov_matrix + jitter)
                except torch.linalg.LinAlgError:
                    print(f"Warning: Cholesky (full) failed for {(time_idx, index)}")
                    continue # Skip this point
                    
                C_inv_locs = torch.cholesky_solve(locs, L_full, upper=False)
                tmp1 = torch.matmul(locs.T, C_inv_locs)
            
                # --- OPTIMIZATION 2: Cholesky for conditional mean/var ---
                sigma = cov_matrix[0, 0]
                cov_yx = cov_matrix[0, 1:]  # (1, N) -> 1D tensor of shape (N,)
                cov_xx = cov_matrix[1:, 1:] # (N, N)
                
                try:
                    jitter_xx = torch.eye(cov_xx.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                    L_xx = torch.linalg.cholesky(cov_xx + jitter_xx)
                except torch.linalg.LinAlgError:
                    print(f"Warning: Cholesky (partial) failed for {(time_idx, index)}")
                    continue # Skip this point

                # --- FIX for UserWarning and RuntimeError ---
                # cov_yx is 1D (N,). We need it as a 2D column vector (N, 1)
                # .unsqueeze(-1) achieves this. .T was unnecessary.
                z = torch.cholesky_solve(cov_yx.unsqueeze(-1), L_xx, upper=False) # (N, 1)
                cond_mean_tmp = z.T # (1, N)
          
                # matmul (1,N) @ (N,1) -> (1,1). Squeeze to scalar
                cov_ygivenx = sigma - torch.matmul(cond_mean_tmp, cov_yx.unsqueeze(-1)).squeeze()
                log_det = torch.log(cov_ygivenx)
            
                cov_map[(time_idx,index)] = {
                    'tmp1': tmp1.clone().detach(),
                    'L_full': L_full.clone().detach(), # Save Cholesky factor
                    'cov_ygivenx': cov_ygivenx.clone().detach(),
                    'cond_mean_tmp': cond_mean_tmp.clone().detach(),
                    'log_det': log_det.clone().detach(),
                    'locs': locs.clone().detach(),
                }
        return cov_map

    def vecchia_oct22(self, params: torch.Tensor, covariance_function: Callable, cov_map:Dict[str,Any]) -> torch.Tensor:
        """
        Optimized version.
        Uses pre-computed Cholesky factors for fast O(N^2) solves.
        Does NOT rebuild data.
        """
        cut_line= self.nheads
        key_list = list(self.input_map.keys())
        if not key_list:
             return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        neg_log_lik = 0.0
        
        # --- Head Calculation ---
        # Ensure 'heads' data is on the correct device
        heads_data = [self.input_map[key_list[0]][:cut_line,:]]
        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[key_list[time_idx]][:cut_line,:]
            heads_data.append(tmp)
        heads = torch.cat(heads_data, dim=0).to(self.device)

        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)          
        # --- End Head Calculation ---
        
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[key_list[time_idx]] # Added this line, was missing
            for index in range(cut_line, self.size_per_hour):
                
                # Load pre-computed values
                map_key = (time_idx, index) if time_idx < 2 else (2, index)
                
                if map_key not in cov_map:
                    # This can happen if saver loop is shorter or cholesky failed
                    continue 
                    
                # Load all pre-computed structural components
                tmp1 = cov_map[map_key]['tmp1']
                L_full = cov_map[map_key]['L_full']
                locs = cov_map[map_key]['locs']
                cov_ygivenx = cov_map[map_key]['cov_ygivenx']
                cond_mean_tmp = cov_map[map_key]['cond_mean_tmp']
                log_det = cov_map[map_key]['log_det']

                # --- FIX: Re-build aggregated_arr and aggregated_y for CURRENT time_idx ---
                # This matches the (correct) logic of your original code, which
                # mixes a stationary covariance structure (from t=2) with
                # a time-varying data vector (from current time_idx).
                aggregated_arr = self._build_conditioning_set(time_idx, index)
                aggregated_y = aggregated_arr[:, 2]
                current_y = aggregated_y[0]
                mu_neighbors_y = aggregated_y[1:]
                # --- End Fix ---

                # --- OPTIMIZATION: O(N^2) solve vs O(N^3) ---
                # Use cholesky_solve with the pre-computed L_full
                C_inv_y = torch.cholesky_solve(aggregated_y.unsqueeze(-1), L_full, upper=False)
                tmp2 = torch.matmul(locs.T, C_inv_y)
                
                try:
                    # Add jitter to the small (2,2) system as well
                    jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                    beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2) # Solves small (2, 2) system
                except torch.linalg.LinAlgError:
                    print(f"Warning: Could not solve for beta (Vecchia) at {(time_idx, index)}")
                    continue # Skip this point
                    
                mu = torch.matmul(locs, beta).squeeze() # (N+1,)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                # Mean and variance of y|x
                
                # --- FIX for RuntimeError ---
                # We need (1,N) @ (N,1) to get a (1,1) tensor, which squeezes to scalar []
                # (mu_neighbors_y - mu_neighbors) is 1D (N,), so unsqueeze it to (N,1)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (mu_neighbors_y - mu_neighbors).unsqueeze(-1)).squeeze()
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
            
                # log_det and quad_form are now both scalars []
                neg_log_lik += 0.5 * (log_det + quad_form)
                
        return neg_log_lik
'''
=======


>>>>>>> 98fc71c474ddced6792e89e9ab27c07529da5b48
=======


>>>>>>> 0a418ac421c02a3cd32b6e4c97b2bdc92cdb79b7
class model_fitting(vecchia_experiment): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    def compute_vecc_nll_may9(self,params , covariance_function, cov_map):
        vecc_nll = self.vecchia_may9(params, covariance_function, cov_map)
        return vecc_nll

    def compute_vecc_nll_grp9(self,params , covariance_function, cov_map):
        vecc_nll = self.vecchia_grouping(params, covariance_function, cov_map)
        return vecc_nll

    def compute_full_nll(self,params, covariance_function):
        full_nll = self.full_likelihood(params=params, input_data=self.aggregated_data, y=self.aggregated_response, covariance_function= covariance_function) 
        return full_nll
    
    def optimizer_fun(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler

    # use adpating lr
    def run_full(self, params, optimizer, scheduler,  covariance_function, epochs=10 ):
        
        """
        Run the training loop for the full likelihood model.

        Args:
            params (torch.Tensor): Model parameters.
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            epochs (int): Number of epochs to train.

        Returns:
            list: Final parameters and loss.
            int: Number of epochs run.
        """

        prev_loss= float('inf')

        tol = 1e-3  # Convergence tolerance
        for epoch in range(epochs):  
            optimizer.zero_grad()  
            
            loss = self.compute_full_nll(params, covariance_function)
            loss.backward()  # Backpropagate the loss
            
            # Print gradients and parameters every 10th epoch
            #if epoch % 10 == 0:
            #    print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  
            scheduler.step()  
            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()

        params = [torch.round(x*1000).detach().numpy()/1000 for x in params]
        loss = (torch.round(loss*1000)/1000).item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {params}')
        return params + [loss], epoch


    
    def run_vecc_may9(self, params, optimizer, scheduler,  covariance_function, cov_map,epochs=10):
        prev_loss= float('inf')
        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  
            optimizer.zero_grad()  
            loss = self.compute_vecc_nll_may9(params, covariance_function, cov_map)
            loss.backward(retain_graph=True) # Backpropagate the loss with retain_graph=True
            # loss.backward()

            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        params = [torch.round(x*1000).detach().numpy()/1000 for x in params]
        loss = (torch.round(loss*1000)/1000).item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {params}')
        return params + [loss], epoch

    def run_vecc_grp9(self, params, optimizer, scheduler,  covariance_function, cov_map,epochs=10):
        prev_loss= float('inf')
        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  
            optimizer.zero_grad()  
            loss = self.compute_vecc_nll_grp9(params, covariance_function, cov_map)
            loss.backward(retain_graph=True) # Backpropagate the loss with retain_graph=True
            # loss.backward()

            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        params = [torch.round(x*1000).detach().numpy()/1000 for x in params]
        loss = (torch.round(loss*1000)/1000).item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {params}')
        return params + [loss], epoch
<<<<<<< HEAD
<<<<<<< HEAD
. '''
class model_fitting(vecchia_experiment): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    def compute_vecc_nll_oct22(self,params , covariance_function, cov_map):
        vecc_nll = self.vecchia_oct22(params, covariance_function, cov_map)
        return vecc_nll

    # TODO: This function references 'self.vecchia_grouping', which is not defined in this file.
    #       Commenting it out to prevent errors.
    # def compute_vecc_nll_grp9(self,params , covariance_function, cov_map):
    #     vecc_nll = self.vecchia_grouping(params, covariance_function, cov_map)
    #     return vecc_nll

    def compute_full_nll(self,params, covariance_function):
        full_nll = self.full_likelihood(params=params, input_data=self.aggregated_data, y=self.aggregated_response, covariance_function= covariance_function) 
        return full_nll
    
    def optimizer_fun(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler


    def optimizer_fun(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, 
                    scheduler_type:str='step', step_size=40, gamma=0.5, T_max=10): # <-- Added new arguments
        
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        
        if scheduler_type.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max) # Uses T_max
        else: # Default to StepLR
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        return optimizer, scheduler

    def optimizer_fun10_23(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, 
                    scheduler_type:str='step', step_size=40, gamma=0.5, T_max=10): # <-- Added new arguments
        
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)
        
        if scheduler_type.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max) # Uses T_max
        else: # Default to StepLR
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        return optimizer, scheduler
    


    # use adpating lr
    def run_full(self, params, optimizer, scheduler,  covariance_function, epochs=10 ):
        
        """
        Run the training loop for the full likelihood model.

        Args:
            params (torch.Tensor): Model parameters.
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            epochs (int): Number of epochs to train.

        Returns:
            list: Final parameters and loss.
            int: Number of epochs run.
        """

        prev_loss= float('inf')

        tol = 1e-3  # Convergence tolerance
        for epoch in range(epochs):  
            optimizer.zero_grad()  
            
            loss = self.compute_full_nll(params, covariance_function)
            loss.backward()  # Backpropagate the loss
            
            # Print gradients and parameters every 10th epoch
            #if epoch % 10 == 0:
            #    print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  
            scheduler.step()  
            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()

        final_params_list = [p.item() for p in params] # Get final params as a list
        loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {final_params_list}')
        return final_params_list + [loss], epoch


    
    def run_vecc_oct22(self, params, optimizer, scheduler,  covariance_function, epochs=10):
        prev_loss= float('inf')
        tol = 1e-4  # Convergence tolerance
        
        for epoch in range(epochs):  
            
            # --- FIX: Re-compute cov_map INSIDE the loop ---
            # The cov_map depends on 'params', so it must be re-computed
            # every time 'params' is updated.
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()  
            loss = self.compute_vecc_nll_oct22(params, covariance_function, cov_map)
            
            # --- FIX: Removed retain_graph=True ---
            # This was causing a memory leak. Since the graph is
            # rebuilt from scratch every epoch (including cov_map),
            # we no longer need to retain it.
            loss.backward()

            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()

        final_params_list = [p.item() for p in params] # Get final params as a list
        loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {final_params_list}')
        return final_params_list + [loss], epoch
    

    def run_vecc_oct23(self, params_list, optimizer, scheduler,  covariance_function, epochs=10):
        prev_loss= float('inf')
        tol = 1e-4  # Convergence tolerance
        
        for epoch in range(epochs):  
            params = torch.cat(params_list)
            # --- FIX: Re-compute cov_map INSIDE the loop ---
            # The cov_map depends on 'params', so it must be re-computed
            # every time 'params' is updated.
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()  
            loss = self.compute_vecc_nll_oct22(params, covariance_function, cov_map)
            
            # --- FIX: Removed retain_graph=True ---
            # This was causing a memory leak. Since the graph is
            # rebuilt from scratch every epoch (including cov_map),
            # we no longer need to retain it.
            loss.backward()

            # Print gradients and parameters every 10th epoch
            if epoch % 50 == 0:
                print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()

        final_params_list = [p.item() for p in params] # Get final params as a list
        loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss}, \n vecc Parameters: {final_params_list}')
        return final_params_list + [loss], epoch
    

    # --- NEW ADVANCED TRAINING LOOP ---

    def run_vecc_advanced(self, params, optimizer, scheduler, covariance_function, epochs=10, log_param_indices:List[int]=None):
        """
        Advanced training loop based on the user's reference.
        - Tracks best loss and best params.
        - Handles NaN/Inf loss.
        - Clips gradients.
        - Prints parameters in "natural scale" (by exponentiating log-params).
        """
        best_loss = float('inf')
        # We only have one tensor, but we'll use a list for compatibility
        best_params_state = params.detach().clone()
        epochs_completed = 0
        
        if log_param_indices is None:
            log_param_indices = []

        def get_printable_params(p_tensor):
            """Helper to convert log-params to natural scale for printing."""
            p_cat = p_tensor.detach().clone().cpu()
            try:
                if log_param_indices:
                    log_vals = p_cat[log_param_indices]
                    if not (torch.isnan(log_vals).any() or torch.isinf(log_vals).any()):
                        p_cat[log_param_indices] = torch.exp(log_vals)
                    else:
                        p_cat[log_param_indices] = float('nan')
            except IndexError:
                print("Warning: log_param_indices out of bounds.")
                pass # Fail gracefully
            return p_cat.numpy().round(4)

        for epoch in range(epochs):
            epochs_completed = epoch
            
            # --- This is the key logic from run_vecc_may9 ---
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()
            
            loss = self.compute_vecc_nll_oct22(params, covariance_function, cov_map)
            # --- End key logic ---

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss became NaN or Inf at epoch {epoch+1}. Stopping.")
                if epoch == 0: 
                    best_params_state = None # Mark as failed from start
                break 

            loss.backward()

            # Check for NaN gradients
            if params.grad is not None and (torch.isnan(params.grad).any() or torch.isinf(params.grad).any()):
                 warnings.warn(f"NaN or Inf gradient at epoch {epoch+1}. Skipping step.")
                 optimizer.zero_grad() 
                 continue 

            # Clip gradients
            torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step() 

            current_loss_item = loss.item()
            if current_loss_item < best_loss:
                # Check if params are valid before saving
                params_valid = not (torch.isnan(params.data).any() or torch.isinf(params.data).any())
                if params_valid:
                    best_loss = current_loss_item
                    best_params_state = params.detach().clone()

            if epoch % 10 == 0 or epoch == epochs - 1: # Print more often
                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
                print(f'--- Epoch {epoch+1}/{epochs} (LR: {current_lr:.6f}) ---')
                print(f' Loss: {current_loss_item:.4f} (Best: {best_loss:.4f})')
                print(f' Parameters (Natural Scale): {get_printable_params(params)}')

        if best_params_state is None:
            print("\n--- Training Failed (NaN/Inf from start) ---")
            return None, epochs_completed

        # --- Prepare final output ---
        final_params_log_scale = best_params_state.cpu()
        final_params_natural_scale = get_printable_params(final_params_log_scale)

        final_params_rounded = [round(p.item(), 4) if not np.isnan(p.item()) else float('nan') for p in final_params_natural_scale]
        final_loss_rounded = round(best_loss, 3) if best_loss != float('inf') else float('inf')

        print("\n--- Training Complete ---")
        print(f'\nFINAL BEST STATE ACHIEVED (during training):')
        print(f'Best Loss: {final_loss_rounded}')
        print(f'Parameters Corresponding to Best Loss (Natural Scale): {final_params_rounded}')

        return final_params_rounded + [final_loss_rounded], epochs_completed
    
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter when we have improvement
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop
    
####################
