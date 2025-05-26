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
from typing import Dict, Any, Callable

import copy    
import logging     # for logging
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

class spatio_temporal_kernels:               #sigmasq range advec beta  nugget
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
        # self.smooth = torch.tensor(smooth,dtype=torch.float64 )
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
        
        distance, non_zero_indices = self.precompute_coords_anisotropy(params, x,y)
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
        
        distance, non_zero_indices = self.precompute_coords_anisotropy(params, x,y)
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
    def __init__(self, epsilon:float, coarse_factor_head:int, coarse_factor_cond:int, smooth:float, input_map: Dict[str, Any], aggregated_data:torch.Tensor, nns_map: np.ndarray, mm_cond_number:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.smooth = torch.tensor(smooth, dtype= torch.float64)
        
        self.epsilon = epsilon  # starting point for the spline fitting
        sample_params = [25, 0.5, 0.5, 0, 0, 2, 5] # just random nuumber to initialize spline
        sample_params = torch.tensor(sample_params, dtype=torch.float64, requires_grad=True)
        
        self.coarse_factor_head = coarse_factor_head
 
        self.coarse_factor_cond = coarse_factor_cond

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
        fit_distances = torch.linspace(0, max_distance + 1e-6 , len_distance_arr// coarse_factor)
        non_zero_indices = fit_distances != 0
        out = torch.zeros_like(fit_distances, dtype= torch.float64)

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

                cond_spline_object = self.fit_cubic_spline(target_distances_for_cond, self.coarse_factor_cond )  # change here  
                cov_matrix = self.interpolate_cubic_spline(params, target_distances_for_cond, cond_spline_object)

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

        print(heads.shape )
        distances_heads, _ = self.precompute_coords_anisotropy(params, heads, heads)
        spline_object_head = self.fit_cubic_spline( distances_heads, self.coarse_factor_head)  # change here

        neg_log_lik += self.full_likelihood_using_spline(params, heads[:,:4], heads[:,2], distances_heads, spline_object_head)
    
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

    def compute_full_nll(self, params:torch.Tensor, distances:torch.Tensor): 
        nll = self.full_likelihood_using_spline( params, distances)
        return nll

    def compute_vecchia_nll(self, params:torch.Tensor): 
        cov_map = self.cov_structure_saver(params)
        nll = self.vecchia_nll_using_spline(params, cov_map)
        return nll

    def optimizer_fun(self, params:torch.Tensor, lr:float =0.01, betas: tuple=(0.9, 0.8), eps:float=1e-8, step_size:int=40, gamma:float=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler

    def run_full(self, params:torch.Tensor, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, epochs:int=10 ):

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
        for epoch in range(epochs):  
            optimizer.zero_grad()  # Zero the gradients 
            distances, non_zero_indices = self.precompute_coords_anisotropy(params, self.new_aggregated_data[:,:4], self.new_aggregated_data[:,:4])
            
            loss = self.compute_full_nll(params, distances)
            loss.backward()  # Backpropagate the loss

            # Gradient and Parameter Logging for every 10th epoch
            #if epoch % 10 == 0:
            #    print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
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

    def fit_vecchia(self, params:torch.Tensor, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, epochs:int=10 ):

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

        for epoch in range(epochs):  
            optimizer.zero_grad()  # Zero the gradients 
            # distance is a function of parameters
            # distances, non_zero_indices = self.precompute_coords_anisotropy(params, self.new_aggregated_data[:,:4], self.new_aggregated_data[:,:4])
            
            loss = self.compute_vecchia_nll(params)
            loss.backward()  # Backpropagate the loss

            # Gradient and Parameter Logging for every 10th epoch
            if epoch % 10 == 0:
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
        ''' 
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
