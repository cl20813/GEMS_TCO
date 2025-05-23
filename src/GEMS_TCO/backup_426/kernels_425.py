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

import copy    
import logging     # for logging
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

class spatio_temporal_kernels:               #sigmasq range advec beta  nugget
    def __init__(self, smooth, input_map, aggregated_data, nns_map, mm_cond_number):
        # self.smooth = torch.tensor(smooth,dtype=torch.float64 )
        self.smooth = smooth
        self.input_map = input_map
        self.aggregated_data = aggregated_data[:,:4]
        self.aggregated_response = aggregated_data[:,2]

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
    def custom_distance_matrix(self, U, V):
        # Efficient distance computation with broadcasting
        spatial_diff = torch.norm(U[:, :2].unsqueeze(1) - V[:, :2].unsqueeze(0), dim=2)

        temporal_diff = torch.abs(U[:, 2].unsqueeze(1) - V[:, 2].unsqueeze(0))
        distance = (spatial_diff**2 + temporal_diff**2)  # move torch.sqrt to covariance function to track gradients of beta and avec
        return distance
    

    def precompute_coords_anisotropy(self, params, y: torch.Tensor, x: torch.Tensor)-> torch.Tensor:
        sigmasq, range_lat, range_lon, advec_lat, advec_lon, beta, nugget = params

        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")

        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3]

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
        out = torch.zeros_like(distance)
        
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
    
class likelihood_function(spatio_temporal_kernels):
    def __init__(self, smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.nheads = nheads
        # Any additional initialization for dignosis class can go here
              
    def full_likelihood(self,params: torch.Tensor, input_np: torch.Tensor, y: torch.Tensor, covariance_function) -> torch.Tensor:
        input_arr = input_np[:, :4]  ## input_np is aggregated data over a day.
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
     
        return  neg_log_lik 
    
    def vecchia_extrapolate(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        self.cov_map = defaultdict(list)
        neg_log_lik = 0.0
          
        for time_idx in range(len(self.input_map)):
            current_np = self.input_map[self.key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            # cur_heads = current_np[:5, :]
            # neg_log_lik += self.full_likelihood(params, cur_heads, cur_heads[:, 2], covariance_function)

            for index in range(0, self.size_per_hour):
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
            
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx']
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs = self.cov_map[index]['locs']
                    
                    last_hour_np = self.input_map[self.key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index] , :]
                    data_list.append(past_conditioning_data)

                    if data_list:
                        conditioning_data = torch.vstack(data_list)
                    else:
                        conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float32)

                    np_arr = torch.vstack((current_row, conditioning_data))
                    y_and_neighbors = np_arr[:, 2]

                    cov_yx = cov_matrix[0, 1:]

                    y_arr = y_and_neighbors
                    tmp1 = tmp_for_beta
                    tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_arr))
                    beta = torch.linalg.solve(tmp1, tmp2)

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

                cov_matrix = covariance_function(params=params, y= np_arr, x= np_arr)
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)
                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
            
                y_arr = y_and_neighbors
                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_arr))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)
 
                if time_idx == 1:
                    self.cov_map[index] = {
                        'tmp_for_beta': tmp1,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
               
                        'cov_ygivenx': cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs': locs
                    }
        return neg_log_lik

    def vecchia_interpolation_1to6(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        cut_line = self.nheads
        self.cov_map = defaultdict(list)
        neg_log_lik = 0.0
        key_list = sorted(self.input_map)
        cut_line = cut_line
        heads = self.input_map[key_list[0]][:cut_line,:]
        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[key_list[time_idx]][:cut_line,:]
            heads = torch.cat( (heads,tmp), dim=0)

        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)          
        
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[self.key_list[time_idx]]

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

                if time_idx > 0 and time_idx<7:
                    last_hour_np = self.input_map[self.key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                    last_hour_np = self.input_map[self.key_list[time_idx +1]]
                    # if index==200:
                    #     print(self.input_map[self.key_list[time_idx-6]])
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                np_arr = torch.vstack((current_row, conditioning_data))
                y_and_neighbors = np_arr[:, 2]
                locs = np_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= np_arr, x= np_arr)
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)
                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
            
                y_arr = y_and_neighbors
                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_arr))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)

        return neg_log_lik




####################

class vecchia_experiment(likelihood_function):
    def __init__(self, smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads):
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

        statistic  = torch.matmul(gradient, torch.linalg.solve(hessian_matrix, gradient))
        # print(f' statistic is {statistic}')
        return statistic
    

    def vecchia_b1(self, params: torch.Tensor, covariance_function ) -> torch.Tensor:
        self.cov_map = defaultdict(list)
        cut_line= self.nheads
        neg_log_lik = 0.0
        heads = self.input_map[self.key_list[0]][:cut_line,:]
        for time_idx in range(1, len(self.input_map)):
            tmp = self.input_map[self.key_list[time_idx]][:cut_line,:]
            heads = torch.cat( (heads,tmp), dim=0)

        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)          
        
        for time_idx in range(0,len(self.input_map)):
            current_np = self.input_map[self.key_list[time_idx]]

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

                if time_idx > 0:
                    last_hour_np = self.input_map[self.key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                np_arr = torch.vstack((current_row, conditioning_data))
                y_and_neighbors = np_arr[:, 2]
                locs = np_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= np_arr, x= np_arr)
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)
                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
                y_arr = y_and_neighbors
                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_arr))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)
 
        return neg_log_lik

    def vecchia_b2(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        self.cov_map = defaultdict(list)
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

                if time_idx > 0:
                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = one_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)

                if time_idx > 1:
                    two_hour_lag = self.input_map[key_list[time_idx -2]]
                    # if index==200:
                    #     print(self.input_map[self.key_list[time_idx-6]])
                    past_conditioning_data = two_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                aggregated_arr = torch.vstack((current_row, conditioning_data))
                aggregated_y = aggregated_arr[:, 2]
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)

                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
        
                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, aggregated_y))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (aggregated_y[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)
 
        return neg_log_lik
    
    def vecchia_b2_cache(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        # becareful this only work for natural order

        self.cov_map = defaultdict(list)
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

                if time_idx > 2:
                    cov_matrix = self.cov_map[index]['cov_matrix']
                    tmp1 = self.cov_map[index]['tmp1']
                    cov_xx_inv = self.cov_map[index]['cov_xx_inv']
                    cov_ygivenx = self.cov_map[index]['cov_ygivenx']
                    cond_mean_tmp = self.cov_map[index]['cond_mean_tmp']
                    log_det = self.cov_map[index]['log_det']
                    locs = self.cov_map[index]['locs']

                    one_hour_lag = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = one_hour_lag[past + [index], :]
                    data_list.append(past_conditioning_data)

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
                    continue

                if time_idx > 0:
                    last_hour_np = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                if time_idx > 1:
                    last_hour_np = self.input_map[key_list[time_idx -2]]
                    # if index==200:
                    #     print(self.input_map[self.key_list[time_idx-6]])
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                aggregated_arr = torch.vstack((current_row, conditioning_data))
                y_and_neighbors = aggregated_arr[:, 2]
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)

                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
            
                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_and_neighbors))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (y_and_neighbors[1:] - mu_neighbors))
                
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)

                if time_idx == 2:
                    self.cov_map[index] = {
                        'tmp1': tmp1,
                        'cov_xx_inv': cov_xx_inv,
                        'cov_matrix': cov_matrix,
                        'cov_ygivenx': cov_ygivenx,
                        'cond_mean_tmp': cond_mean_tmp,
                        'log_det': log_det,
                        'locs': locs
                    }
        return neg_log_lik

    def vecchia_contender(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        self.cov_map = defaultdict(list)
        cut_line = self.nheads
        key_list = list(self.input_map.keys())

        neg_log_lik = 0.0
        heads = torch.cat([self.input_map[key_list[i]][:cut_line, :] for i in range(len(self.input_map))], dim=0)
        neg_log_lik += self.full_likelihood(params, heads, heads[:, 2], covariance_function)

        sph = self.size_per_hour
        for time_idx in range(8):
            time_sph = time_idx * sph
            for index in range(time_sph + cut_line, time_sph + sph):
                current_row = self.aggregated_data[index].reshape(1, -1)
                current_y = current_row[0, 2]

                # Construct conditioning set
                mm_neighbors = self.nns_map[index % self.size_per_hour]
                past = list(mm_neighbors)
                data_list = []

                if past:
                    tmp = [x + time_sph for x in past]
                    data_list.append(self.aggregated_data[tmp])

                if time_idx > 0:
                    past_one_lag = [x + (time_idx - 1) * sph for x in past]
                    past_conditioning_data = self.aggregated_data[past_one_lag + [index - sph]]
                    data_list.append(past_conditioning_data)

                if time_idx > 1:
                    past_two_lag = [x + (time_idx - 2) * sph for x in past]
                    past_conditioning_data = self.aggregated_data[past_two_lag + [index - sph * 2], :]
                    data_list.append(past_conditioning_data)

                conditioning_data = torch.vstack(data_list) if data_list else torch.empty((0, current_row.shape[1]), dtype=torch.float64)
                aggregated_arr = torch.vstack((current_row, conditioning_data))
                aggregated_y = aggregated_arr[:, 2]
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y=aggregated_arr, x=aggregated_arr)
                cov_yx = cov_matrix[0, 1:]

                # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)

                # Compute beta
                tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
                tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, aggregated_y))
                beta = torch.linalg.solve(tmp1, tmp2)

                mu = torch.matmul(locs, beta)
                mu_current = mu[0]
                mu_neighbors = mu[1:]

                # Mean and variance of y|x
                sigma = cov_matrix[0, 0]
                cov_xx = cov_matrix[1:, 1:]
                cov_xx_inv = torch.linalg.inv(cov_xx)

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (aggregated_y[1:] - mu_neighbors))

                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
                log_det = torch.log(cov_ygivenx)
                neg_log_lik += 0.5 * (log_det + quad_form)

        return neg_log_lik
    
    def cov_structure_saver(self, params: torch.Tensor, covariance_function) -> None:
        
        cov_map = defaultdict(lambda: defaultdict(dict))
        cut_line= self.nheads
        key_list = list(self.input_map.keys())

        for time_idx in range(0,8):
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
                    last_hour_np = self.input_map[key_list[time_idx - 1]]
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)

                if time_idx > 1:
                    last_hour_np = self.input_map[key_list[time_idx -2]]
                    # if index==200:
                    #     print(self.input_map[self.key_list[time_idx-6]])
                    past_conditioning_data = last_hour_np[past + [index], :]
                    data_list.append(past_conditioning_data)
                
                if data_list:
                    conditioning_data = torch.vstack(data_list)
                else:
                    conditioning_data = torch.empty((0, current_row.shape[1]), dtype=torch.float64)

                aggregated_arr = torch.vstack((current_row, conditioning_data))
            
                locs = aggregated_arr[:, :2]

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                cov_yx = cov_matrix[0, 1:]
                        # Compute the log determinant of the covariance matrix
                sign, log_det = torch.slogdet(cov_matrix)

                # if sign <= 0:
                #     raise ValueError("Covariance matrix is not positive definite")
            
                # Compute beta
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

    def vecchia_reorder(self, params: torch.Tensor, covariance_function, cov_map) -> torch.Tensor:

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
          
                cov_matrix = cov_map[(time_idx,index)]['cov_matrix']
                tmp1 = cov_map[(time_idx,index)]['tmp1']
                cov_xx_inv = cov_map[(time_idx,index)]['cov_xx_inv']
                cov_ygivenx = cov_map[(time_idx,index)]['cov_ygivenx']
                cond_mean_tmp = cov_map[(time_idx,index)]['cond_mean_tmp']
                log_det = cov_map[(time_idx,index)]['log_det']
                locs = cov_map[(time_idx,index)]['locs']
        

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
  

    def vecchia_ori_order(self, params: torch.Tensor, covariance_function, cov_map) -> torch.Tensor:

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

                # Use below when working on local computer to avoid singular matrix
        cur_heads = self.aggregated_data[:20, :]
        neg_log_lik += self.full_likelihood(params, cur_heads, cur_heads[:, 2], covariance_function)

        for idx in range(20,len(self.aggregated_data)):
            current_row = self.aggregated_data[idx,:4]
            current_y = self.aggregated_data[idx,2]
            conditioning_data = self.aggregated_data[:idx,:4]

            torch_arr = torch.vstack((current_row, conditioning_data))
            y_and_neighbors = torch_arr[:, 2]
            locs = torch_arr[:, :2]

            cov_matrix = covariance_function(params=params, y= torch_arr, x= torch_arr)
            # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
            cov_xx = cov_matrix[1:, 1:]
            # cov_xx_inv = torch.linalg.inv(cov_xx)

            cov_yx = cov_matrix[0, 1:]

            # Compute the log determinant of the covariance matrix
            # sign, log_det = torch.slogdet(cov_matrix)
            # if sign <= 0:
            #     raise ValueError("Covariance matrix is not positive definite")
        
            # Compute beta
            tmp1 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, locs))
            tmp2 = torch.matmul(locs.T, torch.linalg.solve(cov_matrix, y_and_neighbors))
            beta = torch.linalg.solve(tmp1, tmp2)
            mu = torch.matmul(locs, beta)
            mu_current = mu[0]
            mu_neighbors = mu[1:]

            # Mean and variance of y|x
            sigma = cov_matrix[0, 0]
            
            # cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
            cov_ygivenx = sigma - torch.matmul(cov_yx, torch.linalg.solve(cov_xx, cov_yx))
            # cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
            cond_mean = mu_current + torch.matmul(cov_yx, torch.linalg.solve( cov_xx,(y_and_neighbors[1:] - mu_neighbors) ) )
            
            alpha = current_y - cond_mean
            quad_form = alpha**2 * (1 / cov_ygivenx)
            log_det = torch.log(cov_ygivenx)
     
            neg_log_lik += 0.5 * (log_det + quad_form) 
        return neg_log_lik


    def vecchia_like_local_computer(self, params: torch.Tensor, covariance_function) -> torch.Tensor:
        self.cov_map = defaultdict(list)
        neg_log_lik = 0.0
        
        for time_idx in range(len(self.input_map)):
            current_np = self.input_map[self.key_list[time_idx]]

            # Use below when working on local computer to avoid singular matrix
            #cur_heads = current_np[:21, :]
            #neg_log_lik += self.full_likelihood(params, cur_heads, cur_heads[:, 2], covariance_function)

            for index in range(0, self.size_per_hour):
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

                cov_matrix = covariance_function(params=params, y= np_arr, x= np_arr)
                # print(f'Condition number: {torch.linalg.cond(cov_matrix)}')
                L = torch.linalg.cholesky(cov_matrix)
                L11 = L[:1, :1]
                L12 = torch.zeros(L[:1, 1:].shape)
                L21 = L[1:, :1]
                L22 = L[1:, 1:]
                L11_inv = torch.linalg.inv(L11)
                L22_inv = torch.linalg.inv(L22)

                # First block: [L11_inv, L12]
                upper_block = torch.cat((L11_inv, L12), dim=1)  # Concatenate along columns (dim=1)

                # Second block: [-torch.matmul(torch.matmul(L22_inv, L21), L11_inv), L22_inv]
                lower_left = -torch.matmul(torch.matmul(L22_inv, L21), L11_inv)
                lower_block = torch.cat((lower_left, L22_inv), dim=1)  # Concatenate along columns (dim=1)

                # Combine the upper and lower blocks
                L_inv = torch.cat((upper_block, lower_block), dim=0)  # Concatenate along rows (dim=0)

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

                cov_ygivenx = sigma - torch.matmul(cov_yx, torch.matmul(cov_xx_inv, cov_yx))
                cond_mean_tmp = torch.matmul(cov_yx, cov_xx_inv)
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
    

class model_fitting(vecchia_experiment): 
    def __init__(self, smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    # Example function to compute out1
    def compute_vecc_nll_interpolate(self,params, covariance_function):

        vecc_nll = self.vecchia_b2(params, covariance_function)
        return vecc_nll
    
    # Example function to compute out1
    def compute_vecc_nll_reorder(self,params, covariance_function, cov_map):

        vecc_nll = self.vecchia_reorder(params, covariance_function, cov_map)
        return vecc_nll

    def compute_vecc_nll_ori_order(self,params , covariance_function, cov_map):
   
        vecc_nll = self.vecchia_ori_order(params, covariance_function, cov_map)
        return vecc_nll



    def compute_full_nll(self,params, covariance_function):
        full_nll = self.full_likelihood(params=params, input_np=self.aggregated_data, y=self.aggregated_response, covariance_function= covariance_function) 
        return full_nll
    
    def optimizer_fun(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler

    def optimizer_testing(self, params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.5):
        optimizer = torch.optim.Adam([params], lr=lr, betas=betas, eps=eps)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # Decrease LR by a factor of 0.1 every 10 epochs
        return optimizer, scheduler

    # use adpating lr
    def run_full(self, params, optimizer, scheduler,  covariance_function, epochs=10 ):
        prev_loss= float('inf')

        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  # Number of epochs
            optimizer.zero_grad()  # Zero the gradients 
            
            loss = self.compute_full_nll(params, covariance_function)
            loss.backward()  # Backpropagate the loss
            
            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate
            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
    
        return params.detach().numpy().tolist() + [ loss.item()], epoch

    def run_vecc_interpolate(self, params, optimizer, scheduler, covariance_function, epochs=10):
        prev_loss= float('inf')

        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  # Number of epochs
            optimizer.zero_grad()  # Zero the gradients 
            
            loss = self.compute_vecc_nll_interpolate(params , covariance_function)
            loss.backward()  # Backpropagate the loss
            
            # Print gradients and parameters every 10th epoch
            # if epoch % 100 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate
            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1}, : Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')

        return params.detach().numpy().tolist() + [ loss.item()], epoch

    def run_vecc_reorder(self, params, optimizer, scheduler,  covariance_function, cov_map,epochs=10):
        prev_loss= float('inf')

        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  # Number of epochs
            optimizer.zero_grad()  # Zero the gradients 
            
            loss = self.compute_vecc_nll_reorder(params, covariance_function, cov_map)
            loss.backward(retain_graph=True) # Backpropagate the loss with retain_graph=True
            # loss.backward()

            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate
   
            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')

        return params.detach().numpy().tolist() + [ loss.item()], epoch
    

    def run_vecc_ori_order(self, params, optimizer, scheduler,  covariance_function, cov_map,epochs=10):
        prev_loss= float('inf')

        tol = 1e-4  # Convergence tolerance
        for epoch in range(epochs):  # Number of epochs
            optimizer.zero_grad()  # Zero the gradients 
            
            loss = self.compute_vecc_nll_ori_order(params, covariance_function, cov_map)
            loss.backward(retain_graph=True) # Backpropagate the loss with retain_graph=True
            # loss.backward()

            # Print gradients and parameters every 10th epoch
            # if epoch % 500 == 0:
            #     print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            # print(f'Epoch {epoch+1}, Gradients: {params.grad.numpy()}\n Loss: {loss.item()}, Parameters: {params.detach().numpy()}')
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at epoch {epoch}")
                print(f'Epoch {epoch+1},  \n vecc Parameters: {params.detach().numpy()}')
                break

            prev_loss = loss.item()
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {loss.item()}, \n vecc Parameters: {params.detach().numpy()}')

        return params.detach().numpy().tolist() + [ loss.item()], epoch


####################
