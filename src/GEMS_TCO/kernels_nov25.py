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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn
from scipy.interpolate import splrep, splev

# Fit your "spline" by just storing the x and y
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from typing import Dict, Any, Callable, List, Tuple

import time
import copy    
import logging     # for logging
# Add your custom path

import warnings
import torch
from functools import partial
# --- MODIFIED 'optimizer_fun' IN 'model_fitting' CLASS ---
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # Need to import this
# ... (other imports) ...

sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

class SpatioTemporalModel:
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

        self.input_map = input_map
        self.aggregated_data = aggregated_data[:,:4]

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

    def full_likelihood(self,params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
            """
            Optimized likelihood function using Cholesky decomposition.
            NOW INCLUDES AN INTERCEPT (BETA_0).
            """
            input_data = input_data.to(torch.float64)
            y = y.to(torch.float64)
                    
            cov_matrix = covariance_function(params=params, y= input_data, x= input_data)
            
            try:
                # Add jitter *before* Cholesky for stability
                jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                L = torch.linalg.cholesky(cov_matrix + jitter)
            except torch.linalg.LinAlgError:
                print("Warning: Cholesky decomposition failed.")
                return torch.tensor(torch.inf, device=params.device, dtype=params.dtype)
            
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            
            # --- 💥 START FIX: ADD INTERCEPT ---
            # Get original (N, 2) location matrix
            locs_original = input_data[:,:2].to(torch.float64) 
            
            # Create an (N, 1) column of ones for the intercept
            intercept = torch.ones(locs_original.shape[0], 1, 
                                device=locs_original.device, 
                                dtype=torch.float64)
            
            # Concatenate to create the (N, 3) design matrix X
            locs = torch.cat((intercept, locs_original), dim=1)
            # --- 💥 END FIX ---
            
            if y.dim() == 1:
                y_col = y.unsqueeze(-1).to(torch.float64)
            else:
                y_col = y.to(torch.float64)

            # Solve for C_inv_X and C_inv_y
            C_inv_X = torch.cholesky_solve(locs, L, upper=False)  # Solves C*z = X, locs is (N, 3)
            C_inv_y = torch.cholesky_solve(y_col, L, upper=False) # Solves C*z = y

            # Compute beta
            tmp1 = torch.matmul(locs.T, C_inv_X)       # (3, N) @ (N, 3) = (3, 3)
            tmp2 = torch.matmul(locs.T, C_inv_y)       # (3, N) @ (N, 1) = (3, 1)
            
            try:
                # Add jitter to the small (3,3) system
                jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2) # Solves (3, 3) system
            except torch.linalg.LinAlgError:
                print("Warning: Could not solve for beta. X^T C_inv X may be singular.")
                return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

            # Compute the mean
            mu = torch.matmul(locs, beta)              # (N, 3) @ (3, 1) = (N, 1)
            y_mu = y_col - mu
            
            # Compute quadratic form
            C_inv_y_mu = torch.cholesky_solve(y_mu, L, upper=False)
            quad_form = torch.matmul(y_mu.T, C_inv_y_mu) 

            neg_log_lik = 0.5 * (log_det + quad_form.squeeze())
            return  neg_log_lik


class VecchiaLikelihood(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map :Dict[str,Any], aggregated_data: torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.nheads = nheads



class VecchiaLikelihood(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map :Dict[str,Any], aggregated_data: torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.nheads = nheads

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
            Optimized version. Pre-computes all expensive parts.
            NOW INCLUDES AN INTERCEPT (BETA_0).
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
                    
                    # --- 💥 START FIX: ADD INTERCEPT ---
                    # Get original (N+1, 2) location matrix
                    locs_original = aggregated_arr[:, :2]
                    
                    # Create an (N+1, 1) column of ones
                    intercept = torch.ones(locs_original.shape[0], 1, 
                                        device=locs_original.device, 
                                        dtype=torch.float64)
                    
                    # Concatenate to create the (N+1, 3) design matrix X
                    locs = torch.cat((intercept, locs_original), dim=1)
                    # --- 💥 END FIX ---
                    
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
                        
                    # This calculation now uses the (N+1, 3) locs matrix
                    C_inv_locs = torch.cholesky_solve(locs, L_full, upper=False)
                    tmp1 = torch.matmul(locs.T, C_inv_locs) # tmp1 is now (3, 3)
                
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

                    z = torch.cholesky_solve(cov_yx.unsqueeze(-1), L_xx, upper=False) # (N, 1)
                    cond_mean_tmp = z.T # (1, N)
            
                    cov_ygivenx = sigma - torch.matmul(cond_mean_tmp, cov_yx.unsqueeze(-1)).squeeze()
                    log_det = torch.log(cov_ygivenx)
                
                    cov_map[(time_idx,index)] = {
                        'tmp1': tmp1,                     # Saves the (3, 3) matrix
                        'L_full': L_full, 
                        'cov_ygivenx': cov_ygivenx, 
                        'cond_mean_tmp': cond_mean_tmp, 
                        'log_det': log_det, 
                        'locs': locs.clone(),             # Saves the (N+1, 3) matrix
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

    def compute_vecc_nll(self, params , covariance_function, cov_map):
        vecc_nll = self.vecchia_oct22(params, covariance_function, cov_map)
        return vecc_nll

class fit_vecchia_adams(VecchiaLikelihood): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    
    def set_optimizer(self, 
                        param_groups, # <--- Renamed argument to be flexible
                        lr=0.01, 
                        betas=(0.9, 0.99), 
                        eps=1e-8, 
                        scheduler_type:str='step', 
                        step_size=40, 
                        gamma=0.5, 
                        T_max=10):

            # Input is a list of parameter groups (dictionaries)
            optimizer = torch.optim.Adam(
                param_groups, 
                lr=lr, # Used as a global default if no lr in groups
                betas=betas, 
                eps=eps
            )
   
            # 2. Scheduler logic remains the same (it needs the optimizer object)
            if scheduler_type.lower() == 'cosine':
                # Cosine Annealing uses T_max
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max) 
            else: 
                # Default to StepLR, using step_size and gamma
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return optimizer, scheduler
    
    def fit_vecc_scheduler_oct23(self, params_list, optimizer, scheduler,  covariance_function, epochs=10):

        grad_tol = 1e-5  # Convergence tolerance for the max absolute gradient
        for epoch in range(epochs):  
            params = torch.cat(params_list)
            # --- FIX: Re-compute cov_map INSIDE the loop ---
            # The cov_map depends on 'params', so it must be re-computed
            # every time 'params' is updated.
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()  
            loss = self.compute_vecc_nll(params, covariance_function, cov_map)
            
            # --- FIX: Removed retain_graph=True ---
            loss.backward()

            # --- NEW: Calculate max absolute gradient (L-infinity norm) ---
            # We calculate this every epoch to check for convergence.
            max_abs_grad = 0.0
            grad_values = []
            for p in params_list:
                if p.grad is not None:
                    # .item() is correct since params are 1-element tensors
                    grad_values.append(abs(p.grad.item())) 
            
            if grad_values: # Avoid error on empty list
                max_abs_grad = max(grad_values)

            # Print gradients and parameters every 10th epoch
            if epoch % 10 == 0:
                print(f'--- Epoch {epoch+1} / Loss: {loss.item():.6f} ---')
                
                # Iterate through the list of parameter tensors to print
                for i, param_tensor in enumerate(params_list):
                    
                    if param_tensor.grad is not None:
                        grad_value = param_tensor.grad.item()
                    else:
                        grad_value = 'N/A' 
                        
                    print(f'  Param {i}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                
                # --- NEW: Print the max grad ---
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30) # Separator for clarity
            
            optimizer.step()  # Update the parameters
            scheduler.step()  # Update the learning rate

            # --- REVISED: Convergence Check ---
            # Check if the largest absolute gradient is below our tolerance.
            # We add 'epoch > 5' to let the optimizer settle down first.
            if epoch > 5 and max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at epoch {epoch}")
                
                # Get final params *after* the step
                final_params_tensor = torch.cat(params_list) 
                print(f'Epoch {epoch+1},  \n vecc Parameters: {final_params_tensor.detach().numpy()}')
                break

            # --- REMOVED: Old loss-based check ---
            # if abs(prev_loss - loss.item()) < tol:
            #     ...
            # prev_loss = loss.item()

        # --- Cleaned up final reporting ---
        final_params_tensor = torch.cat(params_list)
        final_params_list = [p.item() for p in final_params_tensor] # Get final params as a list
        final_loss = loss.item()
        
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {final_loss}, \n vecc Parameters: {final_params_list}')
        return final_params_list + [final_loss], epoch
    



# (Make sure all other necessary imports like VecchiaLikelihood are present)

class fit_vecchia_lbfgs(VecchiaLikelihood): 
    """
    Alternative fitting class that uses the L-BFGS optimizer.
    
    L-BFGS is a quasi-Newton method that often converges in fewer
    iterations (steps) than Adam, but each step is more computationally
    expensive as it involves a line search.
    """
    
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     

    def set_optimizer(self, 
                        param_groups, 
                        lr=1.0, 
                        max_iter=20,
                        max_eval=None,
                        tolerance_grad=1e-7,
                        tolerance_change=1e-9,
                        history_size=100):
        """
        Sets up the L-BFGS optimizer.
        
        Note: Schedulers are generally not used with L-BFGS.
        The 'lr' parameter acts as an initial step length for the line search.
        """

        optimizer = torch.optim.LBFGS(
            param_groups, 
            lr=lr, 
            max_iter=max_iter,       # Max iterations per optimizer step
            max_eval=max_eval,     # Max function evaluations per step
            tolerance_grad=tolerance_grad,   # Internal grad norm tolerance
            tolerance_change=tolerance_change, # Internal param/loss change tolerance
            history_size=history_size
        )
        
        # L-BFGS manages its own step size, so no scheduler is returned
        return optimizer

    def fit_vecc_scheduler_oct23(self, params_list, optimizer, scheduler,  covariance_function, epochs=10):

            grad_tol = 1e-5  # Convergence tolerance for the max absolute gradient
            
            # --- Store final epoch for reporting ---
            final_epoch = epochs - 1 

            for epoch in range(epochs):  
                params = torch.cat(params_list)
                # --- FIX: Re-compute cov_map INSIDE the loop ---
                # The cov_map depends on 'params', so it must be re-computed
                # every time 'params' is updated.
                cov_map = self.cov_structure_saver(params, covariance_function)
                
                optimizer.zero_grad()  
                loss = self.compute_vecc_nll(params, covariance_function, cov_map)
                
                # --- FIX: Removed retain_graph=True ---
                loss.backward()

                # --- NEW: Calculate max absolute gradient (L-infinity norm) ---
                # We calculate this every epoch to check for convergence.
                max_abs_grad = 0.0
                grad_values = []
                for p in params_list:
                    if p.grad is not None:
                        # .item() is correct since params are 1-element tensors
                        grad_values.append(abs(p.grad.item())) 
                
                if grad_values: # Avoid error on empty list
                    max_abs_grad = max(grad_values)

                # Print gradients and parameters every 10th epoch
                if epoch % 10 == 0:
                    print(f'--- Epoch {epoch+1} / Loss: {loss.item():.6f} ---')
                    
                    # Iterate through the list of parameter tensors to print
                    for i, param_tensor in enumerate(params_list):
                        
                        if param_tensor.grad is not None:
                            grad_value = param_tensor.grad.item()
                        else:
                            grad_value = 'N/A' 
                            
                        print(f'  Param {i}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                    
                    # --- NEW: Print the max grad ---
                    print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                    print("-" * 30) # Separator for clarity
                
                optimizer.step()  # Update the parameters
                scheduler.step()  # Update the learning rate

                # --- REVISED: Convergence Check ---
                # Check if the largest absolute gradient is below our tolerance.
                # We add 'epoch > 5' to let the optimizer settle down first.
                if epoch > 5 and max_abs_grad < grad_tol:
                    print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at epoch {epoch}")
                    
                    # Get final params *after* the step
                    final_params_tensor = torch.cat(params_list) 
                    print(f'Epoch {epoch+1},  \n vecc Parameters: {final_params_tensor.detach().numpy()}')
                    
                    final_epoch = epoch # Store the epoch it stopped at
                    break
            
            # --- 💥 UPDATED FINAL REPORTING 💥 ---
            final_params_tensor = torch.cat(params_list).detach()
            final_raw_params_list = [p.item() for p in final_params_tensor] # Get final params as a list
            final_loss = loss.item() # Get the last computed loss
            
            # Convert to Interpretable Parameters
            # (Assumes _convert_raw_params_to_interpretable is in the class)
            interpretable_params = self._convert_raw_params_to_interpretable(final_raw_params_list)
            
            print(f'FINAL STATE: Epoch {final_epoch+1}, Loss: {final_loss}')
            print(f'  Raw (vecc) Parameters: {final_raw_params_list}')
            print(f'  Interpretable Parameters:')
            
            # Pretty-print the dictionary
            if interpretable_params:
                for key, val in interpretable_params.items():
                    print(f'    {key:10s}: {val:.6f}')
            
            # Return the raw params + loss
            return final_raw_params_list + [final_loss], final_epoch


    def fit_vecc_lbfgs(self, 
                         params_list: List[torch.Tensor], 
                         optimizer: torch.optim.LBFGS, 
                         covariance_function: Callable, 
                         max_steps: int = 50):
        """
        Fits the model using L-BFGS.
        
        'max_steps' is the number of L-BFGS optimization steps (not epochs).
        """

        grad_tol = 1e-5  # Outer convergence tolerance
        print("--- Starting L-BFGS Optimization ---")

        # Define the closure function required by L-BFGS
        def closure():
            optimizer.zero_grad()
            params = torch.cat(params_list)
            
            # --- CRITICAL ---
            # The entire computation graph, including the expensive
            # cov_map, MUST be rebuilt inside the closure.
            cov_map = self.cov_structure_saver(params, covariance_function)
            loss = self.compute_vecc_nll(params, covariance_function, cov_map)
            
            loss.backward()
            return loss

        for i in range(max_steps):
            
            # --- Optimizer Step ---
            loss = optimizer.step(closure)
            
            # --- Reporting (inside no_grad to save memory) ---
            max_abs_grad = 0.0
            grad_values = []
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        grad_values.append(abs(p.grad.item())) 
                
                if grad_values:
                    max_abs_grad = max(grad_values)

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                for j, param_tensor in enumerate(params_list):
                    grad_value = param_tensor.grad.item() if param_tensor.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)

            # --- Outer Convergence Check ---
            if max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at step {i+1}")
                break

        # --- 💥 UPDATED FINAL REPORTING 💥 ---
        final_params_tensor = torch.cat(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor] # Get final params as a list
        final_loss = loss.item()
        
        # Convert to Interpretable Parameters
        interpretable_params = self._convert_raw_params_to_interpretable(final_raw_params_list)
        
        print(f'FINAL STATE: Step {i+1}, Loss: {final_loss}')
        print(f'  Raw (vecc) Parameters: {final_raw_params_list}')
        print(f'  Interpretable Parameters:')
        
        # Pretty-print the dictionary
        if interpretable_params:
            for key, val in interpretable_params.items():
                print(f'    {key:10s}: {val:.6f}')
        
        # Return the raw params + loss, as the calling script expects this
        return final_raw_params_list + [final_loss], i
