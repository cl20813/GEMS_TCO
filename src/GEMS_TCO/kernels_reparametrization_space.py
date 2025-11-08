
# This module contains the spatial-only model classes and methods.
# The parameter we reparameterize are:

# - phi1: The variance parameter divided by the longitude range
# - phi2: The spatial range in the longitude direction
# - phi3: The spatial smoothness parameter
# - nugget: The nugget effect (small-scale variability)
# The temporal parameters have been removed.


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

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch.optim as optim
import torch.optim as optim
import torch.nn as nn
from scipy.interpolate import splrep, splev

# Fit your "spline" by just storing the x and y
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from typing import Dict, Any, Callable, List, Tuple, Optional

import time
import copy    
import logging     # for logging
# Add your custom path
import torch.utils.data as data
import warnings
import torch
from functools import partial
# --- MODIFIED 'optimizer_fun' IN 'model_fitting' CLASS ---
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # Need to import this
# ... (other imports) ...

sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Callable, List
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from scipy.special import gamma # Make sure to import gamma

# (Your other imports like torch.optim go here)

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from collections import defaultdict
from typing import Dict, Any, Callable, List
import numpy as np
from math import gamma # Added for the __init__

# Ensure all necessary imports are present at the top
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from scipy.special import gamma # A common source for this
import torch.optim as optim


class SpatialModel:
    """
    Base class for the spatial-only model.
    All spatio-temporal logic has been removed.
    """
    def __init__(self, smooth:float, input_map: Dict[str, Any], aggregated_data: torch.Tensor, nns_map:Dict[str, Any], mm_cond_number: int):
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
        # We only need lat, lon, response, and a time-key.
        # Assuming col 0:lat, 1:lon, 2:response, 3:time_id
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
    
    
    # --- 💥 NEW SPATIAL COVARIANCE FUNCTION 💥 ---
    def matern_cov_SPATIAL_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        This is the new "bridge" function for the spatial-only model.
        It implements the 4-parameter reparameterization.
        
        params index map:
        [0] log_phi1 (log(sigmasq / range_lon))
        [1] log_phi2 (log(1 / range_lon))
        [2] log_phi3 (log((range_lon / range_lat)^2))
        [3] log_nugget
        """
        
        # --- A. Recover all parameters ---
        # Unpack the 4 raw log-parameters
        phi1   = torch.exp(params[0])
        phi2   = torch.exp(params[1]) # This is range_lon_inv
        phi3   = torch.exp(params[2]) # This is (range_lon / range_lat)^2
        nugget = torch.exp(params[3])

        # --- B. Derive Physical Parameters ---
        # sigmasq = (sigmasq / range_lon) / (1 / range_lon)
        sigmasq = phi1 / phi2  
        
        # range_lon_inv is just phi2
        range_lon_inv = phi2
        
        # range_lat_inv = sqrt(phi3) * range_lon_inv
        range_lat_inv = torch.sqrt(phi3) * phi2

        # --- C. Calculate Anisotropic Spatial Distance ---
        # x and y are (N, 4) and (M, 4) tensors [lat, lon, response, time_id]
        x_lat, x_lon = x[:, 0], x[:, 1]
        y_lat, y_lon = y[:, 0], y[:, 1]
        
        delta_lat = x_lat.unsqueeze(1) - y_lat.unsqueeze(0)
        delta_lon = x_lon.unsqueeze(1) - y_lon.unsqueeze(0)

        # d^2 = (delta_lat / range_lat)^2 + (delta_lon / range_lon)^2
        # d^2 = (delta_lat * range_lat_inv)^2 + (delta_lon * range_lon_inv)^2
        dist_sq = (delta_lat * range_lat_inv).pow(2) + (delta_lon * range_lon_inv).pow(2)
        
        # d is the dimensionless distance
        distance = torch.sqrt(dist_sq + 1e-8) 

        # --- D. Calculate Covariance (Matern 0.5 = Exponential) ---
        # C = sigmasq * exp(-d)
        cov = sigmasq * torch.exp(-distance)

        # --- E. Add Nugget ---
        # Add nugget only if it's a self-covariance matrix
        if x.shape[0] == y.shape[0]:
             cov.diagonal().add_(nugget + 1e-8) # Add jitter to nugget
        
        return cov
    
    # --- END NEW COVARIANCE FUNCTION ---
    
    
    def full_likelihood_avg(self, params: torch.Tensor, input_data: torch.Tensor, y: torch.Tensor, covariance_function: Callable) -> torch.Tensor:
            """
            Calculates the AVERAGE Negative Log-Likelihood (NLL) per data point.
            Includes an intercept (spatial trend).
            """
            input_data = input_data.to(torch.float64)
            y = y.to(torch.float64)
            
            # Get N (number of observations)
            N = input_data.shape[0]
            if N == 0:
                return torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    
            # The covariance function is now the new spatial-only one
            cov_matrix = covariance_function(params=params, y= input_data, x= input_data)
            
            try:
                jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                L = torch.linalg.cholesky(cov_matrix + jitter)
            except torch.linalg.LinAlgError:
                print("Warning: Cholesky decomposition failed.")
                return torch.tensor(torch.inf, device=params.device, dtype=params.dtype)
            
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            
            # --- Spatial Trend (Intercept + Lat + Lon) ---
            locs_original = input_data[:,:2].to(torch.float64) # [lat, lon]
            intercept = torch.ones(locs_original.shape[0], 1, 
                                device=locs_original.device, 
                                dtype=torch.float64)
            locs = torch.cat((intercept, locs_original), dim=1) # [1, lat, lon]
            # --- End Trend ---
            
            if y.dim() == 1:
                y_col = y.unsqueeze(-1).to(torch.float64)
            else:
                y_col = y.to(torch.float64)

            C_inv_X = torch.cholesky_solve(locs, L, upper=False)  # (N, 3)
            C_inv_y = torch.cholesky_solve(y_col, L, upper=False) # (N, 1)

            tmp1 = torch.matmul(locs.T, C_inv_X)       # (3, 3)
            tmp2 = torch.matmul(locs.T, C_inv_y)       # (3, 1)
            
            try:
                jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2) # Solves for [b0, b1, b2]
            except torch.linalg.LinAlgError:
                print("Warning: Could not solve for beta. X^T C_inv X may be singular.")
                return torch.tensor(torch.inf, device=locs.device, dtype=locs.dtype)

            mu = torch.matmul(locs, beta) # (N, 1)
            y_mu = y_col - mu
            
            C_inv_y_mu = torch.cholesky_solve(y_mu, L, upper=False)
            quad_form = torch.matmul(y_mu.T, C_inv_y_mu) 

            # --- NLL Calculation ---
            
            # 1. Core NLL (Log-det + Quad-form)
            neg_log_lik_sum = 0.5 * (log_det + quad_form.squeeze())
            
            # 2. (Optional) Add constant term for the "true" NLL
            # If you want the true average NLL, uncomment the next two lines.
            # If you only want the average core NLL (for optimization comparison), leave them commented.
            
            # log_2pi = torch.log(torch.tensor(2 * np.pi, dtype=torch.float64, device=self.device))
            # neg_log_lik_sum += 0.5 * N * log_2pi
            
            # 3. 💥 REVISED: Return the average NLL
            neg_log_lik_avg = neg_log_lik_sum / N
            
            return  neg_log_lik_avg


# 💥 Fixed inheritance to point to the renamed 'SpatialModel'
class VecchiaLikelihood(SpatialModel):
    def __init__(self, smooth:float, input_map :Dict[str,Any], aggregated_data: torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        # 💥 Fixed inheritance
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number) 
        self.nheads = nheads

    def _build_conditioning_set(self, time_idx: int, index: int):
        """
        Helper function to build the conditioning set data.
        💥 NOW PURELY SPATIAL: Ignores past time steps.
        """
        current_np = self.input_map[self.key_list[time_idx]]
        current_row = current_np[index].reshape(1, -1)
        
        mm_neighbors = self.nns_map[index]
        past = list(mm_neighbors) 
        data_list = []

        # Only add neighbors from the *current* time step
        if past:
            data_list.append(current_np[past])

        # --- 💥 REMOVED all time_idx > 0 logic ---

        if data_list:
            conditioning_data = torch.vstack(data_list)
        else:
            conditioning_data = torch.empty((0, current_row.shape[1]), device=self.device, dtype=torch.float64)
            
        aggregated_arr = torch.vstack((current_row, conditioning_data))
        return aggregated_arr.to(dtype=torch.float64)
    
    def cov_structure_saver(self, params: torch.Tensor, covariance_function: Callable) -> Dict[str,Any]:
            """
            Optimized version. Pre-computes all expensive parts.
            💥 NOW TIME-INVARIANT: Computes structure only once.
            """
            cov_map = defaultdict(dict)
            cut_line= self.nheads
            
            # We only need the structure from one time step (e.g., t=0)
            # since the spatial relationships are stationary.
            time_idx = 0 
            
            for index in range(cut_line, self.size_per_hour):
                
                # This is now purely spatial
                aggregated_arr = self._build_conditioning_set(time_idx, index)
                
                # --- Spatial Trend (Intercept + Lat + Lon) ---
                locs_original = aggregated_arr[:, :2]
                intercept = torch.ones(locs_original.shape[0], 1, 
                                    device=locs_original.device, 
                                    dtype=torch.float64)
                locs = torch.cat((intercept, locs_original), dim=1)
                # --- End Trend ---
                
                aggregated_y = aggregated_arr[:, 2] 

                cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                
                try:
                    jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                    L_full = torch.linalg.cholesky(cov_matrix + jitter)
                except torch.linalg.LinAlgError:
                    print(f"Warning: Cholesky (full) failed for index {index}")
                    continue
                    
                C_inv_locs = torch.cholesky_solve(locs, L_full, upper=False)
                tmp1 = torch.matmul(locs.T, C_inv_locs) # tmp1 is now (3, 3)
            
                sigma = cov_matrix[0, 0]
                cov_yx = cov_matrix[0, 1:]
                cov_xx = cov_matrix[1:, 1:]
                
                try:
                    jitter_xx = torch.eye(cov_xx.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                    L_xx = torch.linalg.cholesky(cov_xx + jitter_xx)
                except torch.linalg.LinAlgError:
                    print(f"Warning: Cholesky (partial) failed for index {index}")
                    continue 

                z = torch.cholesky_solve(cov_yx.unsqueeze(-1), L_xx, upper=False)
                cond_mean_tmp = z.T 
                cov_ygivenx = sigma - torch.matmul(cond_mean_tmp, cov_yx.unsqueeze(-1)).squeeze()
                log_det = torch.log(cov_ygivenx)
            
                # 💥 Key is now just the spatial index
                cov_map[index] = {
                    'tmp1': tmp1,                     # Saves the (3, 3) matrix
                    'L_full': L_full, 
                    'cov_ygivenx': cov_ygivenx, 
                    'cond_mean_tmp': cond_mean_tmp, 
                    'log_det': log_det, 
                    'locs': locs.clone(),             # Saves the (N+1, 3) matrix
                }
            return cov_map

    def vecchia_space_fullbatch(self, params: torch.Tensor, covariance_function: Callable, cov_map:Dict[str,Any]) -> torch.Tensor:
        """
        Optimized version.
        💥 NOW SPATIAL-ONLY: Treats each time step as an independent
        realization of the spatial process.
        
        💥 REVISED: Returns the AVERAGE NLL per observation for the full dataset (all time steps).
        """
        cut_line= self.nheads
        key_list = list(self.input_map.keys())
        T_temporal = len(key_list)
        if T_temporal == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        # This will accumulate the total SUM of the core NLL
        neg_log_lik_SUM = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        # 💥 Loop over all time steps
        for time_idx in range(T_temporal):
    
            # --- Head Calculation (for this time step) ---
            head_data = self.input_map[key_list[time_idx]][:cut_line,:]
            N_head = head_data.shape[0]

            if N_head > 0:
                # 1. 💥 Call the AVG function (from full_likelihood_avg.py)
                # We assume self.full_likelihood is now self.full_likelihood_avg
                avg_head_nll = self.full_likelihood_avg(params, head_data, head_data[:, 2], covariance_function) 
                
                # 2. 💥 Convert AVG back to SUM for accumulation
                # (avg_nll * N_head) = sum_nll
                neg_log_lik_SUM += avg_head_nll * N_head
            # --- End Head Calculation ---

            # --- Conditional Calculation (for this time step) ---
            for index in range(cut_line, self.size_per_hour):
                
                # 💥 Load the time-invariant structure using just the index
                if index not in cov_map:
                    continue 
                    
                # Load all pre-computed structural components
                tmp1 = cov_map[index]['tmp1']
                L_full = cov_map[index]['L_full']
                locs = cov_map[index]['locs']
                cov_ygivenx = cov_map[index]['cov_ygivenx']
                cond_mean_tmp = cov_map[index]['cond_mean_tmp']
                log_det = cov_map[index]['log_det']

                # 💥 Get the actual data for this time_idx and index
                aggregated_arr = self._build_conditioning_set(time_idx, index)
                aggregated_y = aggregated_arr[:, 2]
                current_y = aggregated_y[0]
                mu_neighbors_y = aggregated_y[1:]

                C_inv_y = torch.cholesky_solve(aggregated_y.unsqueeze(-1), L_full, upper=False)
                tmp2 = torch.matmul(locs.T, C_inv_y) # (3, N+1) @ (N+1, 1) = (3, 1)
                
                try:
                    jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                    beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2) # Solves (3, 3) system
                except torch.linalg.LinAlgError:
                    print(f"Warning: Could not solve for beta (Vecchia) at {(time_idx, index)}")
                    continue 
                    
                mu = torch.matmul(locs, beta).squeeze() # (N+1, 3) @ (3, 1) = (N+1,)
                mu_current = mu[0]
                mu_neighbors = mu[1:]
                
                cond_mean = mu_current + torch.matmul(cond_mean_tmp, (mu_neighbors_y - mu_neighbors).unsqueeze(-1)).squeeze()
                alpha = current_y - cond_mean
                quad_form = alpha**2 * (1 / cov_ygivenx)
            
                # 3. 💥 Add the conditional sum
                neg_log_lik_SUM += 0.5 * (log_det + quad_form)

        # --- Final Averaging ---
        N_spatial = self.size_per_hour
        N_total = N_spatial * T_temporal
        
        if N_total == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
            
        # 4. 💥 Return the average NLL per observation
        neg_log_lik_AVG = neg_log_lik_SUM / N_total
        
        return neg_log_lik_AVG
    def compute_vecc_nll_fullbatch(self, params , covariance_function, cov_map):
        vecc_nll = self.vecchia_space_fullbatch(params, covariance_function, cov_map)
        return vecc_nll

class fit_vecchia_adams_fullbatch(VecchiaLikelihood): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    
    # ... inside your class (e.g., fit_vecchia_adams_minibatch) ...

    def set_optimizer(self, 
                        param_groups, 
                        lr=0.01, 
                        betas=(0.9, 0.99), 
                        eps=1e-8, 
                        scheduler_type:str='plateau', # <-- Changed default
                        step_size=40, 
                        gamma=0.5, 
                        T_max=10,
                        patience=5,      # <-- New param for Plateau
                        factor=0.5):     # <-- New param for Plateau

            optimizer = optim.Adam(
                param_groups, 
                lr=lr,
                betas=betas, 
                eps=eps
            )

            if scheduler_type.lower() == 'plateau':
                # 💡 Monitors the loss and reduces LR if it stops improving
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',      # We want to minimize the NLL
                    factor=factor,   # Amount to reduce LR by (e.g., 0.5 = 50% cut)
                    patience=patience, # How many epochs to wait for improvement
                    verbose=True     # Prints a message when LR is reduced
                )
            elif scheduler_type.lower() == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max) 
            else: 
                # Default to StepLR if 'step' or anything else is passed
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
                
            return optimizer, scheduler

    
    def fit_vecc_scheduler_fullbatch(self, params_list, optimizer, scheduler,  covariance_function, epochs=10):
        """
        Fits the model using Adam.
        The params_list should now be 4 elements.
        """
        grad_tol = 1e-5  # Convergence tolerance
        for epoch in range(epochs):  
            params = torch.cat(params_list)
            
            # This is now the efficient time-invariant version
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()  
            # This now loops over all time internally
            loss = self.compute_vecc_nll_fullbatch(params, covariance_function, cov_map)
            
            loss.backward()

            max_abs_grad = 0.0
            grad_values = []
            for p in params_list:
                if p.grad is not None:
                    grad_values.append(abs(p.grad.item())) 
            
            if grad_values:
                max_abs_grad = max(grad_values)

            if epoch % 10 == 0:
                print(f'--- Epoch {epoch+1} / Loss: {loss.item():.6f} ---')
                # 💥 This will now print 4 parameters
                for i, param_tensor in enumerate(params_list):
                    
                    if param_tensor.grad is not None:
                        grad_value = param_tensor.grad.item()
                    else:
                        grad_value = 'N/A' 
                        
                    print(f'  Param {i}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)
            
            optimizer.step()
            scheduler.step(loss)

            if epoch > 5 and max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at epoch {epoch}")
                final_params_tensor = torch.cat(params_list) 
                print(f'Epoch {epoch+1},  \n vecc Parameters: {final_params_tensor.detach().numpy()}')
                break

        final_params_tensor = torch.cat(params_list)
        final_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {final_loss}, \n vecc Parameters: {final_params_list}')
        return final_params_list + [final_loss], epoch

    # --- 💥 NEW HELPER METHOD (SPATIAL-ONLY) 💥 ---
    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """Converts the 4 raw optimized parameters back to interpretable model parameters."""
        try:
            # Unpack the 4 raw (log-space) parameters
            log_phi1 = raw_params_list[0]
            log_phi2 = raw_params_list[1]
            log_phi3 = raw_params_list[2]
            log_nugget = raw_params_list[3]

            # Exponentiate to get 'phi' values
            phi1 = np.exp(log_phi1)
            phi2 = np.exp(log_phi2)
            phi3 = np.exp(log_phi3)
            nugget = np.exp(log_nugget)
            
            # Convert to interpretable parameters
            
            # phi2 = 1 / range_lon  =>  range_lon = 1 / phi2
            range_lon = 1.0 / phi2
            
            # phi1 = sigmasq * phi2 (or sigmasq / range_lon) =>  sigmasq = phi1 / phi2
            sigmasq = phi1 / phi2 
            
            # phi3 = (range_lon / range_lat)^2 => sqrt(phi3) = range_lon / range_lat
            # => range_lat = range_lon / sqrt(phi3)
            range_lat = range_lon / np.sqrt(phi3)
            
            # beta and advection are gone

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "nugget": nugget
            }
        except Exception as e:
            print(f"\nWarning: Could not convert raw params. Error: {e}")
            return {}


class fit_vecchia_lbfgs(VecchiaLikelihood): 
    """
    Alternative fitting class that uses the L-BFGS optimizer.
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

        optimizer = torch.optim.LBFGS(
            param_groups, 
            lr=lr, 
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size
        )
        
        return optimizer

    # --- 💥 NEW HELPER METHOD (SPATIAL-ONLY) 💥 ---
    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """Converts the 4 raw optimized parameters back to interpretable model parameters."""
        try:
            # Unpack the 4 raw (log-space) parameters
            log_phi1 = raw_params_list[0]
            log_phi2 = raw_params_list[1]
            log_phi3 = raw_params_list[2]
            log_nugget = raw_params_list[3]

            # Exponentiate to get 'phi' values
            phi1 = np.exp(log_phi1)
            phi2 = np.exp(log_phi2)
            phi3 = np.exp(log_phi3)
            nugget = np.exp(log_nugget)
            
            # Convert to interpretable parameters
            range_lon = 1.0 / phi2
            sigmasq = phi1 / phi2 
            range_lat = range_lon / np.sqrt(phi3)

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "nugget": nugget
            }
        except Exception as e:
            print(f"\nWarning: Could not convert raw params. Error: {e}")
            return {}

    def fit_vecc_lbfgs(self, 
                         params_list: List[torch.Tensor], 
                         optimizer: torch.optim.LBFGS, 
                         covariance_function: Callable, 
                         max_steps: int = 50):
        """
        Fits the model using L-BFGS.
        The params_list should now be 4 elements.
        """

        grad_tol = 1e-5  # Outer convergence tolerance
        print("--- Starting L-BFGS Optimization ---")

        def closure():
            optimizer.zero_grad()
            params = torch.cat(params_list)
            
            # --- CRITICAL ---
            # Rebuilt inside the closure
            cov_map = self.cov_structure_saver(params, covariance_function)
            loss = self.compute_vecc_nll_fullbatch(params, covariance_function, cov_map)
            
            loss.backward()
            return loss

        for i in range(max_steps):
            
            loss = optimizer.step(closure)
            
            max_abs_grad = 0.0
            grad_values = []
            with torch.no_grad():
                for p in params_list:
                    if p.grad is not None:
                        grad_values.append(abs(p.grad.item())) 
                
                if grad_values:
                    max_abs_grad = max(grad_values)

                print(f'--- Step {i+1}/{max_steps} / Loss: {loss.item():.6f} ---')
                # 💥 This will now print 4 parameters
                for j, param_tensor in enumerate(params_list):
                    grad_value = param_tensor.grad.item() if param_tensor.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)

            if max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at step {i+1}")
                break

        # --- 💥 UPDATED FINAL REPORTING 💥 ---
        final_params_tensor = torch.cat(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        
        # Convert to Interpretable Parameters
        interpretable_params = self._convert_raw_params_to_interpretable(final_raw_params_list)
        
        print(f'FINAL STATE: Step {i+1}, Loss: {final_loss}')
        print(f'  Raw (vecc) Parameters: {final_raw_params_list}')
        print(f'  Interpretable Parameters:')
        
        if interpretable_params:
            for key, val in interpretable_params.items():
                print(f'    {key:10s}: {val:.6f}')
        
        return final_raw_params_list + [final_loss], i
