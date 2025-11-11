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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

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
from functools import partial

import torch.optim as optim

sys.path.append("/cache/home/jl2815/tco")

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_25/st_models/log/fit_st_by_latitude_11_14.log'




# (Your other imports like torch.optim go here)

class SpatioTemporalModel:
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
    
    # --- 💥 START REPLACEMENT ---
    # Re-adding the STABLE log-reparam functions 
    # that match your notebook code.
    
    def precompute_coords_aniso_STABLE(self, dist_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Calculates the reparameterized distance 'd' for the log-phi model.
            
            dist_params index map:
            [0] phi3 (theta_3^2)
            [1] phi4 (beta^2)
            [2] advec_lat (v_lat)
            [3] advec_lon (v_lon)
            """
            
            # 1. Unpack distance parameters
            phi3, phi4, advec_lat, advec_lon = dist_params
            
            # 2. Get coordinates for the two sets of points
            x_lat, x_lon, x_t = x[:, 0], x[:, 1], x[:, 3]
            y_lat, y_lon, y_t = y[:, 0], y[:, 1], y[:, 3]

            # 3. Advected coordinates for set U (N, 1)
            u_lat_adv = x_lat - advec_lat * x_t
            u_lon_adv = x_lon - advec_lon * x_t
            u_t = x_t
            
            # 4. Advected coordinates for set V (M, 1)
            v_lat_adv = y_lat - advec_lat * y_t
            v_lon_adv = y_lon - advec_lon * y_t
            v_t = y_t

            # 5. Calculate matrix of differences (N, M)
            delta_lat_adv = u_lat_adv.unsqueeze(1) - v_lat_adv.unsqueeze(0)
            delta_lon_adv = u_lon_adv.unsqueeze(1) - v_lon_adv.unsqueeze(0)
            delta_t       = u_t.unsqueeze(1)       - v_t.unsqueeze(0)

            # 6. Calculate squared distance terms
            dist_sq = (delta_lat_adv.pow(2) * phi3) + delta_lon_adv.pow(2) + (delta_t.pow(2) * phi4)
            
            # 7. Return the final distance 'd'
            distance = torch.sqrt(dist_sq + 1e-8)
            
            return distance
    
    def matern_cov_aniso_STABLE_log_reparam(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Applies log-exp reparameterization.
            This is the function your notebook is calling.
            
            params index map:
            [0] log_phi1
            [1] log_phi2
            [2] log_phi3
            [3] log_phi4
            [4] advec_lat (unconstrained)
            [5] advec_lon (unconstrained)
            [6] log_nugget
            """
            
            # --- A. Recover all parameters ---
            phi1   = torch.exp(params[0])
            phi2   = torch.exp(params[1]) # This is range_inv
            phi3   = torch.exp(params[2]) # This is theta_3^2
            phi4   = torch.exp(params[3]) # This is beta^2
            nugget = torch.exp(params[6])
            
            advec_lat = params[4]
            advec_lon = params[5]
            
            sigmasq = phi1 / phi2  # (sigma^2/range) / (1/range) = sigma^2

            # --- B. Call internal functions ---
            dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
            distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
            
            # --- C. Calculate covariance ---
            # NOTE: This assumes smooth=0.5 (Exponential kernel)
            # C = sigma^2 * exp(-d / range) = sigmasq * exp(-d * range_inv)
            cov = sigmasq * torch.exp(-distance * phi2)

            # 4. Add nugget to the diagonal
            if x.shape[0] == y.shape[0]:
                 cov.diagonal().add_(nugget + 1e-8) 
            
            return cov
    
        # --- 💥 END REPLACEMENT ---
        # The old matern_cov_anisotropy_v05, precompute_coords_anisotropy,
        # and custom_distance_matrix functions are now gone.
        
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
                        
                # The covariance function is the spatio-temporal one
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
                # log_2pi = torch.log(torch.tensor(2 * np.pi, dtype=torch.float64, device=self.device))
                # neg_log_lik_sum += 0.5 * N * log_2pi
                
                # 3. 💥 REVISED: Return the average NLL
                neg_log_lik_avg = neg_log_lik_sum / N
                
                return  neg_log_lik_avg




# Assuming SpatioTemporalModel is defined above with 
# the corrected 'full_likelihood_avg' method from our
# previous discussion.

class VecchiaLikelihood(SpatioTemporalModel):
    def __init__(self, smooth:float, input_map :Dict[str,Any], aggregated_data: torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number)
        self.nheads = nheads

    def _build_conditioning_set(self, time_idx: int, index: int):
        """
        Helper function to build the conditioning set data.
        (This function is unchanged, as its logic is correct)
        """
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

            # --- 💥 CHANGE 1: Loop over ALL time indices ---
            # Was: for time_idx in range(0,3):
            for time_idx in range(0, len(key_list)):
            # --- END CHANGE 1 ---
                if time_idx >= len(key_list):
                    break
                current_np = self.input_map[key_list[time_idx]]

                for index in range(cut_line, self.size_per_hour):
                    
                    aggregated_arr = self._build_conditioning_set(time_idx, index)
                    
                    locs_original = aggregated_arr[:, :2]
                    intercept = torch.ones(locs_original.shape[0], 1, 
                                        device=locs_original.device, 
                                        dtype=torch.float64)
                    locs = torch.cat((intercept, locs_original), dim=1)
                    
                    aggregated_y = aggregated_arr[:, 2] 

                    cov_matrix = covariance_function(params=params, y= aggregated_arr, x= aggregated_arr )
                    
                    try:
                        jitter = torch.eye(cov_matrix.shape[0], device=self.device, dtype=torch.float64) * 1e-6
                        L_full = torch.linalg.cholesky(cov_matrix + jitter)
                    except torch.linalg.LinAlgError:
                        print(f"Warning: Cholesky (full) failed for {(time_idx, index)}")
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
                        print(f"Warning: Cholesky (partial) failed for {(time_idx, index)}")
                        continue 

                    z = torch.cholesky_solve(cov_yx.unsqueeze(-1), L_xx, upper=False)
                    cond_mean_tmp = z.T 
                    cov_ygivenx = sigma - torch.matmul(cond_mean_tmp, cov_yx.unsqueeze(-1)).squeeze()
                    log_det = torch.log(cov_ygivenx)
                
                    cov_map[(time_idx,index)] = {
                        'tmp1': tmp1,
                        'L_full': L_full, 
                        'cov_ygivenx': cov_ygivenx, 
                        'cond_mean_tmp': cond_mean_tmp, 
                        'log_det': log_det, 
                        'locs': locs.clone(),
                    }
            return cov_map

    def vecchia_space_time_fullbatch(self, params: torch.Tensor, covariance_function: Callable, cov_map:Dict[str,Any]) -> torch.Tensor:
            """
            Calculates the TOTAL AVERAGE NLL.
            (This function now correctly uses the specific cov_map 
             for every time_idx).
            """
            cut_line= self.nheads
            key_list = list(self.input_map.keys())
            if not key_list:
                return torch.tensor(0.0, device=self.device, dtype=torch.float64)

            # --- Head Calculation (Unchanged) ---
            heads_data = [self.input_map[key_list[0]][:cut_line,:]]
            for time_idx in range(1, len(self.input_map)):
                tmp = self.input_map[key_list[time_idx]][:cut_line,:]
                heads_data.append(tmp)
            heads = torch.cat(heads_data, dim=0).to(self.device)

            N_head = heads.shape[0]
            if N_head > 0:
                head_nll_avg = self.full_likelihood_avg(params, heads, heads[:, 2], covariance_function)
                head_nll_sum = head_nll_avg * N_head
            else:
                head_nll_sum = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            # --- End Head Calculation ---

            
            # --- Vecchia (Conditional) Calculation ---
            vecchia_nll_sum = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            N_tail = 0
            
            for time_idx in range(0,len(self.input_map)):
        
                for index in range(cut_line, self.size_per_hour):
                    
                    # --- 💥 CHANGE 2: Use the specific map_key for ALL t ---
                    # Was: map_key = (time_idx, index) if time_idx < 2 else (2, index)
                    map_key = (time_idx, index)
                    # --- END CHANGE 2 ---
                    
                    if map_key not in cov_map:
                        continue 
                        
                    # Load all pre-computed structural components
                    tmp1 = cov_map[map_key]['tmp1']
                    L_full = cov_map[map_key]['L_full']
                    locs = cov_map[map_key]['locs']
                    cov_ygivenx = cov_map[map_key]['cov_ygivenx']
                    cond_mean_tmp = cov_map[map_key]['cond_mean_tmp']
                    log_det = cov_map[map_key]['log_det']

                    # Note: We re-build this small array, which is fast.
                    # We don't save it in cov_map because it contains 'y' values.
                    aggregated_arr = self._build_conditioning_set(time_idx, index)
                    aggregated_y = aggregated_arr[:, 2]
                    current_y = aggregated_y[0]
                    mu_neighbors_y = aggregated_y[1:]

                    C_inv_y = torch.cholesky_solve(aggregated_y.unsqueeze(-1), L_full, upper=False)
                    tmp2 = torch.matmul(locs.T, C_inv_y) # (3, 1)
                    
                    try:
                        jitter_beta = torch.eye(tmp1.shape[0], device=self.device, dtype=torch.float64) * 1e-8
                        beta = torch.linalg.solve(tmp1 + jitter_beta, tmp2) 
                    except torch.linalg.LinAlgError:
                        print(f"Warning: Could not solve for beta (Vecchia) at {(time_idx, index)}")
                        continue 
                        
                    mu = torch.matmul(locs, beta).squeeze() # (N+1,)
                    mu_current = mu[0]
                    mu_neighbors = mu[1:]
                    
                    cond_mean = mu_current + torch.matmul(cond_mean_tmp, (mu_neighbors_y - mu_neighbors).unsqueeze(-1)).squeeze()
                    alpha = current_y - cond_mean
                    quad_form = alpha**2 * (1 / cov_ygivenx)
                
                    vecchia_nll_sum += 0.5 * (log_det + quad_form)
                    N_tail += 1
            # --- End Vecchia Calculation ---

            
            # --- Final Calculation (Unchanged) ---
            total_nll_sum = head_nll_sum + vecchia_nll_sum
            total_points = N_head + N_tail
            
            if total_points == 0:
                return torch.tensor(0.0, device=self.device, dtype=torch.float64)

            return total_nll_sum / total_points

    def compute_vecc_nll(self, params , covariance_function, cov_map):
        vecc_nll = self.vecchia_space_time_fullbatch(params, covariance_function, cov_map)
        return vecc_nll

class fit_vecchia_adams(VecchiaLikelihood): 
    def __init__(self, smooth:float, input_map:Dict[str,Any], aggregated_data:torch.Tensor, nns_map:Dict[str,Any], mm_cond_number:int, nheads:int):
        super().__init__(smooth, input_map, aggregated_data, nns_map, mm_cond_number, nheads)
     
    
    def set_optimizer(self, 
                        param_groups, 
                        lr=0.01, 
                        betas=(0.9, 0.99), 
                        eps=1e-8, 
                        scheduler_type:str='plateau', # <-- Updated default
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

    
    def fit_model(self, params_list, optimizer, scheduler,  covariance_function, epochs=100):
        """
        Fits the model using Adam and a scheduler.
        Handles schedulers that do and do not require the loss.
        """
        grad_tol = 1e-5  # Convergence tolerance
        
        for epoch in range(epochs):  
            params = torch.cat(params_list)
            
            # Must be re-computed every epoch
            cov_map = self.cov_structure_saver(params, covariance_function)
            
            optimizer.zero_grad()  
            # Uses the wrapper from the base class
            loss = self.compute_vecc_nll(params, covariance_function, cov_map)
            
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
                # 💥 This will now print 7 parameters
                for i, param_tensor in enumerate(params_list):
                    grad_value = param_tensor.grad.item() if param_tensor.grad is not None else 'N/A' 
                    print(f'  Param {i}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)
            
            optimizer.step()
            
            # --- 💥 CRITICAL SCHEDULER CHANGE ---
            # Plateau needs the loss, others don't.
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
            # --- END CHANGE ---

            if epoch > 5 and max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at epoch {epoch}")
                break

        # --- Final Reporting ---
        final_params_tensor = torch.cat(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()

        # Convert to Interpretable Parameters
        interpretable_params = self._convert_raw_params_to_interpretable(final_raw_params_list)
        
        print(f'FINAL STATE: Epoch {epoch+1}, Loss: {final_loss}')
        print(f'  Raw (vecc) Parameters: {final_raw_params_list}')
        print(f'  Interpretable Parameters:')
        
        if interpretable_params:
            for key, val in interpretable_params.items():
                print(f'    {key:10s}: {val:.6f}')
        
        return final_raw_params_list + [final_loss], epoch


    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """Converts the 7 raw optimized parameters back to interpretable model parameters."""
        try:
            # Unpack the 7 raw parameters
            log_phi1 = raw_params_list[0]
            log_phi2 = raw_params_list[1]
            log_phi3 = raw_params_list[2]
            log_phi4 = raw_params_list[3]
            advec_lat = raw_params_list[4]
            advec_lon = raw_params_list[5]
            log_nugget = raw_params_list[6]

            # Exponentiate to get 'phi' values
            phi1 = np.exp(log_phi1)
            phi2 = np.exp(log_phi2)
            phi3 = np.exp(log_phi3)
            phi4 = np.exp(log_phi4)
            nugget = np.exp(log_nugget)
            
            # Convert to interpretable parameters
            range_lon = 1.0 / phi2
            sigmasq = phi1 / phi2 
            range_lat = range_lon / np.sqrt(phi3)
            range_time = 1/(np.sqrt(phi4) * phi2) # time range

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "range_time": range_time,
                "advec_lat": advec_lat,
                "advec_lon": advec_lon,
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
        """
        Sets up the L-BFGS optimizer.
        """

        optimizer = torch.optim.LBFGS(
            param_groups, 
            lr=lr, 
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size
        )
        
        # L-BFGS manages its own step size, so no scheduler is returned
        return optimizer


    def _convert_raw_params_to_interpretable(self, raw_params_list: List[float]) -> Dict[str, float]:
        """Converts the 7 raw optimized parameters back to interpretable model parameters."""
        try:
            # Unpack the 7 raw parameters
            log_phi1 = raw_params_list[0]
            log_phi2 = raw_params_list[1]
            log_phi3 = raw_params_list[2]
            log_phi4 = raw_params_list[3]
            advec_lat = raw_params_list[4]
            advec_lon = raw_params_list[5]
            log_nugget = raw_params_list[6]

            # Exponentiate to get 'phi' values
            phi1 = np.exp(log_phi1)
            phi2 = np.exp(log_phi2)
            phi3 = np.exp(log_phi3)
            phi4 = np.exp(log_phi4)
            nugget = np.exp(log_nugget)
            
            # Convert to interpretable parameters
            range_lon = 1.0 / phi2
            sigmasq = phi1 / phi2 
            range_lat = range_lon / np.sqrt(phi3)
            range_time = 1/(np.sqrt(phi4) * phi2) # time range

            return {
                "sigma_sq": sigmasq,
                "range_lon": range_lon,
                "range_lat": range_lat,
                "range_time": range_time,
                "advec_lat": advec_lat,
                "advec_lon": advec_lon,
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
        """

        grad_tol = 1e-5  # Outer convergence tolerance
        print("--- Starting L-BFGS Optimization ---")

        def closure():
            optimizer.zero_grad()
            params = torch.cat(params_list)
            
            # --- CRITICAL ---
            # cov_map MUST be rebuilt inside the closure.
            cov_map = self.cov_structure_saver(params, covariance_function)
            loss = self.compute_vecc_nll(params, covariance_function, cov_map)
            
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
                # 💥 This will now print 7 parameters
                for j, param_tensor in enumerate(params_list):
                    grad_value = param_tensor.grad.item() if param_tensor.grad is not None else 'N/A'
                    print(f'  Param {j}: Value={param_tensor.item():.4f}, Grad={grad_value}')
                print(f'  Max Abs Grad: {max_abs_grad:.6e}') 
                print("-" * 30)

            if max_abs_grad < grad_tol:
                print(f"\nConverged on gradient norm (max|grad| < {grad_tol}) at step {i+1}")
                break

        # --- Final Reporting ---
        final_params_tensor = torch.cat(params_list).detach()
        final_raw_params_list = [p.item() for p in final_params_tensor]
        final_loss = loss.item()
        
        interpretable_params = self._convert_raw_params_to_interpretable(final_raw_params_list)
        
        print(f'FINAL STATE: Step {i+1}, Loss: {final_loss}')
        print(f'  Raw (vecc) Parameters: {final_raw_params_list}')
        print(f'  Interpretable Parameters:')
        
        if interpretable_params:
            for key, val in interpretable_params.items():
                print(f'    {key:10s}: {val:.6f}')
        
        return final_raw_params_list + [final_loss], i
