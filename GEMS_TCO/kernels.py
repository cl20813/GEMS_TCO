# Standard libraries
import logging
import math
import sys
from collections import defaultdict

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

# Type hints
from typing import Callable, Union, Tuple

# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

# Custom imports
# Configure logging to a specific file path
log_file_path = '/home/jl2815/GEMS/logs/fit_st_by_latitude_11_14.log'


class matern_spatio_temporal:               #sigmasq range advec beta  nugget
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
    
    def matern_cov_yx(self,params: Tuple[float,float,float,float,float,float], y_df, x_df) -> pd.DataFrame:
    
        sigmasq, range_lat, range_lon, advec, beta, nugget  = params
            
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
    
    def full_likelihood(self, params: Tuple[float,float,float,float,float,float], input_df, y):
  
        # Compute the covariance matrix from the matern function
        cov_matrix = self.matern_cov_yx(params=params, y_df = input_df, x_df = input_df)
        # Compute the Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        # Solve for the log determinant
        log_det = 2 * np.sum(np.log(np.diagonal(L)))
        locs = np.array(input_df[['Latitude','Longitude']])
        
        tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
        tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y))
        beta = np.linalg.solve(tmp1, tmp2)
        
        mu = np.dot(locs, beta)
        y_mu = y - mu
    
        alpha = np.linalg.solve(L, y_mu)
        quad_form = np.dot(alpha.T, alpha)
        
        # Compute the negative log-likelihood
        n = len(y)
        neg_log_lik = 0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)
        
        return neg_log_lik
    
    def vecchia_likelihood(self, params: Tuple[float,float,float,float,float,float]):
        # initialize negative log-likelihood value
        neg_log_lik = 0
        prev_df = None
        ## likelihood for the first 30 observations
        for time_idx in range(self.number_of_timestamps):
            current_df = self.input_map[self.key_list[time_idx]].reset_index(drop=True)
            cur_heads = current_df.iloc[:31,:]
            neg_log_lik += self.full_likelihood(params,cur_heads, cur_heads["ColumnAmountO3"])

            for index in range(31,self.size_per_hour):

                current_row = current_df .iloc[index:index+1,:]
                current_y = current_row['ColumnAmountO3'].values[0]

                # construct conditioning set on time 0
                
                mm_neighbors = self.nns_map[index]

                past = list(mm_neighbors)

                if past:
                    conditioning_data = current_df.loc[past,: ]
                else:
                    conditioning_data = pd.DataFrame()

                if time_idx >0:
                    if past:
                        past_conditioning_data = prev_df.loc[past,: ]
                    else:
                        past_conditioning_data = pd.DataFrame()
                    conditioning_data = pd.concat((conditioning_data, past_conditioning_data),axis=0)

                df = pd.concat( (current_row, conditioning_data), axis=0)
                y_and_neighbors = df['ColumnAmountO3'].values
                cov_matrix = self.matern_cov_yx(params=params, y_df = df, x_df = df)

                # Regularization:   already did in covariance matrix
                # cov_matrix += nugget * np.eye(cov_matrix.shape[0])

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
            prev_df = current_df
        return neg_log_lik
    
    def mle_parallel(self, key,lat_idx, bounds, initial_params, input_df, mm_cond_number, baseset_from_maxmin, nns_map):
        try:
            logging.info(f"fit_st_bylat_11_16: day {key+1}")
            print(f"fit_st_bylat_11_16: day {key+1}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                initial_params, 
                args=(input_df, mm_cond_number, baseset_from_maxmin, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"Estimated parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number}, bounds={bounds}, smooth={self.smooth}")
            # print(f"Estimated parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number},  bounds={bounds}, smooth={self.smooth}")
            return f"Estimated parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number},  bounds={bounds}, smooth={self.smooth}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
        
    def mle_parallel_nolat_cut(self, key, bounds, initial_params, input_df, mm_cond_number, baseset_from_maxmin, nns_map):
        try:
            logging.info(f"fit_st_nolat_cut_11_16: day {key}")
            print(f"fit_st_nolat_cut_11_16: {key}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                initial_params, 
                args=(input_df, mm_cond_number, baseset_from_maxmin, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"Estimated parameters on day{key},  is : {jitter}, when cond {mm_cond_number}, bounds={bounds}, smooth={self.smooth}")
            
            return f"Estimated parameters on day{key},  is : {jitter}, when cond {mm_cond_number},  bounds={bounds}, smooth={self.smooth}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
        



    
class matern_spatial:
    def __init__(self):
        pass     
      
    def custom_distance(self,u, v):

        d = np.dot(self.sqrt_range_mat, u-v)
        out = np.linalg.norm(d)
        return (out)
    
    def matern_cov_yx(self, params: Tuple[float,float,float,float,float], y_df= None, x_df=None)-> pd.DataFrame:
        sigmasq, range_lat, range_lon, smooth, nugget = params 
        # Validate inputs
        if y_df is None or x_df is None:
            raise ValueError("Both y and x_df must be provided.")

        # Extract values
        x1 = x_df['Longitude'].values
        y1 = x_df['Latitude'].values

        x2 = y_df['Longitude'].values
        y2 = y_df['Latitude'].values

        coords1 = np.stack((x1, y1), axis=-1)
        coords2 = np.stack((x2, y2), axis=-1)

        # Calculate spatial distances using cdist

        sqrt_range_mat = np.diag([ 1/range_lon**0.5, 1/range_lat**0.5])
        self.sqrt_range_mat = sqrt_range_mat

        s_dist = cdist(coords1, coords2, self.custom_distance)
         
        # Initialize the covariance matrix with zeros
        out = s_dist
        
        # Compute the covariance for non-zero distances
        non_zero_indices = s_dist != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = (sigmasq * (2**(1-smooth)) / gamma(smooth) *
                                    (s_dist[non_zero_indices] )**smooth *
                                    kv(smooth, s_dist[non_zero_indices] ))
        out[~non_zero_indices] = sigmasq
        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget
        return pd.DataFrame(out)
        
    def full_likelihood(self, params: Tuple[float,float,float,float,float], input_df, y):
        # Compute the covariance matrix from the matern function
        cov_matrix = self.matern_cov_yx(params=params, y_df = input_df, x_df = input_df)
        # Compute the Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        # Solve for the log determinant
        log_det = 2 * np.sum(np.log(np.diagonal(L)))
        locs = np.array(input_df[['Latitude','Longitude']])
        
        tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
        tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y))
        beta = np.linalg.solve(tmp1, tmp2)
        
        mu = np.dot(locs, beta)
        y_mu = y - mu
    
        alpha = np.linalg.solve(L, y_mu)
        quad_form = np.dot(alpha.T, alpha)
        
        # Compute the negative log-likelihood
        n = len(y)
        neg_log_lik = 0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)
        return neg_log_lik
    
    def vecchia_likelihood(self,params: Tuple[float,float,float, float,float],input_df, mm_cond_number, nns_map):

        # reordered_df['ColumnAmountO3'] = reordered_df['ColumnAmountO3']-np.mean(reordered_df['ColumnAmountO3'])
        neg_log_lik = 0
        ## likelihood for the first 30 observations
        smallset = input_df.iloc[:31,:]
        neg_log_lik += self.full_likelihood(params, smallset, smallset['ColumnAmountO3'])

        for i in range(31,len(input_df)):
            # current_data and conditioning data
            current_data = input_df.iloc[i:i+1,:]
            current_y = current_data['ColumnAmountO3'].values[0]
            mm_past = nns_map[i,:mm_cond_number]
            mm_past = mm_past[mm_past!=-1]
            # mm_past = np.arange(i)

            conditioning_data = input_df.loc[mm_past,: ]
            df = pd.concat( (current_data, conditioning_data), axis=0)
            y_and_neighbors = df['ColumnAmountO3'].values
            cov_matrix = self.matern_cov_yx(params, y_df= df, x_df=df)

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
                           
    def mle_parallel(self, key, bounds, initial_params, input_df, mm_cond_number, nns_map):
        try:
            logging.info(f"fit_spatial_matern,L-BFGS-B, day {key}")
            print(f"fit_purely_space, L-BFGS-B: day {key}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                initial_params, 
                args=(input_df, mm_cond_number, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"Estimated parameters on {key}, is : {jitter}, when cond {mm_cond_number}, bounds={bounds}")
        
            return f"Estimated parameters on {key}, is : {jitter}, when cond {mm_cond_number},  bounds={bounds}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
        
    def mle_parallel_basin(self, key, bounds, initial_params, input_df, mm_cond_number, nns_map, niter=150):
        try:
            logging.info(f"Starting basinhopping optimization for day {key}")
            print(f"Starting basinhopping optimization for day {key}")  # Debugging line

            result = basinhopping(
                func=self.vecchia_likelihood,
                x0=initial_params,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'bounds': bounds,  # Use dynamic bounds
                    'args': (input_df, mm_cond_number, nns_map)
                },
                niter=niter
            )
            optimized_params = result.x
            logging.info(
                f"Estimated parameters on {key}: {optimized_params}, "
                f"cond {mm_cond_number}, bounds={bounds}"
            )
            return f"Estimated parameters on {key}: {optimized_params}, cond {mm_cond_number}, bounds={bounds}"
        except Exception as e:
            logging.error(f"Error occurred on {key}: {str(e)}", exc_info=True)
            return f"Error occurred on {key}"
        
    def mle_parallel_nelder_mead(self, key, bounds, initial_params, input_df, mm_cond_number, nns_map):
        try:
            logging.info(f"Starting Nelder-Mead optimization for day {key}")
            print(f"Starting Nelder-Mead optimization for day {key}")  # Debugging line

            # Define a wrapper to handle bounds for Nelder-Mead
            def constrained_vecchia_likelihood(params, input_df, mm_cond_number, nns_map):
                # Apply manual bounds handling
                for i, (low, high) in enumerate(bounds):
                    if not (low <= params[i] <= high):
                        logging.warning(f"Parameter out of bounds: {params}")
                        return float('inf')  # Penalize out-of-bound values heavily
                # Call the actual likelihood function
                return self.vecchia_likelihood(params, input_df, mm_cond_number, nns_map)

            # Perform optimization with Nelder-Mead
            result = minimize(
                fun=constrained_vecchia_likelihood,
                x0=initial_params,
                args=(input_df, mm_cond_number, nns_map),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'disp': True, 'adaptive': True}  # Enable adaptive step sizes
            )

            # Extract and log results
            optimized_params = result.x
            if result.success:
                logging.info(
                    f"Optimization succeeded for day {key}. "
                    f"Parameters: {optimized_params}, "
                    f"Bounds: {bounds}, Condition number: {mm_cond_number}"
                )
                return f"Optimization succeeded for day {key}. Parameters: {optimized_params}, Bounds: {bounds}"
            else:
                logging.warning(f"Optimization failed for day {key}. Message: {result.message}")
                return f"Optimization failed for day {key}. Message: {result.message}"

        except Exception as e:
            logging.error(f"An error occurred for day {key}: {str(e)}", exc_info=True)
            return f"An error occurred for day {key}: {str(e)}"




class gneiting:
    def __init__(self):
        pass

    def my_gneiting(self, params: Tuple[float,float,float,float,float,float,float], input_df=None)->pd.DataFrame: 
        a, c, tau, alpha,gamma,sigma, beta = params
        nugget = 1
        
        # Convert DataFrame columns into numpy arrays
        x = input_df['Longitude'].values
        y = input_df['Latitude'].values
        t = input_df['Hours_elapsed'].values

        # Efficient distance computation using cdist  (operation is vectorized)
        
        coords = np.stack( (x, y), axis=-1)  # also implemented in C and faster than numpy broadcasting
        s_dist = cdist(coords, coords, 'euclidean')
        t_dist = cdist(t[:, None], t[:, None], 'euclidean')  # Ensure t is a 2D array for cdist

        # Calculate spatial distance. I did sanity check that above gives same result as below:
        # s_dist = np.sqrt((x[:, np.newaxis] - x)**2 + (y[:, np.newaxis] - y)**2)
        # Calculate temporal distance
        # t_dist = np.abs(t[:, np.newaxis] - t)

        # Calculate covariance matrix
        tmp1 = sigma**2 / (a * t_dist**(2 * alpha) + 1)**tau
        tmp2 = np.exp(-c * s_dist**(2 * gamma) / (a * t_dist**(2 * alpha) + 1)**(beta * gamma))
        out = tmp1 * tmp2

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget
        return pd.DataFrame(out)



    def full_likelihood(self, params: Tuple[float,float,float,float,float,float,float], input_df, y):

        # Compute the covariance matrix from the matern function
        cov_matrix = self.my_gneiting(params=params, input_df = input_df)
        # Compute the Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        # Solve for the log determinant
        log_det = 2 * np.sum(np.log(np.diagonal(L)))
        locs = np.array(input_df[['Latitude','Longitude']])
        
        tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
        tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y))
        beta = np.linalg.solve(tmp1, tmp2)
        
        mu = np.dot(locs, beta)
        y_mu = y - mu

        alpha = np.linalg.solve(L, y_mu)
        quad_form = np.dot(alpha.T, alpha)
        
        # Compute the negative log-likelihood
        n = len(y)
        neg_log_lik = 0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)
        
        return neg_log_lik
    
    
    def vecchia_likelihood(self, params: Tuple[float,float,float,float,float,float,float], input_df, mm_cond_number, baseset_from_maxmin, nns_map):
        # initialize negative log-likelihood value
        neg_log_lik = 0
        ## likelihood for the first 30 observations
        smallset = input_df.iloc[:31,:]
        neg_log_lik += self.full_likelihood(params,smallset, smallset["ColumnAmountO3"])

        orbits = input_df['Orbit'].unique()
        orbit_num = len(orbits) 
        obs_per_orbit = int(len(input_df)/orbit_num)
        ''' 
        I plant to group by 8 orbits for each dat so orbit_num 8 
        '''

        for j in range(orbit_num):
            p = j
            for i in range( j*obs_per_orbit, (j+1) * obs_per_orbit ):
                current_data = input_df.iloc[i:i+1,:]
                current_y = current_data['ColumnAmountO3'].values[0]

                # construct conditioning set on time 0
                
                mm_past = nns_map[i%obs_per_orbit,:mm_cond_number]  # array
                mm_past = mm_past[mm_past!=-1]
                past = list(mm_past) + list( np.array(baseset_from_maxmin) + j*obs_per_orbit )    # adjust orbit_num
                while (p<j):
                    past +=  list( np.array(past) + obs_per_orbit)
                    p += 1

                conditioning_data = input_df.loc[past,: ]
                df = pd.concat( (current_data, conditioning_data), axis=0)
                y_and_neighbors = df['ColumnAmountO3'].values
                cov_matrix = self.my_gneiting(params=params, input_df = df)
                
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
    
    def mle_parallel(self, key,lat_idx, bounds, initial_params, input_df, mm_cond_number, baseset_from_maxmin, nns_map):
        try:
            logging.info(f"fit_gneiting: day {key+1}")
            print(f"fit_gneiting: day {key+1}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                initial_params, 
                args=(input_df, mm_cond_number, baseset_from_maxmin, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"fit_gneiting parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number}, bounds={bounds}")
            # print(f"Estimated parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number},  bounds={bounds}, smooth={self.smooth}")
            return f"fit_gneiting parameters on day{key+1}, latitude {lat_idx, lat_idx+1} is : {jitter}, when cond {mm_cond_number},  bounds={bounds}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
