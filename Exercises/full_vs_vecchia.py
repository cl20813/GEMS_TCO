# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict
import math
from scipy.special import gamma  # better than math.gamma(v)
# from skgstat import Variogram

from typing import Callable   # nearest neighbor function input type
import sklearn.neighbors  # nearest neighbor

from sklearn.neighbors import BallTree # for space_center function


import scipy
from scipy.stats import multivariate_normal #simulation
from scipy.optimize import minimize
from scipy.special import kv                # bessel function


import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import sys

from scipy.spatial.distance import cdist # for space and time distance
from scipy.spatial import distance # find closest spatial point
from typing import Callable   # find_nns_naive input type


from GEMS_TCO import orbitmap
# from GEMS_TCO import smoothspace
from GEMS_TCO.smoothspace import space_average

import argparse


def maxmin_naive(dist: np.ndarray, first: np.intp) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs min-max ordering

    The implementation is naive and will not perform well for large inputs.

    Parameters
    ----------
    dist
        distance matrix
    first
        Index of the observation that should be sorted first

    Returns
    -------
    np.ndarray
        The minmax ordering
    np.ndarray
        Array with the distances to the location preceding in ordering
    """

    n = dist.shape[0]
    ord = np.zeros(n, dtype=np.int64)
    ord[0] = first
    dists = np.zeros(n)
    dists[0] = np.nan
    idx = np.arange(n)


    for i in range(1, n):
        # find min dist for each not selected loccation
        mask = ~np.isin(idx, ord[:i])
        min_d = np.min(dist[mask, :][:, ord[:i]], axis=1)

        # find max idx among those
        idx_max = np.argmax(min_d)

        # record dist
        dists[i] = min_d[idx_max]

        # adjust idx for the prevous removed rows
        idx_max = idx[mask][idx_max]
        ord[i] = idx_max
    return ord, dists


def find_nns_naive(
    locs: np.ndarray, dist_fun: Callable | str = "euclidean", max_nn: int = 10, **kwargs
) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering.

    The method is naivly implemented and will not perform well for large inputs.

    Parameters
    ----------
    locs
        an n x m array of ordered locations
    dist_fun
        a distance function
    max_nn
        number of nearest neighbours
    kwargs
        supplied dist_func

    Returns
    -------
    np.ndarray
        Returns an n x max_nn array holding the indices of the nearest neighbors
        preceding in the ordering where -1 indicates missing neighbors.
    """

    n = locs.shape[0]
    nns = np.zeros((n, max_nn), dtype=np.int64) - 1
    for i in range(1, n):
        nn = sklearn.neighbors.BallTree(locs[:i], metric=dist_fun, **kwargs)   # dist_fun= 'euclidean'
        k = min(i-1, max_nn)
        nn_res = nn.query(locs[[i], :], k=k, return_distance=False)
        nns[i, :k] = nn_res
    return nns

def matern_cov_spat(sigmasq: float = 1, range_: float = 1, v: float = 0.5, input_df: pd.DataFrame = None) -> pd.DataFrame:
    x = input_df['Longitude'].values
    y = input_df['Latitude'].values
    d = np.sqrt((x[:, np.newaxis] - x)**2 + (y[:, np.newaxis] - y)**2)
    abs_d = np.abs(d)
    
    # Initialize the covariance matrix with zeros
    out = np.zeros_like(d)
    
    # Compute the covariance for non-zero distances
    non_zero_indices = abs_d != 0
    if np.any(non_zero_indices):
        out[non_zero_indices] = (sigmasq * (2**(1-v)) / math.gamma(v) *
                                 (abs_d[non_zero_indices] / range_)**v *
                                 kv(v, abs_d[non_zero_indices] / range_))
    
    # Fill the diagonal with sigmasq (variance term)
    np.fill_diagonal(out, sigmasq)
    return pd.DataFrame(out)


def gneiting_xy(a, c, tau, alpha,gamma,sigma, beta, x_df=None, y_df2= None)-> pd.DataFrame:
    
    # Extract values
    x1 = x_df['Longitude'].values
    y1 = x_df['Latitude'].values
    t1 = x_df['Hours_elapsed'].values

    x2 = y_df2['Longitude'].values
    y2 = y_df2['Latitude'].values
    t2 = y_df2['Hours_elapsed'].values
    
    coords1 = np.stack((x1, y1), axis=-1)
    coords2 = np.stack((x2, y2), axis=-1)

    # Calculate spatial distances using cdist
    s_dist = cdist(coords1, coords2, 'euclidean')

    # Ensure temporal arrays are 2D for cdist
    t1_2d = t1[:, np.newaxis]
    t2_2d = t2[:, np.newaxis]

    # Calculate temporal distances using cdist
    t_dist = cdist(t1_2d, t2_2d, 'euclidean')
    

    # Calculate covariance matrix
    tmp1 = sigma**2 / (a * t_dist**(2 * alpha) + 1)**tau
    tmp2 = np.exp(-c * s_dist**(2 * gamma) / (a * t_dist**(2 * alpha) + 1)**(beta * gamma))
    sigma_mat = tmp1 * tmp2
    return pd.DataFrame(sigma_mat)

def matern_cov_yx(sigmasq: float = 1, range_: float = 1, v: float = 0.5, y= None, x_df=None) -> pd.DataFrame:
    """
    Compute conditional covariance matrix using matern model

    Parameters
    ----------
    sigmasq
        scale parameter
    range_
        range parameter, decides the rate of decay with distance in covariance function. 
    v
        smooth parameter, v=0.5 for exponential kernel.
    y
        conditioned y, current y with size 1
    
    x_df
        conditioning set
  
    Returns
    -------
    pd.DataFrame
        Returns n x 1 covariance matrix. 
    """
    # Validate inputs
    if y is None or x_df is None:
        raise ValueError("Both y and x_df must be provided.")

    # Extract values
    x1 = x_df['Longitude'].values
    y1 = x_df['Latitude'].values

    x2 = y['Longitude'].values
    y2 = y['Latitude'].values

    coords1 = np.stack((x1, y1), axis=-1)
    coords2 = np.stack((x2, y2), axis=-1)

    # Calculate spatial distances using cdist
    s_dist = cdist(coords1, coords2, 'euclidean')
    
    # Initialize the covariance matrix with zeros
    out = np.zeros_like(s_dist)
    
    # Compute the covariance for non-zero distances
    non_zero_indices = s_dist != 0
    if np.any(non_zero_indices):
        out[non_zero_indices] = (sigmasq * (2**(1-v)) / gamma(v) *
                                 (s_dist[non_zero_indices] / range_)**v *
                                 kv(v, s_dist[non_zero_indices] / range_))
        
    
    out[~non_zero_indices] = sigmasq

    return pd.DataFrame(out)

def neg_log_likelihood_nugget(params, input_df, y):
    v= 0.5
    range_ = 2
    
    # Compute the covariance matrix from the matern function
    cov_matrix = matern_cov_spat(sigmasq=1, range_=range_, v=v, input_df=input_df)
    
    # Add a small jitter term to the diagonal for numerical stability
    cov_matrix += np.eye(cov_matrix.shape[0]) * params
    
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

def neg_ll_nugget(params, input_df, mm_cond_number):
    """
    Compute negative log likelihood function of matern model using Vecchia approximation

    Parameters
    ----------
    sigmasq
        scale parameter
    range_
        range parameter, decides the rate of decay with distance in covariance function. 
    v
        smooth parameter, v=0.5 for exponential kernel.
    y
        conditioned y, current y with size 1
    
    x_df
        conditioning set
  
    Returns
    -------
    pd.DataFrame
        Returns n x 1 covariance matrix. 
    """
    v= 0.5
    range_ = 1
    sigmasq = 1

    # Extract values
    x1 = input_df['Longitude'].values
    y1 = input_df['Latitude'].values 
 
    coords1 = np.stack((x1, y1), axis=-1)

    # Calculate spatial distances using cdist
    s_dist = cdist(coords1, coords1, 'euclidean')

    # reorder data using maxmin
    ord, _ = maxmin_naive(s_dist, 0)
    reordered_df = input_df.iloc[ord,:]

    # Centering data
    reordered_df['ColumnAmountO3'] = reordered_df['ColumnAmountO3']-np.mean(reordered_df['ColumnAmountO3'])

    # initialize negative log-likelihood value
    neg_log_lik = 0

    # Construct nearest neighboring set
    nns_map = find_nns_naive(locs= coords1, dist_fun= 'euclidean', max_nn= mm_cond_number)
    

    smallset = input_df.iloc[:31,:]
    neg_log_lik += neg_log_likelihood_nugget(params, smallset, smallset['ColumnAmountO3'])

    for i in range(31,len(reordered_df)):
        # current_data and conditioning data
        current_data = reordered_df.iloc[i:i+1,:]
        current_y = current_data['ColumnAmountO3'].values[0]
        mm_past = nns_map[i,:]
        mm_past = mm_past[mm_past!=-1]
        mm_past = np.arange(i)
        conditioning_data = reordered_df.loc[mm_past,: ]

        df = pd.concat( (current_data, conditioning_data), axis=0)
        y_and_neighbors = df['ColumnAmountO3'].values

        cov_matrix = matern_cov_yx(sigmasq = sigmasq, range_ = range_, v = v, y= df, x_df=df)
        # Regularization: 
        # regularization_factor = 1e-8
        regularization_factor = params
        cov_matrix += regularization_factor * np.eye(cov_matrix.shape[0])

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


#### run


## param1 (resolution)
## param2 nugget size
## param3  (number of conditning number in vecchia approximation)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Full vs Vecchia Comparison")
    
    # Define the parameters you want to change at runtime
    parser.add_argument('--resolution', type=float, default=0.4, help="Resolution parameter")
    parser.add_argument('--nugget', type=float, default=30, help="Nugget parameter")
    parser.add_argument('--mm_cond_number', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Use args.param1, args.param2 in your script
    resolution = args.resolution
    nugget = args.nugget
    mm_cond_number = args.mm_cond_number

    # Example usage of your functions
    df = pd.read_csv('/home/jl2815/tco/data/data_N2530_E95110/data_24_07_0130_N2530_E95110.csv')

    instance = orbitmap.MakeOrbitdata(df, resolution, resolution,10,20,120,135)
    orbit_map24_7 = instance.makeorbitmap()
    # instance24_7 = orbitmap.MakeOrbitdata(df,10,20,120,135)
    sparse_map_24_7 = instance.make_sparsemap(orbit_map24_7, resolution)

    data = sparse_map_24_7['y24m07day01_1']

    mm_cond_number = mm_cond_number
    out = neg_ll_nugget(nugget, data, mm_cond_number)

    print(f'Full likelihood using nugget size {nugget} is {neg_log_likelihood_nugget(nugget, data, data['ColumnAmountO3'])}')
    print(f'Vecchia approximation likelihood using condition size {mm_cond_number}, nugget size {nugget} is {out}')


if __name__ == '__main__':
    main()