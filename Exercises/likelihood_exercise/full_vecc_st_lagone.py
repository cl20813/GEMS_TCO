# work environment: jl2815
# Standard libraries
import sys
import logging
import argparse # Argument parsing
import math
from collections import defaultdict
import concurrent
from concurrent.futures import ThreadPoolExecutor  # Importing specific executor for clarity
import time

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Nearest neighbor search
import sklearn
from sklearn.neighbors import BallTree

# Special functions and optimizations
from scipy.special import gamma, kv  # Bessel function and gamma function
from scipy.stats import multivariate_normal  # Simulation
from scipy.optimize import minimize
from scipy.spatial.distance import cdist  # For space and time distance
from scipy.spatial import distance  # Find closest spatial point
from scipy.optimize import differential_evolution

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Type hints
from typing import Callable, Union, Tuple

# Add your custom path
sys.path.append("/cache/home/jl2815/tco")

# Custom imports
from GEMS_TCO import orbitmap 
from GEMS_TCO import kernels 
from GEMS_TCO import smoothspace

import pickle

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_output/logs/fit_st1.log'

logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Full vs Vecchia Comparison")
    # Define the parameters you want to change at runtime
    parser.add_argument('--space', type=int,nargs='+', default=[20,20], help="spatial resolution")
    parser.add_argument('--mm_cond_number', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    parser.add_argument('--key', type=int, default=1, help="Index for the datasets.")
    parser.add_argument('--params', type=float,nargs='+', default=[0.5,0.5,0.5,0.5,0.5, 0.5], help="Initial parameters")
   
    # Parse the arguments
    args = parser.parse_args()
    
    # Use args.param1, args.param2 in your script
    lat_lon_resolution = args.space 
    mm_cond_number = args.mm_cond_number
    params= args.params
    key_for_dict= args.key

    ############ 


    # Load the one dictionary to set spaital coordinates
    filepath = "/home/jl2815/tco/data/pickle_data/pickle_2023/coarse_cen_map23_01.pkl"

    with open(filepath, 'rb') as pickle_file:
        coarse_dict_24_1 = pickle.load(pickle_file)

    sample_df = coarse_dict_24_1['y23m01day01_hm02:12']

    sample_key = coarse_dict_24_1.get('y23m01day01_hm02:12')
    if sample_key is None:
        print("Key 'y23m01day01_hm02:12' not found in the dictionary.")

    # { (20,20):(5,1), (5,5):(20,40) }
    rho_lat = lat_lon_resolution[0]          
    rho_lon = lat_lon_resolution[1]
    lat_n = sample_df['Latitude'].unique()[::rho_lat]
    lon_n = sample_df['Longitude'].unique()[::rho_lon]

    lat_number = len(lat_n)
    lon_number = len(lon_n)

    # Set spatial coordinates for each dataset
    coarse_dicts = {}

    years = ['2024']
    for year in years:
        for month in range(7, 8):  # Iterate over all months
            filepath = f"/home/jl2815/tco/data/pickle_data/pickle_{year}/coarse_cen_map{year[2:]}_{month:02d}.pkl"
            with open(filepath, 'rb') as pickle_file:
                loaded_map = pickle.load(pickle_file)
                for key in loaded_map:
                    tmp_df = loaded_map[key]
                    coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                    coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)

    
    key_idx = sorted(coarse_dicts)
    if not key_idx:
        raise ValueError("coarse_dicts is empty")

    # extract first hour data because all data shares the same spatial grid
    data_for_coord = coarse_dicts[key_idx[0]]
    x1 = data_for_coord['Longitude'].values
    y1 = data_for_coord['Latitude'].values 
    coords1 = np.stack((x1, y1), axis=-1)

    instance = orbitmap.MakeOrbitdata()
    s_dist = cdist(coords1, coords1, 'euclidean')
    ord_mm, _ = instance.maxmin_naive(s_dist, 0)

    data_for_coord = data_for_coord.iloc[ord_mm].reset_index(drop=True)
    coords1_reordered = np.stack((data_for_coord['Longitude'].values, data_for_coord['Latitude'].values), axis=-1)
    nns_map = instance.find_nns_naive(locs=coords1_reordered, dist_fun='euclidean', max_nn=mm_cond_number)

    

    analysis_data_map = {}
    for i in range(key_for_dict):
        tmp = coarse_dicts[key_idx[i]]
        tmp = tmp.iloc[ord_mm].reset_index(drop=True)  
        analysis_data_map[key_idx[i]] = tmp

    aggregated_data = pd.DataFrame()
    for i in range((key_for_dict)):
        tmp = coarse_dicts[key_idx[i]]
        tmp = tmp.iloc[ord_mm].reset_index(drop=True)  
        aggregated_data = pd.concat((aggregated_data, tmp), axis=0)
    
    
    print(f'Aggregated data shape: {aggregated_data.shape}')
    # print(aggregated_data.to_string())
    #####################################################################

    instance = kernels.matern_spatio_temporal(smooth=0.5, input_map=analysis_data_map, nns_map=nns_map, mm_cond_number=mm_cond_number)
    # data = data.iloc[ord, :]

    out = instance.vecchia_likelihood(params)
    out2 = instance.vecchia_likelihood2(params)

    start_time = time.time()
    full_likelihood = instance.full_likelihood(params, aggregated_data, aggregated_data["ColumnAmountO3"])
    print(f'Spatial grid lat ({lat_number}) * lon ({lon_number}), {key_for_dict} timestamps:\n Full likelihood using params={params} is {full_likelihood}')
    end_time = time.time()  # Record the end time
    iteration_time = end_time - start_time  # Calculate the time spent
    print(f"Full likelihood took {iteration_time:.4f} seconds")

    start_time2 = time.time()

    # Introduce a small delay for testing purposes
    time.sleep(0.01)  # Sleep for 10 milliseconds

    print(f'Spatial grid lat ({lat_number}) * lon ({lon_number}), {key_for_dict} timestamps:\n Vecchia approximation likelihood using condition size {mm_cond_number}, params={params} is {out}')
    end_time2 = time.time()  # Record the end time
    iteration_time2 = end_time2 - start_time2  # Calculate the time spent
    print(f"Vecchia approximation took {iteration_time2:.4f} seconds")


    start_time2 = time.time()

    # Introduce a small delay for testing purposes
    time.sleep(0.01)  # Sleep for 10 milliseconds

    print(f'Spatial grid lat ({lat_number}) * lon ({lon_number}), {key_for_dict} timestamps:\n Vecchia approximation likelihood using condition size {mm_cond_number}, params={params} is {out}')
    end_time2 = time.time()  # Record the end time
    iteration_time2 = end_time2 - start_time2  # Calculate the time spent
    print(f"Vecchia approximation2 took {iteration_time2:.4f} seconds")


if __name__ == '__main__':
    main()
   
 