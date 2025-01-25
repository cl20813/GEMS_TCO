# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict
import math
from scipy.special import gamma  # better than math.gamma(v)


from typing import Callable   # nearest neighbor function input type
import sklearn.neighbors  # nearest neighbor

from sklearn.neighbors import BallTree # for space_center function

import scipy
from scipy.stats import multivariate_normal #simulation
from scipy.optimize import minimize
from scipy.special import kv                # bessel function

import concurrent.futures
import sys

from scipy.spatial.distance import cdist # for space and time distance
from scipy.spatial import distance # find closest spatial point

from typing import Callable, Union   # nearest neighbor function input type
from typing import Tuple

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import orbitmap 
from GEMS_TCO import kernels 
from GEMS_TCO import smoothspace

import argparse
import pickle



def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Full vs Vecchia Comparison")
    # Define the parameters you want to change at runtime
    parser.add_argument('--space', type=int,nargs='+', default=[20,20], help="spatial resolution")
    parser.add_argument('--mm_cond_number', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    parser.add_argument('--key', type=int, default=1, help="Index for the datasets.")
    parser.add_argument('--params', type=float,nargs='+', default=[0.5,0.5,0.5,0.5, 0.5], help="Initial parameters")
    # Parse the arguments
    args = parser.parse_args()
    
    # Use args.param1, args.param2 in your script
    lat_lon_resolution = args.space 
    mm_cond_number = args.mm_cond_number
    params= args.params
    key_dict= args.key

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
    data = coarse_dicts[key_idx[key_dict]]
    #####################################################################

    # Extract values
    x1 = data['Longitude'].values
    y1 = data['Latitude'].values 
    coords1 = np.stack((x1, y1), axis=-1)
    # Calculate spatial distances using cdist

    instance = orbitmap.MakeOrbitdata()
    s_dist = cdist(coords1, coords1, 'euclidean')
    # reorder data using maxmin
    ord_mm, _ = instance.maxmin_naive(s_dist, 0)
    # Construct nearest neighboring set
    
    # nns_map = find_nns_naive(locs= coords1, dist_fun= 'euclidean', max_nn= mm_cond_number)

    # Reorder the DataFrame
    data = data.iloc[ord_mm].reset_index(drop=True)  
    coords1_reordered = np.stack((data['Longitude'].values, data['Latitude'].values), axis=-1)
    nns_map = instance.find_nns_naive(locs=coords1_reordered, dist_fun='euclidean', max_nn=mm_cond_number)

    instance = kernels.matern_spatial()
    # data = data.iloc[ord,:]
    out = instance.vecchia_likelihood(params, data, mm_cond_number, nns_map)

    print(f'Full likelihood using {params} is {instance.full_likelihood(params, data, data["ColumnAmountO3"])}')
    print(f'Vecchia approximation likelihood using condition size {mm_cond_number}, {params} is {out}')


if __name__ == '__main__':
    main()
