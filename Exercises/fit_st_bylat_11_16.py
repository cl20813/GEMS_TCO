# work environment: jl2815
# Standard libraries
import sys
import logging
import argparse # Argument parsing
import math
from collections import defaultdict
import concurrent
from concurrent.futures import ThreadPoolExecutor  # Importing specific executor for clarity

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


# Configure logging to a specific file path
log_file_path = '/home/jl2815/GEMS/logs/fit_st_by_latitude_11_14.log'

logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Fit spatio-temporal model")
    
    #sigmasq (0.05,600), range_ (0.05,600), advec (-200,200), beta (0,600), nugget (0,600)
    parser.add_argument('--v', type=float, default=0.5, help="smooth")
    parser.add_argument('--rho', type=int, default=8, help="Resolution for coarse set")
    parser.add_argument('--bounds', type=float, nargs='+', default=[0.05, 600, 0.05, 600, -200, 200, 0.5, 600, 0.5, 600], help="Bounds for parameters" )
    parser.add_argument('--params', type=float,nargs='+', default=[0.5,0.5,0.5,0.5,0.5], help="sigmasq, range_, advec, beta, nugget ")
    # nargs='+' for --initial_params: Ensures multiple float values are accepted as a single list.
    # parser.add_argument('--nugget', type=float, default=5, help="Nugget parameter")
    parser.add_argument('--mm_cond_number', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    parser.add_argument('--key', type=int, default=1, help="Index for the days")
    parser.add_argument('--lat_idx', type=int, default=1, help="latitude")
   
    # Parse the arguments
    args = parser.parse_args()

    smooth = args.v
    rho = args.rho
    # Convert bounds to a list of tuples
    bounds = [(args.bounds[i], args.bounds[i+1]) for i in range(0, len(args.bounds), 2)]
    initial_params= args.params 
    mm_cond_number = args.mm_cond_number
    key= args.key
    lat_idx= args.lat_idx

    # data
    # df = pd.read_csv('/home/jl2815/tco/data/data_N2530_E95110/data_24_07_0130_N2530_E95110.csv')
    df = pd.read_csv('/home/jl2815/tco/data/data_N510_E110120/data_24_07_0130_N510_E110120.csv')
    # df = pd.read_csv('/home/jl2815/tco/data/data_N2530_E95110/data_24_07_0130_N2530_E95110.csv')
    
    # make coarse set
    instance = orbitmap.MakeOrbitdata(df, 5,10,110,120)
    orbit_map24_7 = instance.makeorbitmap()
    resolution_uni = 0.05 
    sparse_map_24_7 = instance.make_sparsemap(orbit_map24_7, resolution_uni)
    # rho:observations  -->  1: 20000  3: 2223 4:1250, 5:1190, 6:556,, 8:313
    coarse_map24_7 =  instance.make_coarsemap(sparse_map_24_7,rho)  

    # reorder data and initiate parallel computing functions
    databyday_instance = orbitmap.databyday_24July()
    grp, baseset_from_maxmin, nns_map  = databyday_instance.process(coarse_map24_7,lat_idx,mm_cond_number)
    matern_st_11 = kernels.matern_st_11(smooth)

    # data = grp[key]
    ####################################  data set up
    # reset keys for new map
    keys = sorted(grp)[0:key]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                matern_st_11.mle_parallel,
                key, lat_idx, bounds,initial_params, grp[key], mm_cond_number, baseset_from_maxmin, nns_map
            )
            for key in keys
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == '__main__':
    main()