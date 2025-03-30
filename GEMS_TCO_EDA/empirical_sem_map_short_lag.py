# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import logging
import argparse # Argument parsing

# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import copy                    # clone tensor
from collections import defaultdict

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
import GEMS_TCO
from GEMS_TCO import kernels
from GEMS_TCO import orbitmap 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data_amarel

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_output/logs/emp_sem.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Fit spatio-temporal model")
    parser.add_argument('--space', type=int,nargs='+', default=[20,20], help="spatial resolution")
    parser.add_argument('--days', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")

    # Parse the arguments
    args = parser.parse_args()

    # Use args.param1, args.param2 in your script
    lat_lon_resolution = args.space 
    days = args.days
    key_for_dict= [0,8]
    rho_lat = lat_lon_resolution[0]          
    rho_lon = lat_lon_resolution[1]
    ############################## 

    # Load the one dictionary to set spaital coordinates
    years = ['2024']
    month_range =[7,8]
    
    mm_cond_number=20
    instance = load_data_amarel()
    ym_map, ord_mm, nns_map= instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)
    
    def empirical_semivariogram(data, deltas ,tolerance, temporal_lag):
        # Extract data columns: latitude, longitude, time, values
        lat = data[:, 0]
        lon = data[:, 1]
        time = data[:, 3]
        values = data[:, 2]  # Assuming the values are in the 3rd column

        # Demean the values
        values = values - torch.mean(values)

        # Calculate spatial and temporal distances
        spatial_distances = torch.sqrt((lat.unsqueeze(1) - lat.unsqueeze(0))**2 + (lon.unsqueeze(1) - lon.unsqueeze(0))**2)
        temporal_distances = torch.abs(time.unsqueeze(1) - time.unsqueeze(0))

            # Calculate spatial and temporal distances
        lat_distances = lat.unsqueeze(1) - lat.unsqueeze(0)
        lon_distances = lon.unsqueeze(1) - lon.unsqueeze(0)

        empirical_sem = torch.zeros(len(deltas))
        
        # Calculate the semivariogram for each delta value
        for i, delta in enumerate(deltas):
            # Mask for spatial distances within the delta range and specific temporal lag
            mask = (torch.abs(lat_distances - delta) <= tolerance) & \
                (torch.abs(lon_distances - delta) <= tolerance) & \
                (temporal_distances == temporal_lag)
            
            if torch.any(mask):
                # Calculate semivariogram using squared differences of values
                differences = torch.abs(values.unsqueeze(1) - values.unsqueeze(0))
                empirical_sem[i] = 0.5 * torch.mean(differences[mask]**2)
        return deltas, empirical_sem    
    
    #deltas = torch.cat((torch.linspace(0.1, 0.9, 9), torch.linspace(1.1, 2.5, 7)), dim=0)
    deltas = torch.linspace(0.1, 0.1,1)
    colors = plt.cm.viridis(np.linspace(0, 1, 8))  # Color map for the temporal lags
    
    emp_sem_map = {}
    print(f'data size per hour: {int((200/rho_lat)*(100/rho_lon))}')

    tolerance=0.005

    for day in range(days):
        print(f'day {day+1}, data size per hour: {int((200/rho_lat)*(100/rho_lon))}')
        
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = instance.load_working_data_byday( ym_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        for temporal_lag in range(8): 
            deltas, empirical_sem= empirical_semivariogram(aggregated_data, deltas,tolerance, temporal_lag)
            mask = empirical_sem!=0

            deltas_copy = deltas[mask].clone().detach()
            emp_st_sem = empirical_sem[mask].clone().detach()

            emp_sem_map[(1, temporal_lag,0)] = deltas_copy
            emp_sem_map[(1, temporal_lag,1)] =  emp_st_sem

    output_filename = f"empirical_short_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/eda"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(emp_sem_map, pickle_file)    

if __name__ == '__main__':
    main()