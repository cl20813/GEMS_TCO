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
from GEMS_TCO import evaluate
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
    

    # Set the days of the month
    

    #############
    ############# longitude semivariogram 

    idx_for_datamap= [0,248]
    print(f'lon_sem, data size per hour: {int((200/rho_lat)*(100/rho_lon))}')

    analysis_data_map, aggregated_data = instance.load_working_data_byday( ym_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

    deltas = [(0,0), (0, 0.2), (0, 0.4), (0, 0.6), (0, 0.8), (0, 1) ]
    # Include negative deltas
    deltas += [(d[0], -d[1]) for d in deltas if d[1] != 0] + [(-d[0], d[1]) for d in deltas if d[0] != 0]

    map = analysis_data_map
    tolerance = 0.02

    days= list(np.arange(1,32))  # Example days you want to plot
    instance_sem = evaluate.CrossVariogram()
    lon_lag_sem = instance_sem.cross_lon_lat(deltas, map, days, tolerance)


    output_filename = f"empirical_lon_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/eda"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(lon_lag_sem, pickle_file)    

    #############
    ############# latitude semivariogram 


    print(f'lat_sem, data size per hour: {int((200/rho_lat)*(100/rho_lon))}')
    idx_for_datamap= [0,248]
    analysis_data_map, aggregated_data = instance.load_working_data_byday( ym_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

    deltas = [ (0,0), (0.2, 0), (0.4, 0), (0.6, 0), (0.8, 0), (1, 0)]
    # Include negative deltas
    deltas += [(d[0], -d[1]) for d in deltas if d[1] != 0] + [(-d[0], d[1]) for d in deltas if d[0] != 0]

    map = analysis_data_map
    tolerance = 0.02

    instance_sem = evaluate.CrossVariogram()
    days= list(np.arange(1,32))

    lat_lag_sem = instance_sem.cross_lon_lat(deltas, map, days, tolerance)

    output_filename = f"empirical_lat_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/eda"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(lat_lag_sem, pickle_file)    


    #############
    ############# directional semivariogram y= x

    print(f'directional_sem, data size per hour: {int((200/rho_lat)*(100/rho_lon))}')
    idx_for_datamap= [0,248]
    analysis_data_map, aggregated_data = instance.load_working_data_byday( ym_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

    d45 = np.arctan2(1,1)
    d135 = np.arctan2(-1,-1)  
    neg_deltas = [ 1, 0.8,0.6, 0.4 ,0.2, 0]
    pos_deltas = [ 0.2, 0.4, 0.6, 0.8, 1.0]
    deltas = neg_deltas + pos_deltas

    direction1 = d135
    direction2 = d45

    days= list(np.arange(1,32))  # Example days you want to plot
    map = analysis_data_map
    tolerance = 0.03  # no pairs for 0.02
    instance_sem = evaluate.CrossVariogram()
    d135_45_sem = instance_sem.cross_directional_sem(deltas, map,  days, tolerance, direction1=dn45, direction2=dn135)

    output_filename = f"empirical_{(direction1*(180/np.pi)) ,(direction2*(180/np.pi))}_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/eda"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(d135_45_sem, pickle_file)  


    #############
    ############# directional semivariogram y= -x
  
    
    print(f'directional_sem, data size per hour: {int((200/rho_lat)*(100/rho_lon))}')
    idx_for_datamap= [0,248]
    analysis_data_map, aggregated_data = instance.load_working_data_byday( ym_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

    dn45 = np.arctan2(-1,1)
    dn135 = np.arctan2(1,-1)
    direction1 = dn45
    direction2 = dn135

    neg_deltas = [ 1, 0.8,0.6, 0.4 ,0.2, 0]
    pos_deltas = [ 0.2, 0.4, 0.6, 0.8, 1.0]
    deltas = neg_deltas + pos_deltas

    days= list(np.arange(1,32))  # Example days you want to plot
    map = analysis_data_map
    tolerance = 0.03  # no pairs for 0.02

    instance_sem = evaluate.CrossVariogram()
    dn45_n135_sem = instance_sem.cross_directional_sem(deltas, map,  days, tolerance, direction1=dn45, direction2=dn135)

    output_filename = f"empirical_{(direction1*(180/np.pi)) ,(direction2*(180/np.pi))}_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/eda"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(dn45_n135_sem, pickle_file)  



if __name__ == '__main__':
    main()