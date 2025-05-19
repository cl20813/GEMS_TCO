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
import time

# Custom imports
import GEMS_TCO
from GEMS_TCO import kernels
from GEMS_TCO import data_preprocess 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data_amarel

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_output/logs/fit_vecc.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Fit spatio-temporal model")
    parser.add_argument('--v', type=float, default=0.5, help="smooth")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--space', type=int,nargs='+', default=[20,20], help="spatial resolution")
    parser.add_argument('--days', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    parser.add_argument('--mm_cond_number', type=int, default=1, help="Number of nearest neighbors in Vecchia approx.")
    parser.add_argument('--params', type=float,nargs='+', default=[20, 8.25, 5.25, .2, .2, .05 , 5], help="Initial parameters")
    parser.add_argument('--epochs', type=int, default=100, help="Number of iterations in optimization")
    
    # Parse the arguments
    args = parser.parse_args()

    # Use args.param1, args.param2 in your script
    lat_lon_resolution = args.space 
    days = args.days
    mm_cond_number = args.mm_cond_number
    params= torch.tensor(args.params, requires_grad=True)
    v = args.v
    key_for_dict= [0,8]
    lr = args.lr
    epochs = args.epochs
    rho_lat = lat_lon_resolution[0]          
    rho_lon = lat_lon_resolution[1]
    ############################## 

    # Load the one dictionary to set spaital coordinates
    years = ['2024']
    month_range =[7,8]
    
    instance = load_data_amarel()
    map, ord_mm, nns_map= instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)
   
    result_inter = {}

    for day in range(days):
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        lenth_of_analysis = key_for_dict[1]-key_for_dict[0]
        print(f'day {day+1}, data size per hour: {aggregated_data.shape[0]/lenth_of_analysis}')
        print(lat_lon_resolution, mm_cond_number, key_for_dict, params, v,lr)

        params = [24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34]
        params = torch.tensor(params, requires_grad=True)

        model_instance = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map,
                aggregated_data=aggregated_data,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number
            )


        start_time = time.time()

        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer, scheduler = model_instance.optimizer_fun(params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=20, gamma=0.9)    
        out = model_instance.run_vecc_interpolate(params, optimizer,scheduler, epochs=epochs)
        result_inter[day+1] = out

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'day {day + 1} took {epoch_time:.2f}')


    output_filename = f"vecc_inter_estimates_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/estimates"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(result_inter, pickle_file)    

if __name__ == '__main__':
    main()