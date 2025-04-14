#!/opt/anaconda3/envs/faiss_env/bin/python

import sys
import os
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)


# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle 

import GEMS_TCO
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import load_data_local_computer

import torch
from collections import defaultdict

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy  # clone tensor

def main():
    input_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates"

    # input_filename = "vecc_extra_estimates_50_july24.pkl"
    # input_filename = "vecc_inter_estimates_1250_july24.pkl"

    input_filename = "vecc_inter_estimates_5000_july24.pkl"
    # input_filename = "estimation_200_july24.pkl"
    input_filename = "full_estimation_1250_july24.pkl"
    input_filepath = os.path.join(input_path, input_filename)
    # Load pickle
    with open(input_filepath, 'rb') as pickle_file:
        amarel_map1250 = pickle.load(pickle_file)

    # Assuming df_1250 is your DataFrame
    df_1250 = pd.DataFrame()
    for key in amarel_map1250:
        tmp = pd.DataFrame(amarel_map1250[key][0].reshape(1, -1), columns=['sigmasq', 'range_lat', 'range_lon', 'advec_lat', 'advec_lon', 'beta', 'nugget'])
        tmp['loss'] = amarel_map1250[key][1]
        df_1250 = pd.concat((df_1250, tmp), axis=0)

    # Generate date range
    date_range = pd.date_range(start='07-01-24', end='07-31-24')

    # Ensure the number of dates matches the number of rows in df_1250
    if len(date_range) == len(df_1250):
        df_1250.index = date_range
    else:
        print("The number of dates does not match the number of rows in the DataFrame.")

  
    df = df_1250

    lat_lon_resolution = [2, 2]
    years = ['2024']
    month_range = [7, 8]
    # nheads = 100  
    
    # 50 for 10 work best for competitor 2 lags reordered, cahced
    #  50 for resolution 10: result1 [11,10,9] result2 = [9,11,10]
    #  200 for resolution 4, result1 [21,2,7] result2=[7,7,16]
    # 300 for resolution 4, result1 [23, 1, 6]  result2 = [6,10,14]
    result_2 = defaultdict(list)
    result_1 =  defaultdict(list)
    for day in range(1,3):
        print(f'\n Day {day} data size per day: { (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) } \n')

        # parameters
        mm_cond_number = 10 + day
        idx_for_datamap = [8 * (day - 1), 8 * day]
        # params = [27.25, 2.18, 2.294, 4.099e-4, -0.07915, 0.0999, 3.65]   #200
        params = list(df.iloc[day - 1][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        # data
        instance = load_data_local_computer()
        map, ord_mm, nns_map = instance.load_mm20k_data_bymonthyear(lat_lon_resolution=lat_lon_resolution, mm_cond_number=mm_cond_number, years_=years, months_=month_range)
        analysis_data_map, aggregated_data = instance.load_working_data_byday(map, ord_mm, nns_map, idx_for_datamap=idx_for_datamap)

        # different approximations
        key_order = [0, 1, 2, 4, 3, 5, 7, 6]
        keys = list(analysis_data_map.keys())
        reordered_dict = {keys[key]: analysis_data_map[keys[key]] for key in key_order}

        nheads=100
        instance_ori = kernels.vecchia_experiment(0.5, analysis_data_map, aggregated_data, nns_map, mm_cond_number, nheads)
        # out = instance_ori.full_likelihood(params, aggregated_data[:, :4], aggregated_data[:, 2], instance_ori.matern_cov_anisotropy_v05)
        # print(f'full: {out}')  
        out = 100

        nheads = 100
        instance = kernels.vecchia_experiment(0.5, reordered_dict, aggregated_data, nns_map, mm_cond_number, nheads)
        out2 = instance.vecchia_b2(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc two lags: {out2}')  
        out3 = instance.vecchia_competitor(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc competitor: {out3}')  

        nheads = 200
        instance = kernels.vecchia_experiment(0.5, reordered_dict, aggregated_data, nns_map, mm_cond_number, nheads)
        out22 = instance.vecchia_b2(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc two lags: {out22}')  
        out33 = instance.vecchia_competitor(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc competitor: {out33}')  

        nheads = 300
        instance = kernels.vecchia_experiment(0.5, reordered_dict, aggregated_data, nns_map, mm_cond_number, nheads)
        out222 = instance.vecchia_b2(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc two lags: {out222}')  
        out333 = instance.vecchia_competitor(params, instance.matern_cov_anisotropy_v05)
        print(f'vecc competitor: {out333}')  


        approx_map = {0: 'two lag nh100', 1: 'competitor nh100', 2: 'two lag nh200', 3: 'competitor nh200', 4: 'two lag nh300', 5: 'competitor nh300'   }

        tmp = [out2,out3, out22,out33, out222,out333]
        tmp = torch.stack( tmp)

        tmp_result = [ torch.abs(out - out2), torch.abs(out - out3),  torch.abs(out - out22), torch.abs(out - out33), torch.abs(out - out222), torch.abs(out - out333)   ]
        stacked_tensor = torch.stack(tmp_result)
        top2_indices = torch.topk(stacked_tensor, 5, largest=False).indices

        best_index = top2_indices[0].item()
        second_best_index = top2_indices[1].item()
        third_best_index = top2_indices[2].item()
        fourth_best_index = top2_indices[3].item()
        fifth_best_index = top2_indices[4].item()

        # Update the result for the best approximation
        result_1[(day,best_index)] = [(best_index, tmp[best_index].item())]
        result_1[(day,second_best_index)] = [(best_index, tmp[second_best_index].item())]
        result_1[(day,third_best_index)] = [(best_index, tmp[third_best_index].item())]
        result_1[(day,fourth_best_index)] = [(best_index, tmp[fourth_best_index].item())]
        result_1[(day,fifth_best_index)] = [(best_index, tmp[fifth_best_index].item())]

    
        # Print the results
        print(f'\n\n Day {day} full likelihood: {out}\n parameters: {params.tolist()} \n')
        print(f'Best approximation: {approx_map[best_index]} with abs_diff: {stacked_tensor[best_index]}')
        print(f'Second best approximation: {approx_map[second_best_index]} with abs_diff: {stacked_tensor[second_best_index].item()}')
        print(f'third best approximation: {approx_map[third_best_index]} with abs_diff: {stacked_tensor[third_best_index].item()}')
        print(f'results summary {result_1}')
if __name__ == "__main__":
    main()