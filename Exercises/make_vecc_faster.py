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
from GEMS_TCO import load_data

import torch
from collections import defaultdict

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy                    # clone tensor

from pathlib import Path
import json
from json import JSONEncoder

import time

import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity





     
# kernprof -l script_to_profile.py
# C:\Users\joonw\anaconda3\envs\faiss_env\python.exe -m kernprof -l "C:\Users\joonw\tco\GEMS_TCO-2\Exercises\make_vecc_faster.py"  window
# /opt/anaconda3/envs/faiss_env/bin/python /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/make_vecc_faster.py  mac  


# df = pd.read_csv("C:/Users/joonw/tco/GEMS_TCO-2/Exercises/st_model/estimates/full_estimates_1250_july24.csv")   # window

df = pd.read_csv("/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/estimates/full_estimates_1250_july24.csv") 

lat_lon_resolution = [10,10]  # 4,4 to coarse factor 2 // 2,2 to 4  5 is worse

years = ['2024']
month_range =[7,8]
nheads = 2

for day in range(5,6):
    print(f'\n Day {day} data size per day: { (200/lat_lon_resolution[0])*(100/lat_lon_resolution[0])  } \n')

    # parameters
    mm_cond_number = 10
    idx_for_datamap= [ 8*(day-1),8*day]
    # params = [ 27.25, 2.18, 2.294, 4.099e-4, -0.07915, 0.0999, 3.65]   #200
    params = list(df.iloc[day-1][:-1])
    params = torch.tensor(params, dtype=torch.float64, requires_grad=True)

    # data
    # input_path = Path("C:\\Users\\joonw\\tco\\Extracted_data")  # window

    input_path = Path("/Users/joonwonlee/Documents/GEMS_DATA")  # mac
    instance = load_data(input_path)
    map, ord_mm, nns_map= instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)
    analysis_data_map, aggregated_data = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap= idx_for_datamap)



    # different approximations
    key_order = [0,1,2,4,3,5,7,6]
    reordered_dict, reordered_df = instance.reorder_data(analysis_data_map, key_order)
    instance_ori = kernels.vecchia_experiment(0.5, analysis_data_map, aggregated_data,nns_map,mm_cond_number, nheads)
    instance = kernels.vecchia_experiment(0.5, reordered_dict, reordered_df, nns_map,mm_cond_number, nheads)

    
    start_time = time.time()
    out1 = instance.full_likelihood(params, aggregated_data[:,:4],aggregated_data[:,2], instance_ori.matern_cov_anisotropy_v05)
    end_time = time.time()
    epoch_time1 = end_time - start_time
    print(f'Exact likelihood: {out1} time: {epoch_time1:.2f}') 
    
    
    cov_map_ori = instance_ori.cov_structure_saver(params, instance_ori.matern_cov_anisotropy_v05)
    cov_map_new = instance.cov_structure_saver(params, instance.matern_cov_anisotropy_v05)

    start_time = time.time()
    out2 = instance_ori.vecchia_ori_order(params, instance_ori.matern_cov_anisotropy_kv, cov_map_ori)
    end_time = time.time()
    epoch_time2 = end_time - start_time
    print(f'Vecchia likelihood: {out2},  time: {epoch_time2:.2f}') 

    start_time = time.time()
    out2 = instance_ori.vecchia_grouping(params, instance_ori.matern_cov_anisotropy_kv, cov_map_ori)
    end_time = time.time()
    epoch_time2 = end_time - start_time
    print(f'Vecchia grouping: {out2},  time: {epoch_time2:.2f}') 
    

    ''' 
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with record_function("vecchia_ori_order"):
            start_time = time.time()
            out2 = instance_ori.vecchia_ori_order(params, instance_ori.matern_cov_anisotropy_kv, cov_map_ori)
            end_time = time.time()
            epoch_time2 = end_time - start_time
            print(f'Vecchia likelihood: {out2},  time: {epoch_time2:.2f}')

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace_vecchia_ori_order.json")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with record_function("vecchia_ori_order"):
            start_time = time.time()
            out2 = instance_ori.vecchia_grouping(params, instance_ori.matern_cov_anisotropy_kv, cov_map_ori)
            end_time = time.time()
            epoch_time2 = end_time - start_time
            print(f'Vecchia likelihood: {out2},  time: {epoch_time2:.2f}')

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace_vecchia_ori_order.json")
    '''





