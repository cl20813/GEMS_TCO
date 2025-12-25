# Standard libraries
import sys
import os
import logging
import argparse 
from pathlib import Path
from typing import Optional, List, Tuple
import pickle
import json
from json import JSONEncoder

# Third-party libraries
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import typer

# ---------------------------------------------------------
# [Amarel Path Setting]
# ---------------------------------------------------------
sys.path.append("/cache/home/jl2815/tco")

# Custom imports
from GEMS_TCO import data_preprocess 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import alg_optimization, alg_opt_Encoder
from GEMS_TCO import evaluate
from GEMS_TCO import configuration as config

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Range of days to process (start, end)"),
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors")
) -> None:
    
    ## 1. Initialize setting
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    
    # 처리할 날짜 리스트 (예: 0~30)
    target_days_list = list(range(days_s_e[0], days_s_e[1]))
    
    years = ['2024']
    month_range = [7]

    # [Amarel Config] 데이터 로드 경로
    data_load_instance = load_data2(config.amarel_data_load_path)

    # [Amarel Setting] 분석 범위
    lat_range_input = [0, 5]      
    lon_range_input = [120.5, 130.5] 

    print("--- [Amarel] Loading MaxMin Ordered Data ---")
    
    # load_maxmin_ordered_data_bymonthyear는 내부적으로 'rect' 파일을 찾음
    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=lat_lon_resolution, 
        mm_cond_number=mm_cond_number,
        years_=years, 
        months_=month_range,
        lat_range=lat_range_input,   
        lon_range=lon_range_input
    )
    
    ############################## 
    # [Amarel Config] 저장 경로 설정
    instance_sem = evaluate.CrossVariogram(config.amarel_save_computed_semi_path, 7)

    daily_hourly_maps_tmp = []      

    print(f"--- Processing Days: {target_days_list} ---")

    # ---------------------------------------------------------
    # [Safe Loading Logic]
    # 인덱스 하드코딩 대신, 실제 존재하는 날짜 Key를 찾아 로딩
    # ---------------------------------------------------------
    for day_num in target_days_list:
        # 데이터 파일 내 날짜 포맷 생성 (예: day00 -> day01)
        # 데이터가 1일부터 시작한다면 day_num + 1
        current_day_str = f"day{day_num + 1:02d}"
        
        # 전체 맵에서 해당 날짜(dayXX)가 포함된 Key만 추출
        day_keys = [k for k in sorted(df_map.keys()) if current_day_str in k]
        
        if not day_keys:
            print(f"Warning: No data found for {current_day_str}. Skipping.")
            continue
            
        # 해당 날짜 데이터만 담은 subset 생성
        day_df_map_subset = {k: df_map[k] for k in day_keys}
        
        # load_working_data에 subset 전달 (인덱스는 0부터 개수만큼)
        subset_indices = [0, len(day_keys)]

        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
            coarse_dicts=day_df_map_subset, 
            idx_for_datamap=subset_indices, 
            ord_mm=None,  
            dtype=torch.float64, 
            keep_ori=True   
        )

        daily_hourly_maps_tmp.append(day_hourly_map)

    # 리스트를 하나의 딕셔너리로 병합
    daily_hourly_maps = {}
    for cur in daily_hourly_maps_tmp:
        daily_hourly_maps.update(cur)

    if not daily_hourly_maps:
        print("Error: No data loaded. Exiting.")
        return

    # 디버깅용: 로드된 키 확인
    loaded_keys = sorted(daily_hourly_maps.keys())
    print(f"Successfully loaded {len(loaded_keys)} timestamps.")

    '''
    In this study, the center matched data is used. 
    '''
    tmp_lon = np.concatenate([
        -np.arange(0.18, 2, 0.063 * 3)[::-1],
        [-0.126, -0.063, 0, 0.063, 0.126],
        np.arange(0.18, 2, 0.063 * 3)
    ])

    tmp_lat = np.concatenate([
        -np.arange(0.176, 2.3, 0.044 * 5)[::-1],
        [-0.132, -0.044, 0, 0.044, 0.132],
        np.arange(0.176, 2.3, 0.044 * 5)
    ])

    tmp_lon_uni = np.concatenate([
        [0, 0.063, 0.126],
        np.arange(0.18, 2, 0.063 * 3)
    ])

    tmp_lat_uni = np.concatenate([
        [0, 0.044, 0.132],
        np.arange(0.176, 2.3, 0.044 * 5)
    ])

    lat_deltas = [ ( round(a,3),0 ) for a in tmp_lat]
    lon_deltas = [ (0, round(a,3)) for a in tmp_lon]
    
    tolerance = 0.015
    
    # Create deltas for univariate only
    lat_deltas_uni = [(round(lat, 3), 0) for lat in tmp_lat_uni]
    lon_deltas_uni = [(0, round(lon, 3)) for lon in tmp_lon_uni]

    print("--- Computing Semivariograms ---")
    
    # Compute
    lat_lag_sem = instance_sem.compute_directional_semivariogram(lat_deltas_uni, daily_hourly_maps, target_days_list, tolerance)
    lon_lag_sem = instance_sem.compute_directional_semivariogram(lon_deltas_uni, daily_hourly_maps, target_days_list, tolerance)

    cross_lat_lag_sem = instance_sem.compute_cross_lon_lat(lat_deltas, daily_hourly_maps, target_days_list, tolerance)
    cross_lon_lag_sem = instance_sem.compute_cross_lon_lat(lon_deltas, daily_hourly_maps, target_days_list, tolerance)
    
    # ---------------------------------------------------------
    # [Fix for Amarel] Dictionary slicing error fix
    # ---------------------------------------------------------
    sample_key = target_days_list[0] + 1
    
    print(f"Lat results sample (Day {sample_key}):", lat_lag_sem[sample_key][:1] if sample_key in lat_lag_sem else "Not computed")
    print(f"Lon results sample (Day {sample_key}):", lon_lag_sem[sample_key][:1] if sample_key in lon_lag_sem else "Not computed")
    # ---------------------------------------------------------

    lat_filename = f"empirical_lat_sem_july24.pkl"
    lon_filename = f"empirical_lon_sem_july24.pkl"

    cross_lat_filename = f"empirical_cross_lat_sem_july24.pkl"
    cross_lon_filename = f"empirical_cross_lon_sem_july24.pkl"

    output_path = instance_sem.save_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print(f"--- Saving Results to {output_path} ---")

    lat_filepath = os.path.join(output_path, lat_filename)
    with open(lat_filepath, 'wb') as pickle_file:
        pickle.dump(lat_lag_sem, pickle_file)  

    lon_filepath = os.path.join(output_path, lon_filename)
    with open(lon_filepath, 'wb') as pickle_file:
        pickle.dump(lon_lag_sem, pickle_file)   

    cross_lat_filepath = os.path.join(output_path, cross_lat_filename)
    with open(cross_lat_filepath, 'wb') as pickle_file:
        pickle.dump(cross_lat_lag_sem, pickle_file)  

    cross_lon_filepath = os.path.join(output_path, cross_lon_filename)
    with open(cross_lon_filepath, 'wb') as pickle_file:
        pickle.dump(cross_lon_lag_sem, pickle_file)
    
    print("Done.")

if __name__ == '__main__':
    app()