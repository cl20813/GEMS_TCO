# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import torch
import typer
import numpy as np
import pandas as pd
import math
from typing import List
from pathlib import Path  # [추가] 경로 처리를 위해 필요

# Custom Imports
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2
from GEMS_TCO import debiased_whittle 


def transform_raw_to_model_params(raw_params: list) -> list:
    """
    Transforms raw parameters into the model's log-space parameter format.
    Input: [sigma_sq, range_lat, range_lon, advec_lat, range_time, advec_lon, nugget]
    Output: [log(phi1), log(phi2), log(phi3), log(phi4), advec_lat, advec_lon, log(nugget)]
    """
    # 1. Unpack raw parameters
    sigma_sq   = raw_params[0]
    range_lat  = raw_params[1]
    range_lon  = raw_params[2]
    range_time = raw_params[3]
    advec_lat  = raw_params[4]
    advec_lon  = raw_params[5]
    nugget     = raw_params[6]

    # 2. Calculate Phis
    phi2 = 1.0 / range_lon
    phi1 = sigma_sq * phi2
    phi3 = (range_lon / range_lat)**2
    phi4 = (range_lon / range_time)**2

    # 3. Apply Log
    transformed_params = [
        math.log(phi1),
        math.log(phi2),
        math.log(phi3),
        math.log(phi4),
        advec_lat,
        advec_lon,
        math.log(max(nugget, 1e-8))
    ]
    return transformed_params

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Range of days to process"),
    lat_range: List[str] = typer.Option(['0','5'], help="Total latitude range (e.g. 0,5)"),
    lon_range: List[str] = typer.Option(['123','133'], help="Total longitude range (e.g. 123,133)"),
    mm_cond_number: int = typer.Option(10, help="Vecchia neighbors"),
    nheads: int = typer.Option(200, help="Head points"),
    keep_exact_loc: bool = typer.Option(True, help="keep exact loc")
) -> None:
      
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7]

    # 입력받은 전체 범위
    total_lat_range = [int(s) for s in lat_range[0].split(',')]
    total_lon_range = [int(s) for s in lon_range[0].split(',')]

    print(f"Total Analysis Range: Lat {total_lat_range}, Lon {total_lon_range}")

    # =========================================================================
    # 1. Sub-Region 생성 로직 (Sliding Window)
    # =========================================================================
    lat_window, lat_stride = 3, 2
    lon_window, lon_stride = 5, 5

    sub_regions = []
    
    for lat_s in range(total_lat_range[0], total_lat_range[1], lat_stride):
        lat_e = lat_s + lat_window
        if lat_e > total_lat_range[1]: break 

        for lon_s in range(total_lon_range[0], total_lon_range[1], lon_stride):
            lon_e = lon_s + lon_window
            if lon_e > total_lon_range[1]: break
            
            sub_regions.append({
                'lat': [lat_s, lat_e],
                'lon': [lon_s, lon_e]
            })

    print(f"Generated {len(sub_regions)} sub-regions for analysis:")
    for i, r in enumerate(sub_regions):
        print(f"  Region {i+1}: Lat {r['lat']}, Lon {r['lon']}")

    # =========================================================================
    # 2. Parameters Loading & Transformation
    # =========================================================================
    # [수정] Path 객체로 감싸주어야 '/' 연산자가 작동합니다.
    path = Path("/cache/home/jl2815/tco/exercise_output/estimates/day/real_fit_dw_and_vecc_july24")
    vecc_path = path / 'real_vecc_july24_h1000_mm16.csv'
    dw_path = path / 'real_dw_july24.csv'

    print(f"Loading parameters from:\n  {vecc_path}\n  {dw_path}")
    
    dw_real = pd.read_csv(dw_path)
    vecc_real = pd.read_csv(vecc_path)

    # 파라미터 컬럼 슬라이싱 (인덱스 4부터 7개)
    dw_real = dw_real.iloc[:28, 4:(4+7)]
    vecc_real = vecc_real.iloc[:28, 4:(4+7)]

    whole_params = []
    # 28일치 파라미터 변환 및 저장
    for i in range(28):
        dw_params = dw_real.iloc[i].tolist()
        vecc_params = vecc_real.iloc[i].tolist()

        dw_transformed = transform_raw_to_model_params(dw_params)
        vecc_transformed = transform_raw_to_model_params(vecc_params)
        
        whole_params.append([vecc_transformed, dw_transformed])

    opt_method = ['Vecchia_Params', 'Whittle_Params']

    # 데이터 로더 인스턴스
    data_load_instance = load_data2(config.amarel_data_load_path)

    # =========================================================================
    # 3. Main Loop (Days -> Sub-Regions)
    # =========================================================================
    for day_idx in days_list:
        # 파라미터가 없는 날짜는 스킵
        if day_idx >= len(whole_params): 
            print(f"Day {day_idx+1}: No parameters found. Skipping.")
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing Day {day_idx+1}")
        print(f"{'='*40}")

        for k, model_params in enumerate(whole_params[day_idx]):
            print(f"\n>>> Using {opt_method[k]}: {[round(p,4) for p in model_params]}")
            
            region_full_nlls = []
            region_vecc_nlls = []
            region_whittle_nlls = []

            for r_idx, region in enumerate(sub_regions):
                print(f"  [Region {r_idx+1}] {region['lat']}, {region['lon']} ... ", end="")
                
                # (A) Data Loading
                try:
                    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
                        lat_lon_resolution=lat_lon_resolution, mm_cond_number=mm_cond_number,
                        years_=years, months_=month_range,
                        lat_range=region['lat'], lon_range=region['lon']
                    )
                except Exception as e:
                    print(f"Skipping (Data Load Error: {e})")
                    continue

                hour_indices = [day_idx * 8, (day_idx + 1) * 8]

                # 1. Whittle Data (Raw)
                day_hourly_map_dw, day_aggregated_tensor_dw = data_load_instance.load_working_data(
                    df_map, hour_indices, ord_mm=None, dtype=torch.float64, keep_ori=keep_exact_loc
                )
                
                # 2. Vecchia Data (Ordered)
                day_hourly_map_vecc, day_aggregated_tensor_vecc = data_load_instance.load_working_data(
                    df_map, hour_indices, ord_mm=ord_mm, dtype=torch.float64, keep_ori=keep_exact_loc
                )

                if day_aggregated_tensor_vecc.shape[0] < 10:
                    print("Skipping (Not enough data)")
                    continue

                # Wrapper용 리스트 포장 (인덱싱 트릭)
                list_agg_vecc = [day_aggregated_tensor_vecc] * 31 
                list_map_vecc = [day_hourly_map_vecc] * 31
                list_agg_dw = [day_aggregated_tensor_dw] * 31
                list_map_dw = [day_hourly_map_dw] * 31

                # (B) Model Execution
                instance = debiased_whittle.full_vecc_dw_likelihoods(
                    list_agg_vecc, list_map_vecc, 
                    day_idx=0, 
                    params_list=model_params, 
                    lat_range=region['lat'], lon_range=region['lon']
                )
                
                instance.initiate_model_instance_vecchia(v, nns_map, mm_cond_number, nheads)
                
                res = instance.likelihood_wrapper(
                    params=instance.params_tensor,
                    cov_fun=instance.model_instance.matern_cov_aniso_STABLE_log_reparam,
                    daily_aggregated_tensors_dw=list_agg_dw,
                    daily_hourly_maps_dw=list_map_dw
                )

                full_val = res[0].item()
                vecc_val = res[1].item()
                whittle_val = res[2].item()

                region_full_nlls.append(full_val)
                region_vecc_nlls.append(vecc_val)
                region_whittle_nlls.append(whittle_val)

                print(f"Done. (F:{full_val:.1f}, V:{vecc_val:.1f}, W:{whittle_val:.1f})")

            # (C) Average Results
            if len(region_full_nlls) > 0:
                avg_full = sum(region_full_nlls) / len(region_full_nlls)
                avg_vecc = sum(region_vecc_nlls) / len(region_vecc_nlls)
                avg_whittle = sum(region_whittle_nlls) / len(region_whittle_nlls)

                print("-" * 20)
                print(f"  >>> Average Results over {len(region_full_nlls)} Regions:")
                print(f"     Full NLL (Avg Sum):    {round(avg_full, 2)}")
                print(f"     Vecchia NLL (Avg Sum): {round(avg_vecc, 2)}")
                print(f"     Whittle (Avg Sum):     {round(avg_whittle, 2)}")
                print("-" * 20)
            else:
                print("  No valid regions processed.")

if __name__ == "__main__":
    app()



