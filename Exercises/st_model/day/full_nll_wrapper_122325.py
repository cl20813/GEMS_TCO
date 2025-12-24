# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import torch
import typer
import numpy as np
from typing import List
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2
from GEMS_TCO import debiased_whittle 

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
    # 규칙: Lat (Window 3, Stride 2), Lon (Window 5, Stride 5)
    lat_window, lat_stride = 3, 2
    lon_window, lon_stride = 5, 5

    sub_regions = []
    
    # Latitude Sliding
    for lat_s in range(total_lat_range[0], total_lat_range[1], lat_stride):
        lat_e = lat_s + lat_window
        if lat_e > total_lat_range[1]: break # 범위를 넘어가면 중단

        # Longitude Sliding
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
    # 2. Parameters 정의
    # =========================================================================
    day1_vl = [4.2843, 1.7136, 0.4887, -3.7712, 0.0202, -0.1616, -14.7220]
    day1_dwl = [4.2739, 1.8060, 0.7948, -3.3599, 0.0223, -0.1672, -11.8381]
    day2_vl = [3.7440, 1.2167, 0.6473, -4.0566, 0.00106, -0.2202, 0.7403]
    day2_dwl =[4.1200, 1.6540, 0.8909, -3.4966, -0.0263, -0.2601, -0.0986]
    day3_vl = [4.39425, 1.60585, 0.50261, -4.30459, -0.03894, -0.2451, 0.26052]
    day3_dwl = [4.0950, 1.6663, 0.6876, -3.3118, -0.0500, -0.2666, -0.5033]
    day4_vl = [3.9555, 1.4425, 0.7809, -4.0119, 0.03009, -0.1465, 0.1118]
    day4_dwl = [3.9351, 1.8070, 1.0980, -3.5154, 0.0214, -0.1712, -0.5348]

    whole_params = [
        [day1_vl, day1_dwl], [day2_vl, day2_dwl], 
        [day3_vl, day3_dwl], [day4_vl, day4_dwl]
    ]
    opt_method = ['Vecchia_Params', 'Whittle_Params']

    # 데이터 로더 인스턴스 (경로는 고정이므로 밖에서 생성)
    data_load_instance = load_data2(config.amarel_data_load_path)

    # =========================================================================
    # 3. Main Loop (Days -> Sub-Regions)
    # =========================================================================
    for day_idx in days_list:
        if day_idx >= len(whole_params): continue
        print(f"\n{'='*40}")
        print(f"Processing Day {day_idx+1}")
        print(f"{'='*40}")

        # 파라미터 세트별로 순회 (Vecchia 파라미터, Whittle 파라미터 각각 적용해보기 위함인 듯)
        for k, model_params in enumerate(whole_params[day_idx]):
            print(f"\n>>> Using {opt_method[k]}: {[round(p,4) for p in model_params]}")
            
            # 평균 계산을 위한 리스트
            region_full_nlls = []
            region_vecc_nlls = []
            region_whittle_nlls = []

            # 4개 영역 순회
            for r_idx, region in enumerate(sub_regions):
                print(f"  [Region {r_idx+1}] {region['lat']}, {region['lon']} ... ", end="")
                
                # -----------------------------------------------------------------
                # (A) Data Loading (Region Specific)
                # MaxMin Ordering과 Neighbor는 해당 Grid에 맞게 새로 계산되어야 하므로 루프 안에서 로드
                # -----------------------------------------------------------------
                try:
                    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
                        lat_lon_resolution=lat_lon_resolution, mm_cond_number=mm_cond_number,
                        years_=years, months_=month_range,
                        lat_range=region['lat'], lon_range=region['lon']
                    )
                except Exception as e:
                    print(f"Skipping (Data Load Error: {e})")
                    continue

                # 해당 Day의 데이터 추출
                hour_indices = [day_idx * 8, (day_idx + 1) * 8]

                # 1. Whittle Data (Raw)
                day_hourly_map_dw, day_aggregated_tensor_dw = data_load_instance.load_working_data(
                    df_map, hour_indices, ord_mm=None, dtype=torch.float64, keep_ori=keep_exact_loc
                )
                
                # 2. Vecchia Data (Ordered)
                day_hourly_map_vecc, day_aggregated_tensor_vecc = data_load_instance.load_working_data(
                    df_map, hour_indices, ord_mm=ord_mm, dtype=torch.float64, keep_ori=keep_exact_loc
                )

                # 데이터가 너무 적으면 스킵
                if day_aggregated_tensor_vecc.shape[0] < 10:
                    print("Skipping (Not enough data)")
                    continue

                # 리스트 형태로 포장 (Wrapper가 리스트 입력을 기대하므로)
                # 단일 Day 처리를 위해 리스트에 넣음
                list_agg_vecc = [day_aggregated_tensor_vecc] * 31 # 인덱스 맞추기용 더미 (실제론 day_idx만 씀)
                list_map_vecc = [day_hourly_map_vecc] * 31
                list_agg_dw = [day_aggregated_tensor_dw] * 31
                list_map_dw = [day_hourly_map_dw] * 31

                # -----------------------------------------------------------------
                # (B) Model Execution
                # -----------------------------------------------------------------
                instance = debiased_whittle.full_vecc_dw_likelihoods(
                    list_agg_vecc, list_map_vecc, 
                    day_idx=0, # 리스트를 단일 데이터로 채웠으므로 0으로 접근
                    params_list=model_params, 
                    lat_range=region['lat'], lon_range=region['lon']
                )
                
                # Vecchia 초기화
                instance.initiate_model_instance_vecchia(v, nns_map, mm_cond_number, nheads)
                
                # Wrapper 호출
                # 여기서 day_idx는 instance 생성시 0으로 고정했으므로, Wrapper 내부도 0번째 데이터를 씀
                # 하지만 Wrapper가 내부에서 self.day_idx를 쓰므로, 위에서 day_idx=0으로 넘긴 것이 중요함.
                res = instance.likelihood_wrapper(
                    params=instance.params_tensor,
                    cov_fun=instance.model_instance.matern_cov_aniso_STABLE_log_reparam,
                    daily_aggregated_tensors_dw=list_agg_dw,
                    daily_hourly_maps_dw=list_map_dw
                )

                # 결과 저장
                full_val = res[0].item()
                vecc_val = res[1].item()
                whittle_val = res[2].item()

                region_full_nlls.append(full_val)
                region_vecc_nlls.append(vecc_val)
                region_whittle_nlls.append(whittle_val)

                print(f"Done. (F:{full_val:.1f}, V:{vecc_val:.1f}, W:{whittle_val:.1f})")

            # -----------------------------------------------------------------
            # (C) Calculate & Print Average across Regions
            # -----------------------------------------------------------------
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



