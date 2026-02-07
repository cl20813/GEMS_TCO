# 사용자 경로 설정

from pathlib import Path
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO import orderings as _orderings
import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from statsmodels.tsa.ar_model import AutoReg  # AR 모델링용

# 사용자 경로 설정
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

class load_data2:
    def __init__(self, datapath: str):
        self.datapath = datapath
    
    def load_coarse_data_dicts(
            self,
            lat_lon_resolution: List[int] = [10, 10], 
            years_: List[str] = ['2024'], 
            months_: List[int] = [7]
        ) -> Dict[str, pd.DataFrame]:
            """
            Loads coarse data (Rectangular Grid) from pickle files.
            """
            # 1. 샘플 파일 로드 (격자 정보 확인용)
            sample_df = None
            for y in years_:
                for m in months_:
                    filename = f"coarse_cen_map_rect{str(y)[2:]}_{m:02d}.pkl"
                    filepath_sample = Path(self.datapath) / f"pickle_{y}" / filename
                    
                    if filepath_sample.exists():
                        try:
                            with open(filepath_sample, 'rb') as f:
                                temp_dict = pickle.load(f)
                                if temp_dict:
                                    sample_df = temp_dict[list(temp_dict.keys())[0]]
                                    break
                        except Exception as e:
                            print(f"Warning: Failed to load sample {filepath_sample}: {e}")
                if sample_df is not None:
                    break
            
            if sample_df is None:
                print("Error: Could not find any valid pickle file to determine spatial grid.")
                return {}

            # 2. Grid 정보 추출
            rho_lat = lat_lon_resolution[0]          
            rho_lon = lat_lon_resolution[1]

            unique_lats = sample_df['Latitude'].unique()
            sorted_lats_descending = np.sort(unique_lats)[::-1] 
            lat_n = sorted_lats_descending[::rho_lat]
            
            unique_lons = sample_df['Longitude'].unique()
            sorted_lons_descending = np.sort(unique_lons)[::-1] 
            lon_n = sorted_lons_descending[::rho_lon]

            # 3. 전체 데이터 로드 및 필터링
            coarse_dicts = {}
            for year in years_:
                for month in months_:  
                    filename = f"coarse_cen_map_rect{str(year)[2:]}_{month:02d}.pkl"
                    filepath = Path(self.datapath) / f"pickle_{year}" / filename
                   
                    try:
                        with open(filepath, 'rb') as pickle_file:
                            loaded_map = pickle.load(pickle_file)
                            for key in loaded_map:
                                tmp_df = loaded_map[key]
                                coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                                coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)
                    except FileNotFoundError:
                        print(f"Warning: File not found, skipping. {filepath}")
                    except Exception as e:
                        print(f"Error reading file {filepath}: {e}")
            
            return coarse_dicts
    
    def subset_df_map(
            self, 
            df_map: Dict[str, pd.DataFrame],
            lat_range: List[float] = [0.0, 5.0],
            lon_range: List[float] = [123.0, 133.0]
        ) -> Dict[str, pd.DataFrame]:
            subsetted_map = {}
            for key, df in df_map.items():
                lat_mask = (df['Latitude'] >= lat_range[0]) & (df['Latitude'] <= lat_range[1])
                lon_mask = (df['Longitude'] >= lon_range[0]) & (df['Longitude'] <= lon_range[1])
                subsetted_map[key] = df[lat_mask & lon_mask].reset_index(drop=True).copy()
            return subsetted_map

    def get_spatial_ordering(
            self, 
            coarse_dicts: Dict[str, pd.DataFrame],
            mm_cond_number: int = 10
        ) -> Tuple[np.ndarray, np.ndarray]:
            key_idx = sorted(coarse_dicts)
            if not key_idx: return np.array([]), np.array([])

            data_for_coord = coarse_dicts[key_idx[0]]
            if data_for_coord.empty: return np.array([]), np.array([])

            x1 = data_for_coord['Longitude'].values
            y1 = data_for_coord['Latitude'].values 
            coords1 = np.stack((x1, y1), axis=-1)

            ord_mm = _orderings.maxmin_cpp(coords1)
            
            data_for_coord_reordered = data_for_coord.iloc[ord_mm].reset_index(drop=True)
            coords1_reordered = np.stack(
                (data_for_coord_reordered['Longitude'].values, data_for_coord_reordered['Latitude'].values), 
                axis=-1
            )
            
            nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
            return ord_mm, nns_map

    def load_maxmin_ordered_data_bymonthyear(
        self, 
        lat_lon_resolution: List[int] = [10, 10], 
        mm_cond_number: int = 10, 
        years_: List[str] = ['2024'], 
        months_: List[int] = [7],
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray, Dict[str, float]]:
        """
        데이터 로드 + 공간 정렬 + [핵심] AR(1) 기반 일별 Offset 계산
        """
        
        # 1. 데이터 로드 및 범위 필터링
        coarse_dicts = self.load_coarse_data_dicts(lat_lon_resolution, years_, months_)
        if not coarse_dicts: return {}, np.array([]), np.array([]), {}
        
        if lat_range and lon_range:
            coarse_dicts = self.subset_df_map(coarse_dicts, lat_range, lon_range)

        # 2. [AR(1) 분석] 일별 평균 및 잔차 계산
        print("\n--- [AR(1) Analysis for Mean Function] ---")
        
        # (1) 일별 평균 수집
        day_stats = defaultdict(list)
        for key, df in coarse_dicts.items():
            # key: "2024_07_hm01_..." -> "2024_07_01" (날짜 추출)
            day_str = "_".join(key.split('_')[:3]) 
            # 2번 컬럼(Ozone) 평균
            day_stats[day_str].append(df.iloc[:, 2].mean())

        # (2) Pandas Series로 변환 (날짜순 정렬 보장)
        daily_means = pd.Series({d: np.mean(v) for d, v in day_stats.items()}).sort_index()
        
        # (3) 월 평균 (a_y) 계산
        a_y = daily_means.mean()
        print(f"  Global Monthly Mean (a_y): {a_y:.4f}")
        
        # (4) 잔차 (e_{d,y}) 계산
        residuals = daily_means - a_y
        
        # (5) AR(1) 피팅 및 Offset 딕셔너리 생성
        day_offsets = {}
        
        try:
            # trend='n': 이미 a_y를 뺐으므로 잔차의 평균은 0 가정 (No constant)
            # 데이터 포인트가 너무 적으면(예: 2일치) 에러날 수 있음
            if len(residuals) > 2:
                model_res = AutoReg(residuals.values, lags=1, trend='n').fit()
                phi = model_res.params[0]
                print(f"  Fitted AR(1) Phi: {phi:.4f}")
            else:
                phi = 0.0
                print("  Warning: Not enough days for AR(1). Phi set to 0.")

            # (6) Offset 계산: Offset_d = a_y + (Phi * Resid_{d-1})
            # 첫째 날(Index 0)은 어제 데이터가 없으므로 그냥 a_y (잔차 0 가정)
            first_day = residuals.index[0]
            day_offsets[first_day] = a_y
            
            for i in range(1, len(residuals)):
                today_date = residuals.index[i]
                yesterday_resid = residuals.iloc[i-1] # 어제의 '관측된' 잔차
                
                # 예측된 오늘의 베이스라인
                predicted_offset = a_y + (phi * yesterday_resid)
                day_offsets[today_date] = predicted_offset
                
        except Exception as e:
            print(f"  Warning: AR(1) fitting failed ({e}). Using simple Monthly Mean.")
            for d in residuals.index:
                day_offsets[d] = a_y

        # 3. 공간 Ordering 계산
        ord_mm, nns_map = self.get_spatial_ordering(coarse_dicts, mm_cond_number)
        
        # day_offsets 딕셔너리 반환
        return coarse_dicts, ord_mm, nns_map, day_offsets

    def load_working_data(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        day_offsets: Dict[str, float], # [핵심] 위에서 만든 AR Offset 딕셔너리
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.double,
        keep_ori: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        import torch.nn.functional as F
        
        key_idx = sorted(coarse_dicts)
        if not key_idx: raise ValueError("coarse_dicts is empty")
        
        analysis_data_map = {}
        aggregated_tensor_list = []
        np_dtype = np.float64 
        
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]
        
        # 현재 배치의 날짜 확인 및 Offset 조회
        first_key = selected_keys[0]
        date_str = "_".join(first_key.split('_')[:3])
        
        # 해당 날짜의 AR 예측값 (없으면 0.0)
        current_offset_val = day_offsets.get(date_str, 0.0)

        for t_idx, key in enumerate(selected_keys):
            tmp = coarse_dicts[key].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)
            
            if ord_mm is not None:
                tmp_processed = tmp.iloc[ord_mm].reset_index(drop=True)
            else:
                tmp_processed = tmp
            
            # [Lat, Lon, Val, Time]
            target_cols = [5, 6, 2, 3] if keep_ori else [0, 1, 2, 3]
            base_tensor = torch.from_numpy(tmp_processed.iloc[:, target_cols].to_numpy(dtype=np_dtype)).to(dtype)
            
            # [타임 더미] (7개)
            dummies = F.one_hot(torch.tensor([t_idx]), num_classes=8).repeat(len(base_tensor), 1)
            dummies = dummies[:, 1:].to(dtype) 
            
            # [12번째 열: AR Offset 변수 추가]
            # 이 값은 상수처럼 반복되지만, 모델에게 "오늘의 예상 베이스라인"을 알려주는 변수(Covariate) 역할
            offset_col = torch.full((len(base_tensor), 1), current_offset_val, dtype=dtype)
            
            # 최종 결합: [Lat, Lon, Val, Time, D1...D7, AR_Offset] -> 12열
            final_tensor = torch.cat([base_tensor, dummies, offset_col], dim=1)
            
            analysis_data_map[key] = final_tensor
            aggregated_tensor_list.append(final_tensor)

        if not aggregated_tensor_list:
            return analysis_data_map, torch.empty(0, 12, dtype=dtype)

        aggregated_data = torch.cat(aggregated_tensor_list, dim=0)
        
        return analysis_data_map, aggregated_data

class exact_location_filter: # (full_vecc_dw_likelihoods):
    
    # NOTE: The __init__ was empty. If it needs to call super(), 
    # it should be added back. I'm keeping it as you provided.
    def __init__(self):
        pass
    
    # =========================================================================
    # 1. Tapering & Data Functions
    # =========================================================================
    @staticmethod
    def filter_by_location_deviation(
        hourly_maps_grid, 
        hourly_maps_exact, 
        aggregated_tensors_grid, 
        aggregated_tensors_exact,
        lat_threshold=0.025,
        lon_threshold=0.04
    ):
        """
        Filters two pairs of daily data structures (hourly maps and aggregated tensors)
        based on the absolute geographical deviation between their corresponding
        Latitude and Longitude values.

        The mask is created daily by enforcing the condition that ALL hours 
        for a given row/observation must satisfy the (lat/lon) threshold.

        Args:
            hourly_maps_grid (list[dict]): List of daily dicts (hour: tensor) with grid locations.
            hourly_maps_exact (list[dict]): List of daily dicts (hour: tensor) with exact locations.
            aggregated_tensors_grid (list[tensor]): List of daily aggregated tensors (grid locations).
            aggregated_tensors_exact (list[tensor]): List of daily aggregated tensors (exact locations).
            lat_threshold (float): Maximum acceptable absolute difference for Latitude.
            lon_threshold (float): Maximum acceptable absolute difference for Longitude.

        Returns:
            tuple: (
                filtered_hourly_maps (dict): Filtered hourly data (grid locations), keyed by day index.
                filtered_aggregated_tensors (dict): Filtered aggregated tensors (grid location tensor only), keyed by day index.
            )
        """
        # --- Configuration ---
        START_DAY = 0
        max_days = min(len(hourly_maps_grid), len(hourly_maps_exact), 
                    len(aggregated_tensors_grid), len(aggregated_tensors_exact))
        END_DAY = max_days - 1
        
        LAT_COL = 0
        LON_COL = 1

        filtered_hourly_maps = {} 
        filtered_aggregated_tensors = {} # This will now store a tensor object per day

        print(f"--- Starting Deviation Filter (Days {START_DAY}-{END_DAY}) ---")
        print(f"Thresholds: Lat <= {lat_threshold}, Lon <= {lon_threshold}")
        
        # Iterate through days
        for day_idx in range(START_DAY, END_DAY + 1):
            
            # --- 1. Filter Hourly Data & Generate Master Mask (Intersection Logic) ---
            grid_hourly_day = hourly_maps_grid[day_idx]
            exact_hourly_day = hourly_maps_exact[day_idx]
            data_agg_exact = aggregated_tensors_exact[day_idx]
            data_agg_grid = aggregated_tensors_grid[day_idx]

            time_keys = sorted(grid_hourly_day.keys())
            
            if not time_keys:
                print(f"Day {day_idx}: No time keys found in hourly data.")
                continue
            
            num_hours_per_day = len(time_keys)
            n_rows_hourly = grid_hourly_day[time_keys[0]].shape[0]
            n_rows_agg = data_agg_exact.shape[0]

            master_mask = torch.ones(n_rows_hourly, dtype=torch.bool) 

            for t_key in time_keys:
                grid_t = grid_hourly_day[t_key]
                exact_t = exact_hourly_day[t_key]
                
                lat_diff = torch.abs(grid_t[:, LAT_COL] - exact_t[:, LAT_COL])
                lon_diff = torch.abs(grid_t[:, LON_COL] - exact_t[:, LON_COL])
                
                hour_mask = (lat_diff <= lat_threshold) & (lon_diff <= lon_threshold)
                master_mask = master_mask & hour_mask

            n_final_hourly = torch.sum(master_mask).item()
            
            if n_rows_hourly > 0:
                pct_kept = (n_final_hourly / n_rows_hourly) * 100
            else:
                pct_kept = 0.0

            filtered_hourly_maps[day_idx] = {}
            for t_key in time_keys:
                # Storing the GRID location data
                filtered_hourly_maps[day_idx][t_key] = grid_hourly_day[t_key][master_mask] 
                
            print(f"Day {day_idx} (Hourly): {n_rows_hourly} -> {n_final_hourly} rows kept ({pct_kept:.2f}%)")

            
            # --- 2. Filter Aggregated Data (Expand Master Mask and return only GRID) ---
            
            expected_agg_rows = n_rows_hourly * num_hours_per_day
            
            if n_rows_agg == expected_agg_rows and n_rows_agg > 0:
                
                expanded_mask = master_mask.repeat(num_hours_per_day)
                
                # ⭐ REVISED: Store only the filtered GRID location tensor
                filtered_aggregated_tensors[day_idx] = data_agg_grid[expanded_mask]
                
                rows_filtered_agg = filtered_aggregated_tensors[day_idx].shape[0]
                expected_filtered_rows = n_final_hourly * num_hours_per_day
                
                if rows_filtered_agg == expected_filtered_rows:
                    print(f"Day {day_idx} (Aggregated): {n_rows_agg} -> {rows_filtered_agg} rows kept (Shape consistent with {n_final_hourly} filtered hourly rows).")
                else:
                    print(f"Day {day_idx} (Aggregated): {n_rows_agg} -> {rows_filtered_agg} rows kept (Unexpected final row count).")
                
            else:
                print(f"⚠️ Warning: Day {day_idx} Aggregated data row count ({n_rows_agg}) does not match expected stacked row count ({expected_agg_rows}). Skipping Aggregated filter for this day.")

            print("-" * 30)

        print("-" * 40)
        print("Filtering Complete.")
        print(f"Filtered {len(filtered_hourly_maps)} days of hourly data.")
        print(f"Filtered {len(filtered_aggregated_tensors)} days of aggregated data.")

        return filtered_hourly_maps, filtered_aggregated_tensors

    @staticmethod
    def print_deviation_report(name, orig, grid, mask, threshold):
        # Filter the 1D arrays used for stats
        orig_subset = orig[mask].detach().cpu().numpy()
        grid_subset = grid[mask].detach().cpu().numpy()
        
        deviation = orig_subset - grid_subset
        large_deviations_mask = np.abs(deviation) > threshold
        count = np.sum(large_deviations_mask)
        total_points = len(deviation)
        
        if total_points > 0:
            percentage = (count / total_points) * 100
        else:
            percentage = 0.0

        print(f"\n--- {name} Fine-Scale Deviation Report (Post-Filter) ---")
        print(f"Threshold: > {threshold}")
        print(f"Total data points: {total_points}")
        print(f"Points with large deviation: {count}")
        print(f"Percentage with large deviation: {percentage:.2f}%")
    
    @staticmethod

    def get_spatial_ordering(
            coarse_dicts,
            mm_cond_number: int = 10
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Computes the MaxMin ordering and nearest neighbors from a sample 
            of the coarse data.
            
            Assumes all dataframes in coarse_dicts share the same spatial grid.
            """
            key_idx = sorted(coarse_dicts)
            if not key_idx:
                print("Warning: coarse_dicts is empty, cannot compute ordering.")
                return np.array([]), np.array([])

            # Extract first hour data because all data shares the same spatial grid
            data_for_coord = coarse_dicts[key_idx[0]]
            x1 = data_for_coord[:,0]
            y1 = data_for_coord[:,1]
            coords1 = np.stack((x1, y1), axis=-1)

            # Calculate MaxMin ordering
            ord_mm = _orderings.maxmin_cpp(coords1)
            
            # Reorder coordinates to find nearest neighbors
            data_for_coord_reordered = data_for_coord[ord_mm]
            coords1_reordered = np.stack(
                (data_for_coord_reordered[:,0], data_for_coord_reordered[:,1]), 
                axis=-1
            )
            
            # Calculate nearest neighbors map
            nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
            
            return ord_mm, nns_map
    '''
    ord_mm, nns_map = get_spatial_ordering(a[0],8)
    '''

