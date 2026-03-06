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
from scipy.spatial import KDTree

# 02/07/26 no ar, but time dummies one intercept,lat, lon 

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
            File pattern: coarse_cen_map_rect{YY}_{MM}.pkl
            """
            
            # 1. 샘플 파일 로드 (격자 정보 확인용)
            # 하드코딩 없이 입력된 연도/월 리스트의 첫 번째를 사용
            sample_df = None
            for y in years_:
                for m in months_:
                    # [수정됨] 파일명 포맷: rect 적용
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
                                # Coarsening Filter
                                coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                                coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)
                    except FileNotFoundError:
                        print(f"Warning: File not found, skipping. {filepath}")
                    except Exception as e:
                        print(f"Error reading file {filepath}: {e}")
            
            return coarse_dicts

    def load_whittle_data_dicts(
            self,
            lat_lon_resolution: List[int] = [10, 10], 
            years_: List[str] = ['2024'], 
            months_: List[int] = [7]
        ) -> Dict[str, pd.DataFrame]:
            """
            Loads coarse data explicitly for Debiased Whittle from pickle files.
            File pattern: coarse_cen_map_rect_whittle_{YY}_{MM}.pkl
            """
            
            # 1. 샘플 파일 로드 (격자 정보 확인용)
            sample_df = None
            for y in years_:
                for m in months_:
                    # [수정됨] 파일명 포맷: rect_whittle_ 적용
                    filename = f"coarse_cen_map_rect_whittle_{str(y)[2:]}_{m:02d}.pkl"
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
                    # [수정됨] 전체 데이터 로드 시에도 파일명 변경
                    filename = f"coarse_cen_map_rect_whittle_{str(year)[2:]}_{month:02d}.pkl"
                    filepath = Path(self.datapath) / f"pickle_{year}" / filename
                   
                    try:
                        with open(filepath, 'rb') as pickle_file:
                            loaded_map = pickle.load(pickle_file)
                            for key in loaded_map:
                                tmp_df = loaded_map[key]
                                # Coarsening Filter
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
            """
            Subsets each DataFrame to a specific lat/lon range.
            """
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
            """
            Computes MaxMin ordering based on the first available dataframe.
            """
            key_idx = sorted(coarse_dicts)
            if not key_idx:
                print("Warning: coarse_dicts is empty, cannot compute ordering.")
                return np.array([]), np.array([])

            data_for_coord = coarse_dicts[key_idx[0]]
            
            # Safety check
            if data_for_coord.empty:
                return np.array([]), np.array([])

            # Coordinate Extraction
            x1 = data_for_coord['Longitude'].values
            y1 = data_for_coord['Latitude'].values 
            coords1 = np.stack((x1, y1), axis=-1)

            # MaxMin Ordering (C++ ext)
            ord_mm = _orderings.maxmin_cpp(coords1)
            
            # Reorder & Find Nearest Neighbors
            data_for_coord_reordered = data_for_coord.iloc[ord_mm].reset_index(drop=True)
            coords1_reordered = np.stack(
                (data_for_coord_reordered['Longitude'].values, data_for_coord_reordered['Latitude'].values), 
                axis=-1
            )
            
            nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
            
            return ord_mm, nns_map
    '''
    def load_maxmin_ordered_data_bymonthyear(
            self, 
            lat_lon_resolution: List[int] = [10, 10], 
            mm_cond_number: int = 10, 
            years_: List[str] = ['2024'], 
            months_: List[int] = [7],
            lat_range: Optional[List[float]] = None,
            lon_range: Optional[List[float]] = None
        ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
            
            # 1. Load Data
            coarse_dicts = self.load_coarse_data_dicts(
                lat_lon_resolution=lat_lon_resolution,
                years_=years_,
                months_=months_
            )
            
            if not coarse_dicts:
                return {}, np.array([]), np.array([])
                
            # 2. Subset Data (Before Ordering)
            if lat_range is not None and lon_range is not None:
                coarse_dicts = self.subset_df_map(
                    coarse_dicts,
                    lat_range=lat_range,
                    lon_range=lon_range
                )
                
            # 3. Compute Ordering (on Subsetted Data)
            ord_mm, nns_map = self.get_spatial_ordering(
                coarse_dicts=coarse_dicts, 
                mm_cond_number=mm_cond_number
            )
            
            return coarse_dicts, ord_mm, nns_map

    def load_working_data(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.double,
        keep_ori: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        import torch.nn.functional as F
        
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")
        
        analysis_data_map = {}
        aggregated_tensor_list = []
        
        np_dtype = np.float64 # 사용자 dtype 유지
        
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]
        
        for t_idx, key in enumerate(selected_keys):
            tmp = coarse_dicts[key].copy()
            
            # --- [수정 금지] 사용자 원본 시간 로직 유지 ---
            # 만약 477721 그대로를 원하시면 아래 줄을 주석 처리하거나, 
            # 기존처럼 477700을 빼고 싶으시면 그대로 두시면 됩니다.
            # 여기서는 사용자님의 기존 코드 방식을 존중하여 유지합니다.
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)
            
            # Ordering 적용
            if ord_mm is not None:
                tmp_processed = tmp.iloc[ord_mm].reset_index(drop=True)
            else:
                tmp_processed = tmp
            
            # 기본 4열 선택 [Lat, Lon, O3, Hours_elapsed]
            if keep_ori:
                # [Src_Lat(5), Src_Lon(6), O3(2), Hours_elapsed(3)]
                tmp_data_df = tmp_processed.iloc[:, [5, 6, 2, 3]] 
            else:
                # [Lat(0), Lon(1), O3(2), Hours_elapsed(3)]
                tmp_data_df = tmp_processed.iloc[:, [0, 1, 2, 3]]
        
            base_tensor = torch.from_numpy(tmp_data_df.to_numpy(dtype=np_dtype)).to(dtype)
            
            # --- [추가] 시간 더미 변수 생성 (7개) ---
            # t_idx(0~7)를 사용하여 0시를 제외한 7개의 더미 생성
            dummies = F.one_hot(torch.tensor([t_idx]), num_classes=8).repeat(len(base_tensor), 1)
            dummies = dummies[:, 1:].to(dtype) # 첫 번째 열 제외 (Intercept용)
            
            # 최종 11열 결합
            final_tensor = torch.cat([base_tensor, dummies], dim=1)
            
            analysis_data_map[key] = final_tensor
            aggregated_tensor_list.append(final_tensor)

        if not aggregated_tensor_list:
            return analysis_data_map, torch.empty(0, 11, dtype=dtype)

        aggregated_data = torch.cat(aggregated_tensor_list, dim=0)
        
        return analysis_data_map, aggregated_data
    
        ''' 
    def load_maxmin_ordered_data_bymonthyear(
        self, 
        lat_lon_resolution: List[int] = [10, 10], 
        mm_cond_number: int = 10, 
        years_: List[str] = ['2024'], 
        months_: List[int] = [7],
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None,
        is_whittle: bool = True  # [추가됨] Whittle용 데이터를 불러올지 선택하는 스위치
    ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray, float]:
        
        from collections import defaultdict

        # 1. 데이터 로드 (스위치에 따라 분기)
        if is_whittle:
            print("Loading Strict Whittle Data...")
            coarse_dicts = self.load_whittle_data_dicts(lat_lon_resolution, years_, months_)
        else:
            print("Loading Standard Coarse Data...")
            coarse_dicts = self.load_coarse_data_dicts(lat_lon_resolution, years_, months_)
            
        if not coarse_dicts: return {}, np.array([]), np.array([]), 0.0
        
        if lat_range and lon_range:
            coarse_dicts = self.subset_df_map(coarse_dicts, lat_range, lon_range)

        # 2. [수정됨] 월 평균(Global Monthly Mean) 계산
        total_ozone_values = []
        for key, df in coarse_dicts.items():
            # 인덱스 2번이 Ozone이라고 가정 (혹은 컬럼명 확인 필요)
            total_ozone_values.append(df.iloc[:, 2].values)
            
        if total_ozone_values:
            all_values = np.concatenate(total_ozone_values)
            
            # [매우 중요] NaN 값을 무시하고 평균을 구해야 함!
            monthly_mean = np.nanmean(all_values) 
            
            print(f"--- Global Monthly Mean for {years_[0]}-{months_[0]}: {monthly_mean:.4f} ---")
        else:
            monthly_mean = 0.0

        # 3. 공간 Ordering 계산
        ord_mm, nns_map = self.get_spatial_ordering(coarse_dicts, mm_cond_number)
        
        # monthly_mean을 반환값에 추가
        return coarse_dicts, ord_mm, nns_map, monthly_mean

    def load_working_data(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        monthly_mean: float = 0.0, # [추가] 월 평균 값을 인자로 받음
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.double,
        keep_ori: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        import torch.nn.functional as F
        
        key_idx = sorted(coarse_dicts)
        analysis_data_map = {}
        aggregated_tensor_list = []
        np_dtype = np.float64 
        
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]
        
        for t_idx, key in enumerate(selected_keys):
            tmp = coarse_dicts[key].copy()
            
            # 시간 정규화
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)
            
            # [핵심] 월 평균 반영 (Centering)
            # 오존 값에서 월 평균을 미리 뺍니다. 
            # 이렇게 하면 Intercept는 "월 평균으로부터의 편차"를 학습하게 됩니다.
            # 원본 데이터프레임을 건드리지 않기 위해 복사본이나 iloc 할당 주의
            if keep_ori:
                # 2번 컬럼이 Ozone
                ozone_vals = tmp.iloc[:, 2].values - monthly_mean
            else:
                ozone_vals = tmp.iloc[:, 2].values - monthly_mean
            
            # Ordering 적용
            if ord_mm is not None:
                tmp_processed = tmp.iloc[ord_mm].reset_index(drop=True)
                ozone_vals = ozone_vals[ord_mm] # 오존 값도 순서 변경
            else:
                tmp_processed = tmp
            
            # 텐서 생성 준비
            # [Lat, Lon, Val(Centered), Time]
            target_cols = [5, 6, 2, 3] if keep_ori else [0, 1, 2, 3]
            base_numpy = tmp_processed.iloc[:, target_cols].to_numpy(dtype=np_dtype)
            
            # Centered Ozone 값으로 덮어쓰기 (인덱스 2번)
            base_numpy[:, 2] = ozone_vals
            
            base_tensor = torch.from_numpy(base_numpy).to(dtype)
            
            # [타임 더미 생성] (8개 시간대 -> 7개 더미)
            # t_idx: 0~7
            dummies = F.one_hot(torch.tensor([t_idx]), num_classes=8).repeat(len(base_tensor), 1)
            dummies = dummies[:, 1:].to(dtype) # 첫 번째 시간대(0)를 Reference로 제외 -> 7개
            
            # 최종 11열 결합: [Lat, Lon, Val_Centered, Time, D1 ... D7]
            final_tensor = torch.cat([base_tensor, dummies], dim=1)
            
            analysis_data_map[key] = final_tensor
            aggregated_tensor_list.append(final_tensor)

        if not aggregated_tensor_list:
            return analysis_data_map, torch.empty(0, 11, dtype=dtype)

        aggregated_data = torch.cat(aggregated_tensor_list, dim=0)
        
        return analysis_data_map, aggregated_data

    '''
    # To replace load_working_data_byday (with reordering, as double):
    analysis_map_mm, agg_data_mm = self.load_working_data(
        coarse_dicts, 
        idx_for_datamap, 
        ord_mm=your_ord_mm_array, 
        dtype=torch.double
    )

    # To replace load_working_data_byday_wo_mm (no reordering, as float):
    analysis_map_no_mm, agg_data_no_mm = self.load_working_data(
        coarse_dicts, 
        idx_for_datamap, 
        ord_mm=None,  # or just omit it
        dtype=torch.float # or just omit it
    )
    
    '''


### 2026 03 03
# Missing 자동 차단 (Rule 1): lat_diffs와 lon_diffs를 각각 계산하여,
#  격자 크기의 절반(0.022, 0.0315)을 넘어가는 데이터는 남의 방 데이터를 훔쳐 온 것으로 간주하고 가차 없이 NaN으로 만듦.
# 네가 말한 대로 8시간 내내 공간 구조(상대적 거리)가 같기 때문에, ord_mm을 매시간 새로 구하는 연산 낭비 없이 한 번 구해서 끝까지 돌려써.


import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from scipy.spatial import KDTree
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

class load_data_dynamic_grid:
    def __init__(self, 
                 datapath: str = None,         
                 master_lat: float = 1.982,       
                 master_lon: float = 130.986,     
                 lat_bounds: List[float] = [-3.0, 2.0], 
                 lon_bounds: List[float] = [121.0, 131.0],
                 v_drift_lat: float = 0.0,        
                 v_drift_lon: float = -0.0048):   
        
        self.datapath = datapath
        self.lat_step = 0.044
        self.lon_step = 0.063
        
        self.master_lat = master_lat
        self.master_lon = master_lon
        self.v_drift_lat = v_drift_lat
        self.v_drift_lon = v_drift_lon
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        
        # Missing 처리 기준 (격자 크기의 절반)
        self.lat_threshold = self.lat_step / 2.0
        self.lon_threshold = self.lon_step / 2.0

        self.generate_base_grid()

    # ==========================================================
    # 1. Dynamic Grid 생성
    # ==========================================================
    def generate_base_grid(self):
        n_south = int(np.ceil((self.master_lat - self.lat_bounds[0]) / self.lat_step))
        n_north = int(np.ceil((self.lat_bounds[1] - self.master_lat) / self.lat_step))
        n_west = int(np.ceil((self.master_lon - self.lon_bounds[0]) / self.lon_step))
        n_east = int(np.ceil((self.lon_bounds[1] - self.master_lon) / self.lon_step))
        
        base_lats = self.master_lat + np.arange(-n_south, n_north + 1) * self.lat_step
        base_lons = self.master_lon + np.arange(-n_west, n_east + 1) * self.lon_step
        
        base_lats = base_lats[(base_lats >= self.lat_bounds[0]) & (base_lats <= self.lat_bounds[1])]
        base_lons = base_lons[(base_lons >= self.lon_bounds[0]) & (base_lons <= self.lon_bounds[1])]
        
        xv, yv = np.meshgrid(base_lats, base_lons, indexing='ij')
        self.base_grid_coords = np.stack((xv.flatten(), yv.flatten()), axis=-1)
        return self.base_grid_coords

    def get_dynamic_grid_for_hour(self, t_idx: int):
        shift_lat = t_idx * self.v_drift_lat
        shift_lon = t_idx * self.v_drift_lon
        shifted_coords = self.base_grid_coords + np.array([shift_lat, shift_lon])
        return shifted_coords

    def get_spatial_ordering(self, coords: np.ndarray, mm_cond_number: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        try:
            ord_mm = _orderings.maxmin_cpp(coords)
            coords_reordered = coords[ord_mm]
            nns_map = _orderings.find_nns_l2(locs=coords_reordered, max_nn=mm_cond_number)
            return ord_mm, nns_map
        except NameError:
            print("Warning: _orderings module is not defined. Returning linear order.")
            return np.arange(len(coords)), np.array([])

    def load_raw_data_dicts(self, years_: List[str], months_: List[int], is_whittle: bool) -> Dict[str, pd.DataFrame]:
        raw_dicts = {}
        for year in years_:
            for month in months_:  
                prefix = "rect_whittle_" if is_whittle else "rect"
                filename = f"coarse_cen_map_{prefix}{str(year)[2:]}_{month:02d}.pkl"
                filepath = Path(self.datapath) / f"pickle_{year}" / filename
                try:
                    with open(filepath, 'rb') as pickle_file:
                        loaded_map = pickle.load(pickle_file)
                        for key in loaded_map:
                            tmp_df = loaded_map[key]
                            lat_mask = (tmp_df['Latitude'] >= self.lat_bounds[0]) & (tmp_df['Latitude'] <= self.lat_bounds[1])
                            lon_mask = (tmp_df['Longitude'] >= self.lon_bounds[0]) & (tmp_df['Longitude'] <= self.lon_bounds[1])
                            raw_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[lat_mask & lon_mask].reset_index(drop=True)
                except FileNotFoundError:
                    pass
        return raw_dicts

    # ==========================================================
    # 2. 데이터 로드 및 격자 매핑 (사용자 기존 인풋 형식 완벽 호환)
    # ==========================================================
    def load_maxmin_ordered_data_bymonthyear(
        self, 
        lat_lon_resolution: List[int] = [10, 10], # 호환성을 위해 남겨둠
        mm_cond_number: int = 10, 
        years_: List[str] = ['2024'], 
        months_: List[int] = [7],
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None,
        is_whittle: bool = True  
    ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray, float]:
        
        # Bounding box 업데이트
        if lat_range: self.lat_bounds = lat_range
        if lon_range: self.lon_bounds = lon_range
        self.generate_base_grid()

        print(f"Loading Raw Data (Whittle={is_whittle})...")
        raw_dicts = self.load_raw_data_dicts(years_, months_, is_whittle)
        
        if not raw_dicts:
            return {}, np.array([]), np.array([]), 0.0

        # 월 평균(Global Monthly Mean) 계산
        total_ozone_values = []
        for df in raw_dicts.values():
            ozone_col = 'Ozone' if 'Ozone' in df.columns else df.columns[2]
            total_ozone_values.append(df[ozone_col].values)
            
        monthly_mean = np.nanmean(np.concatenate(total_ozone_values)) if total_ozone_values else 0.0
        print(f"--- Global Monthly Mean for {years_[0]}-{months_[0]}: {monthly_mean:.4f} ---")

        # 공간 Ordering 계산 (Base Grid 기준 1회만)
        ord_mm, nns_map = None, None
        if not is_whittle:
            ord_mm, nns_map = self.get_spatial_ordering(self.base_grid_coords, mm_cond_number)

        # Dynamic Grid 매칭 (df_map 생성)
        df_map = {}
        sorted_keys = sorted(raw_dicts.keys())

        for i, key in enumerate(sorted_keys):
            t_idx = i % 8 # 0~7 시간대 인덱스
            df_raw = raw_dicts[key]
            
            if len(df_raw) == 0: continue
            
            grid_coords_h = self.get_dynamic_grid_for_hour(t_idx)
            raw_coords = df_raw[['Latitude', 'Longitude']].values
            
            ozone_col = 'Ozone' if 'Ozone' in df_raw.columns else df_raw.columns[2]
            raw_vals = df_raw[ozone_col].values
            hours_elapsed = df_raw['Hours_elapsed'].values
            
            # KDTree 매칭
            tree = KDTree(raw_coords)
            distances, indices = tree.query(grid_coords_h)
            
            matched_vals = raw_vals[indices]
            matched_raw_coords = raw_coords[indices]
            matched_hours = hours_elapsed[indices]
            
            # Missing 차단 (탈선 마스킹)
            lat_diffs = np.abs(grid_coords_h[:, 0] - matched_raw_coords[:, 0])
            lon_diffs = np.abs(grid_coords_h[:, 1] - matched_raw_coords[:, 1])
            invalid_mask = (lat_diffs > self.lat_threshold) | (lon_diffs > self.lon_threshold)
            
            matched_vals[invalid_mask] = np.nan
            matched_raw_coords[invalid_mask] = np.nan
            
            # 디버깅하기 좋은 포맷으로 DataFrame 조립

            df_matched = pd.DataFrame({
                'Latitude': self.base_grid_coords[:, 0],        # 무조건 고정된 Base Grid 좌표 부여!
                'Longitude': self.base_grid_coords[:, 1],       # 무조건 고정된 Base Grid 좌표 부여!
                ozone_col: matched_vals,                        
                'Hours_elapsed': matched_hours,
                'Source_Latitude': matched_raw_coords[:, 0],    
                'Source_Longitude': matched_raw_coords[:, 1]
            })
            
            df_map[key] = df_matched

        return df_map, ord_mm, nns_map, monthly_mean

    # ==========================================================
    # 3. 텐서 변환 및 Time Dummies 생성 (사용자 기존 함수 완벽 이식)
    # ==========================================================
    def load_working_data(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        monthly_mean: float = 0.0, 
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.double,
        keep_ori: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        key_idx = sorted(coarse_dicts)
        analysis_data_map = {}
        aggregated_tensor_list = []
        np_dtype = np.float64 
        
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]
        
        for t_idx, key in enumerate(selected_keys):
            tmp = coarse_dicts[key].copy()
            
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)
            
            ozone_col = 'Ozone' if 'Ozone' in tmp.columns else tmp.columns[2]
            ozone_vals = tmp[ozone_col].values - monthly_mean
            
            if ord_mm is not None:
                tmp = tmp.iloc[ord_mm].reset_index(drop=True)
                ozone_vals = ozone_vals[ord_mm] 
            
            if keep_ori:
                target_data = tmp[['Source_Latitude', 'Source_Longitude', ozone_col, 'Hours_elapsed']].copy()
            else:
                target_data = tmp[['Latitude', 'Longitude', ozone_col, 'Hours_elapsed']].copy()
            
            base_numpy = target_data.to_numpy(dtype=np_dtype)
            base_numpy[:, 2] = ozone_vals 
            
            base_tensor = torch.from_numpy(base_numpy).to(dtype)
            
            dummies = F.one_hot(torch.tensor([t_idx]), num_classes=8).repeat(len(base_tensor), 1)
            dummies = dummies[:, 1:].to(dtype) 
            
            final_tensor = torch.cat([base_tensor, dummies], dim=1)
            
            analysis_data_map[key] = final_tensor
            aggregated_tensor_list.append(final_tensor)

        if not aggregated_tensor_list:
            return analysis_data_map, torch.empty(0, 11, dtype=dtype)

        aggregated_data = torch.cat(aggregated_tensor_list, dim=0)
        
        return analysis_data_map, aggregated_data
    





### 아래는  test_regular_grid_seemissings_122225.ipynb에서 쓰인건데
### 실제 격자가 0.044, 0.063 이 맞는지 확인하는데 쓰였어
    

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

