from pathlib import Path

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings as _orderings

import sys
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)


class load_data_dynamic_processed:
    def __init__(self, datapath: str):
        self.datapath = datapath

    def load_tco_grid_dicts(self, years_: List[str], months_: List[int]) -> Dict[str, pd.DataFrame]:
        """
        tco_grid_{yy}_{mm}.pkl 파일 로드.
        Vecchia / Debiased Whittle 공통으로 사용.
        """
        coarse_dicts = {}
        for year in years_:
            for month in months_:
                filename = f"tco_grid_{str(year)[2:]}_{month:02d}.pkl"
                filepath = Path(self.datapath) / f"pickle_{year}" / filename
                try:
                    with open(filepath, 'rb') as f:
                        loaded_map = pickle.load(f)
                        for key in loaded_map:
                            coarse_dicts[f"{year}_{month:02d}_{key}"] = loaded_map[key].reset_index(drop=True)
                except FileNotFoundError:
                    print(f"Warning: File not found {filepath}")
        return coarse_dicts

    def get_spatial_ordering(
        self,
        coarse_dicts: Dict[str, pd.DataFrame],
        mm_cond_number: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        첫 번째 DataFrame의 격자 좌표로 MaxMin ordering 및 NNS map 계산.
        모든 시간대가 동일한 격자를 공유하므로 한 번만 계산.
        """
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            print("Warning: coarse_dicts is empty, cannot compute ordering.")
            return np.array([]), np.array([])

        data_for_coord = coarse_dicts[key_idx[0]]
        if data_for_coord.empty:
            return np.array([]), np.array([])

        coords = np.stack(
            (data_for_coord['Longitude'].values, data_for_coord['Latitude'].values),
            axis=-1
        )
        ord_mm = _orderings.maxmin_cpp(coords)

        coords_reordered = coords[ord_mm]
        nns_map = _orderings.find_nns_l2(locs=coords_reordered, max_nn=mm_cond_number)

        return ord_mm, nns_map

    def load_maxmin_ordered_data_bymonthyear(
        self,
        lat_lon_resolution: List[int] = [1, 1],
        mm_cond_number: int = 10,
        years_: List[str] = ['2024'],
        months_: List[int] = [7],
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None,
        is_whittle: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        데이터 로드 → bounding box 필터 → 월평균 계산 → ordering(Vecchia만).

        is_whittle=True  : ord_mm, nns_map = None (DW는 ordering 불필요)
        is_whittle=False : C++ MaxMin ordering + NNS map 생성 (Vecchia용)
        """
        coarse_dicts = self.load_tco_grid_dicts(years_, months_)
        if not coarse_dicts:
            return {}, None, None, 0.0

        # Bounding box 필터
        if lat_range is not None and lon_range is not None:
            filtered = {}
            for key, df in coarse_dicts.items():
                mask = (
                    (df['Latitude']  >= lat_range[0]) & (df['Latitude']  <= lat_range[1]) &
                    (df['Longitude'] >= lon_range[0]) & (df['Longitude'] <= lon_range[1])
                )
                filtered[key] = df[mask].reset_index(drop=True)
            coarse_dicts = filtered

        # 월평균 계산
        ozone_vals = [
            pd.to_numeric(df['ColumnAmountO3'], errors='coerce').values
            for df in coarse_dicts.values()
        ]
        monthly_mean = float(np.nanmean(np.concatenate(ozone_vals))) if ozone_vals else 0.0
        print(f"--- Global Monthly Mean for {years_[0]}-{months_[0]}: {monthly_mean:.4f} ---")

        if is_whittle:
            return coarse_dicts, None, None, monthly_mean

        print("--- Generating NNS Map for Vecchia (C++ Accelerated) ---")
        ord_mm, nns_map = self.get_spatial_ordering(coarse_dicts, mm_cond_number)
        return coarse_dicts, ord_mm, nns_map, monthly_mean

    def load_working_data(
        self,
        coarse_dicts: Dict[str, pd.DataFrame],
        monthly_mean: float = 0.0,
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.double,
        keep_ori: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        일별 시간 구간의 데이터를 텐서로 변환.

        keep_ori=True  : Source_Latitude/Longitude 사용 (Vecchia — 실측 좌표)
        keep_ori=False : Latitude/Longitude 사용        (DW — 격자 좌표)

        출력 텐서 열 구성 (11열):
        [Lat, Lon, O3_centered, Hours_elapsed, D1, D2, D3, D4, D5, D6, D7]
        """
        key_idx = sorted(coarse_dicts)
        analysis_data_map = {}
        aggregated_tensor_list = []
        np_dtype = np.float64
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]

        for t_idx, key in enumerate(selected_keys):
            tmp = coarse_dicts[key].copy()

            # NaN Hours_elapsed 복구 (forward binning 후 일부 격자에서 발생 가능)
            if not tmp['Hours_elapsed'].dropna().empty:
                valid_time = tmp['Hours_elapsed'].dropna().median()
                tmp['Hours_elapsed'] = tmp['Hours_elapsed'].fillna(valid_time)

            # 시간 정규화
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)

            # Vecchia ordering 적용
            if ord_mm is not None:
                tmp = tmp.iloc[ord_mm].reset_index(drop=True)

            # 오존 centering
            ozone_vals = pd.to_numeric(tmp['ColumnAmountO3'], errors='coerce').values - monthly_mean

            # 좌표 선택
            if keep_ori:
                target_data = tmp[['Source_Latitude', 'Source_Longitude', 'ColumnAmountO3', 'Hours_elapsed']].copy()
            else:
                target_data = tmp[['Latitude', 'Longitude', 'ColumnAmountO3', 'Hours_elapsed']].copy()

            base_numpy = target_data.to_numpy(dtype=np_dtype)
            base_numpy[:, 2] = ozone_vals  # centered O3 덮어쓰기
            base_tensor = torch.from_numpy(base_numpy).to(dtype)

            # 시간 더미 (8시간대 → 7개 더미, 첫 시간대 reference)
            dummies = F.one_hot(torch.tensor([t_idx]), num_classes=8).repeat(len(base_tensor), 1)
            dummies = dummies[:, 1:].to(dtype)

            final_tensor = torch.cat([base_tensor, dummies], dim=1)
            analysis_data_map[key] = final_tensor
            aggregated_tensor_list.append(final_tensor)

        aggregated_data = (
            torch.cat(aggregated_tensor_list, dim=0)
            if aggregated_tensor_list
            else torch.empty(0, 11, dtype=dtype)
        )
        return analysis_data_map, aggregated_data


class exact_location_filter:

    def __init__(self):
        pass

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
        격자 좌표와 실측 좌표의 편차가 임계값 이상인 행을 제거.
        하루 내 모든 시간대에서 조건을 만족해야 유효(AND 조건).
        """
        LAT_COL, LON_COL = 0, 1
        max_days = min(
            len(hourly_maps_grid), len(hourly_maps_exact),
            len(aggregated_tensors_grid), len(aggregated_tensors_exact)
        )

        filtered_hourly_maps = {}
        filtered_aggregated_tensors = {}

        print(f"--- Starting Deviation Filter (Days 0-{max_days-1}) ---")
        print(f"Thresholds: Lat <= {lat_threshold}, Lon <= {lon_threshold}")

        for day_idx in range(max_days):
            grid_hourly_day  = hourly_maps_grid[day_idx]
            exact_hourly_day = hourly_maps_exact[day_idx]
            data_agg_grid    = aggregated_tensors_grid[day_idx]

            time_keys = sorted(grid_hourly_day.keys())
            if not time_keys:
                continue

            n_rows = grid_hourly_day[time_keys[0]].shape[0]
            master_mask = torch.ones(n_rows, dtype=torch.bool)

            for t_key in time_keys:
                lat_diff = torch.abs(grid_hourly_day[t_key][:, LAT_COL] - exact_hourly_day[t_key][:, LAT_COL])
                lon_diff = torch.abs(grid_hourly_day[t_key][:, LON_COL] - exact_hourly_day[t_key][:, LON_COL])
                master_mask = master_mask & (lat_diff <= lat_threshold) & (lon_diff <= lon_threshold)

            n_final = master_mask.sum().item()
            pct = (n_final / n_rows * 100) if n_rows > 0 else 0.0

            filtered_hourly_maps[day_idx] = {
                t_key: grid_hourly_day[t_key][master_mask] for t_key in time_keys
            }
            print(f"Day {day_idx} (Hourly): {n_rows} -> {n_final} rows kept ({pct:.2f}%)")

            expected_agg = n_rows * len(time_keys)
            n_rows_agg   = aggregated_tensors_exact[day_idx].shape[0]
            if n_rows_agg == expected_agg and n_rows_agg > 0:
                expanded_mask = master_mask.repeat(len(time_keys))
                filtered_aggregated_tensors[day_idx] = data_agg_grid[expanded_mask]
                print(f"Day {day_idx} (Aggregated): {n_rows_agg} -> {filtered_aggregated_tensors[day_idx].shape[0]} rows kept")
            else:
                print(f"Warning Day {day_idx}: row count mismatch ({n_rows_agg} vs {expected_agg}). Skipping.")

            print("-" * 30)

        print("Filtering Complete.")
        return filtered_hourly_maps, filtered_aggregated_tensors

    @staticmethod
    def get_spatial_ordering(
        coarse_dicts,
        mm_cond_number: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        from GEMS_TCO import orderings as _orderings
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            return np.array([]), np.array([])

        data = coarse_dicts[key_idx[0]]
        coords = np.stack((data[:, 0], data[:, 1]), axis=-1)
        ord_mm = _orderings.maxmin_cpp(coords)
        coords_reordered = np.stack(
            (data[ord_mm][:, 0], data[ord_mm][:, 1]), axis=-1
        )
        nns_map = _orderings.find_nns_l2(locs=coords_reordered, max_nn=mm_cond_number)
        return ord_mm, nns_map
