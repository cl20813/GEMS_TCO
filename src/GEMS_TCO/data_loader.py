import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch

from pathlib import Path
import json
from json import JSONEncoder
import csv

from typing import Optional, List, Dict, Tuple, Any
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

import GEMS_TCO
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config

    # -----------------------------------------

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
            Loads and filters coarse data from pickle files by month and year.
            
            This method uses a sample file (July 2024) to determine the 
            spatial grid for filtering.
            """
            # Load the dictionary to set spatial coordinates
            # Below is for instrument moving donward calibration

            #filepath = Path(self.datapath) / "pickle_2024/coarse_cen_map24_07.pkl"
            filepath = Path(self.datapath) / f"pickle_2024/coarse_cen_map_without_decrement_latitude{str(2024)[2:]}_{7:02d}.pkl"


            try:
                with open(filepath, 'rb') as pickle_file:
                    coarse_dict_24_1 = pickle.load(pickle_file)
            except FileNotFoundError:
                print(f"Error: Sample file not found at {filepath}")
                return {}
            except Exception as e:
                print(f"Error loading sample file {filepath}: {e}")
                return {}

            keys = list(coarse_dict_24_1.keys())
            if not keys:
                print("Error: Sample pickle file is empty.")
                return {}

            sample_df = coarse_dict_24_1[keys[0]]
            
            rho_lat = lat_lon_resolution[0]          
            rho_lon = lat_lon_resolution[1]

            unique_lats = sample_df['Latitude'].unique()
            sorted_lats_descending = np.sort(unique_lats)[::-1] 
            lat_n = sorted_lats_descending[::rho_lat]
            
            unique_lons = sample_df['Longitude'].unique()
            sorted_lons_descending = np.sort(unique_lons)[::-1] 
            lon_n = sorted_lons_descending[::rho_lon]

            coarse_dicts = {}
            for year in years_:
                for month in months_:  
                    #filepath = Path(self.datapath) / f"pickle_{year}/coarse_cen_map{year[2:]}_{month:02d}.pkl"
                    filepath = Path(self.datapath) / f"pickle_{year}/coarse_cen_map_without_decrement_latitude{str(year)[2:]}_{month:02d}.pkl"
                   
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

    def get_spatial_ordering(
            self, 
            coarse_dicts: Dict[str, pd.DataFrame],
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
            x1 = data_for_coord['Longitude'].values
            y1 = data_for_coord['Latitude'].values 
            coords1 = np.stack((x1, y1), axis=-1)

            # Calculate MaxMin ordering
            ord_mm = _orderings.maxmin_cpp(coords1)
            
            # Reorder coordinates to find nearest neighbors
            data_for_coord_reordered = data_for_coord.iloc[ord_mm].reset_index(drop=True)
            coords1_reordered = np.stack(
                (data_for_coord_reordered['Longitude'].values, data_for_coord_reordered['Latitude'].values), 
                axis=-1
            )
            
            # Calculate nearest neighbors map
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
        ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
            """
            Loads, optionally subsets, and gets MaxMin ordering for data.
            
            Flow:
            1. load_coarse_data_dicts()
            2. (Optional) subset_df_map()
            3. get_spatial_ordering()
            """
            
            # --- Step 1: Load the data ---
            coarse_dicts = self.load_coarse_data_dicts(
                lat_lon_resolution=lat_lon_resolution,
                years_=years_,
                months_=months_
            )
            
            if not coarse_dicts:
                print("Warning: Data loading returned no data. Returning empty.")
                return {}, np.array([]), np.array([])
                
            # --- Step 2: (Optional) Subset the data ---
            if lat_range is not None and lon_range is not None:
                print(f"Subsetting data to lat: {lat_range}, lon: {lon_range}")
                coarse_dicts = self.subset_df_map(
                    coarse_dicts,
                    lat_range=lat_range,
                    lon_range=lon_range
                )
                # Stop if subsetting removed all data
                if not coarse_dicts:
                    print("Warning: Subsetting returned no data. Returning empty.")
                    return {}, np.array([]), np.array([])
            
            # --- Step 3: Get spatial ordering *on the final data* ---
            ord_mm, nns_map = self.get_spatial_ordering(
                coarse_dicts=coarse_dicts, # This is now the correct (potentially subsetted) dict
                mm_cond_number=mm_cond_number
            )
            
            # --- Step 4: Return all results ---
            return coarse_dicts, ord_mm, nns_map
    '''
    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
    lat_lon_resolution=lat_lon_resolution, 
    mm_cond_number=mm_cond_number,
    years_=years, 
    months_=month_range,
    lat_range=[0.0, 5.0],      # <-- Add this
    lon_range=[123.0, 133.0]   # <-- Add this
    )
    
    '''

    def subset_df_map(
            self, 
            df_map: Dict[str, pd.DataFrame],
            lat_range: List[float] = [0.0, 5.0],
            lon_range: List[float] = [123.0, 133.0]
        ) -> Dict[str, pd.DataFrame]:
            """
            Subsets each DataFrame in a dictionary to a specific lat/lon range.
            
            Assumes DataFrames have 'Latitude' and 'Longitude' columns.
            Returns a *new* dictionary with the subsetted DataFrames.
            """
            subsetted_map = {}
            for key, df in df_map.items():
                # Create boolean masks for latitude and longitude
                lat_mask = (df['Latitude'] >= lat_range[0]) & (df['Latitude'] <= lat_range[1])
                lon_mask = (df['Longitude'] >= lon_range[0]) & (df['Longitude'] <= lon_range[1])
                
                # Apply the combined mask and store a copy to avoid SettingWithCopyWarning
                subsetted_map[key] = df[lat_mask & lon_mask].copy()
                
            return subsetted_map
    ''' 
    df_map_subsetted = data_load_instance.subset_df_map(
    df_map, 
    lat_range=[0.0, 5.0], 
    lon_range=[123.0, 133.0]
    )
    '''

    def load_working_data(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        idx_for_datamap: List[int] = [0, 8],
        ord_mm: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load and process working data by day, with optional MaxMin reordering.

        Parameters:
        - coarse_dicts (Dict[str, pd.DataFrame]): Dictionary of processed dataframes.
        - idx_for_datamap (List[int]): Indices for the data map.
        - ord_mm (Optional[np.ndarray]): If provided, applies this ordering to the data.
        - dtype (torch.dtype): The target data type (e.g., torch.float or torch.double).

        Returns:
        - Tuple[Dict[str, torch.Tensor], torch.Tensor]: 
            - analysis_data_map: Dictionary of tensors for analysis.
            - aggregated_data: Aggregated tensor data.
        """
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")
        
        analysis_data_map = {}
        aggregated_df_list = []
        
        # Determine the corresponding numpy dtype
        np_dtype = np.float32 if dtype == torch.float else np.float64
        
        selected_keys = key_idx[idx_for_datamap[0]:idx_for_datamap[1]]
        
        for key in selected_keys:
            tmp = coarse_dicts[key].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700).astype(np_dtype)
            
            # --- Conditionally apply reordering ---
            if ord_mm is not None:
                tmp_processed = tmp.iloc[ord_mm].reset_index(drop=True)
            else:
                tmp_processed = tmp
            
            # Slice to the first 4 columns
            # This is more efficient than the original `load_working_data_byday`
            # which appended the *entire* reordered df to the list.
            tmp_data_df = tmp_processed.iloc[:, :4]

            # 1. Create data for analysis_data_map
            tmp_np = tmp_data_df.to_numpy(dtype=np_dtype)
            analysis_data_map[key] = torch.from_numpy(tmp_np) # .to(dtype) is redundant

            # 2. Store the df for aggregation
            aggregated_df_list.append(tmp_data_df)

        if not aggregated_df_list:
            return analysis_data_map, torch.empty(0, 4, dtype=dtype)

        # Concat once
        aggregated_data_df = pd.concat(aggregated_df_list, axis=0, ignore_index=True)
        aggregated_data_np = aggregated_data_df.to_numpy(dtype=np_dtype)
        
        # Create final tensor
        aggregated_data = torch.from_numpy(aggregated_data_np)
        
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