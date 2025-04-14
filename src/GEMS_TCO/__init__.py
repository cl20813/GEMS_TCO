
import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch


gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)
from GEMS_TCO import orderings as _orderings

### 


    # print(f"lat_lon_resolution: {lat_lon_resolution}, mm_cond_number: {mm_cond_number}, params: {params}, v: {v}, lr: {lr}, epochs: {epochs}, nheads: {nheads}")

###

class load_data_local_computer:
    def __init__(self):
        pass
    
    def load_mm20k_data_bymonthyear(self, lat_lon_resolution= [10,10], mm_cond_number=10, years_=['2024'], months_=[7,8]):

        ## Load the one dictionary to set spaital coordinates
        # filepath = "C:/Users/joonw/TCO/GEMS_data/data_2023/sparse_cen_map23_01.pkl"
        filepath = "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/coarse_cen_map23_01.pkl"
        with open(filepath, 'rb') as pickle_file:
            coarse_dict_24_1 = pickle.load(pickle_file)

        sample_df = coarse_dict_24_1['y23m01day01_hm02:12']
        sample_key = coarse_dict_24_1.get('y23m01day01_hm02:12')
        if sample_key is None:
            print("Key 'y23m01day01_hm02:12' not found in the dictionary.")

        rho_lat = lat_lon_resolution[0]          
        rho_lon = lat_lon_resolution[1]
        lat_n = sample_df['Latitude'].unique()[::rho_lat]
        lon_n = sample_df['Longitude'].unique()[::rho_lon]

        # Set spatial coordinates for each dataset
        coarse_dicts = {}
        years = years_
        for year in years:
            for month in range(months_[0], months_[1]):  # Iterate over all months
                # filepath = f"C:/Users/joonw/TCO/GEMS_data/data_{year}/sparse_cen_map{year[2:]}_{month:02d}.pkl"
                filepath = f"/Users/joonwonlee/Documents/GEMS_DATA/pickle_{year}/coarse_cen_map{year[2:]}_{month:02d}.pkl"
                with open(filepath, 'rb') as pickle_file:
                    loaded_map = pickle.load(pickle_file)
                    for key in loaded_map:
                        tmp_df = loaded_map[key]
                        coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                        coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)

        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")

        # extract first hour data because all data shares the same spatial grid
        data_for_coord = coarse_dicts[key_idx[0]]
        x1 = data_for_coord['Longitude'].values
        y1 = data_for_coord['Latitude'].values 
        coords1 = np.stack((x1, y1), axis=-1)

        # instance = orbitmap.MakeOrbitdata(data_for_coord, lat_s=5, lat_e=10, lon_s=110, lon_e=120)
        # s_dist = cdist(coords1, coords1, 'euclidean')
        # ord_mm, _ = instance.maxmin_naive(s_dist, 0)

        ord_mm = _orderings.maxmin_cpp(coords1)
        data_for_coord = data_for_coord.iloc[ord_mm].reset_index(drop=True)
        coords1_reordered = np.stack((data_for_coord['Longitude'].values, data_for_coord['Latitude'].values), axis=-1)
        # nns_map = instance.find_nns_naive(locs=coords1_reordered, dist_fun='euclidean', max_nn=mm_cond_number)
        nns_map=_orderings.find_nns_l2(locs= coords1_reordered  ,max_nn = mm_cond_number)

        return coarse_dicts, ord_mm, nns_map

    def load_working_data_byday(self, coarse_dicts,  ord_mm, nns_map, idx_for_datamap=[0,8]):
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")
        analysis_data_map = {}
        for i in range(idx_for_datamap[0],idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed']-477700)

            tmp = tmp.iloc[ord_mm, :4].to_numpy()  # reorder the data
            # tmp = torch.from_numpy(tmp).float()  # Convert NumPy to Tensor
            tmp = torch.from_numpy(tmp).double()
            analysis_data_map[key_idx[i]] = tmp

        aggregated_data = pd.DataFrame()
        for i in range(idx_for_datamap[0],idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed']-477700)
            tmp = tmp.iloc[ord_mm].reset_index(drop=True)  
            aggregated_data = pd.concat((aggregated_data, tmp), axis=0)

        aggregated_data = aggregated_data.iloc[:, :4].to_numpy()
        #aggregated_data = torch.from_numpy(aggregated_data).float() 
        aggregated_data = torch.from_numpy(aggregated_data).double()

        return analysis_data_map, aggregated_data
    

class load_data_amarel:
    def __init__(self):
        pass
    
    def load_mm20k_data_bymonthyear(self, lat_lon_resolution= [10,10], mm_cond_number=10, years_=['2024'], months_=[7,8]):

        ## Load the one dictionary to set spaital coordinates
        filepath = "/home/jl2815/tco/data/pickle_data/pickle_2023/coarse_cen_map23_01.pkl"

        with open(filepath, 'rb') as pickle_file:
            coarse_dict_24_1 = pickle.load(pickle_file)

        sample_df = coarse_dict_24_1['y23m01day01_hm02:12']
        sample_key = coarse_dict_24_1.get('y23m01day01_hm02:12')
        if sample_key is None:
            print("Key 'y23m01day01_hm02:12' not found in the dictionary.")

        rho_lat = lat_lon_resolution[0]          
        rho_lon = lat_lon_resolution[1]
        lat_n = sample_df['Latitude'].unique()[::rho_lat]
        lon_n = sample_df['Longitude'].unique()[::rho_lon]

        # Set spatial coordinates for each dataset
        coarse_dicts = {}
        years = years_
        for year in years:
            for month in range(months_[0], months_[1]):  # Iterate over all months
                filepath = f"/home/jl2815/tco/data/pickle_data/pickle_{year}/coarse_cen_map{year[2:]}_{month:02d}.pkl"
                with open(filepath, 'rb') as pickle_file:
                    loaded_map = pickle.load(pickle_file)
                    for key in loaded_map:
                        tmp_df = loaded_map[key]
                        coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                        coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)

        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")

        # extract first hour data because all data shares the same spatial grid
        data_for_coord = coarse_dicts[key_idx[0]]
        x1 = data_for_coord['Longitude'].values
        y1 = data_for_coord['Latitude'].values 
        coords1 = np.stack((x1, y1), axis=-1)

        # instance = orbitmap.MakeOrbitdata(data_for_coord, lat_s=5, lat_e=10, lon_s=110, lon_e=120)
        # s_dist = cdist(coords1, coords1, 'euclidean')
        # ord_mm, _ = instance.maxmin_naive(s_dist, 0)

        ord_mm = _orderings.maxmin_cpp(coords1)
        data_for_coord = data_for_coord.iloc[ord_mm].reset_index(drop=True)
        coords1_reordered = np.stack((data_for_coord['Longitude'].values, data_for_coord['Latitude'].values), axis=-1)
        # nns_map = instance.find_nns_naive(locs=coords1_reordered, dist_fun='euclidean', max_nn=mm_cond_number)
        nns_map=_orderings.find_nns_l2(locs= coords1_reordered  ,max_nn = mm_cond_number)

        return coarse_dicts, ord_mm, nns_map

    def load_working_data_byday(self, coarse_dicts, ord_mm, nns_map, idx_for_datamap=[0,8]):
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")
        analysis_data_map = {}
        for i in range(idx_for_datamap[0],idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed']-477700)

            tmp = tmp.iloc[ord_mm, :4].to_numpy()  # reorder the data
            #tmp = torch.from_numpy(tmp).float()  # Convert NumPy to Tensor
            tmp = torch.from_numpy(tmp).double()  # Convert NumPy to Tensor
            analysis_data_map[key_idx[i]] = tmp

        aggregated_data = pd.DataFrame()
        for i in range(idx_for_datamap[0],idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed']-477700)
            tmp = tmp.iloc[ord_mm].reset_index(drop=True)  
            aggregated_data = pd.concat((aggregated_data, tmp), axis=0)

        aggregated_data = aggregated_data.iloc[:, :4].to_numpy()
        #aggregated_data = torch.from_numpy(aggregated_data).float() 
        aggregated_data = torch.from_numpy(aggregated_data).double()
        return analysis_data_map, aggregated_data


if __name__ == "__main__":
    app()