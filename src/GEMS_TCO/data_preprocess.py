# work environment: jl2815
import pandas as pd
import pickle
import sys
import os
import numpy as np
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from collections import defaultdict
import sklearn
from sklearn.neighbors import BallTree
import xarray as xr # for netCDF4 
from netCDF4 import Dataset

# from scipy.optimize import minimize
# from scipy.spatial.distance import cdist  # For space and time distance
# from scipy.spatial import distance  # Find closest spatial point

from typing import Callable, Union, Tuple
from pathlib import Path

from GEMS_TCO import configuration as config

class GemsORITocsvHour:          
    def __init__(self, lat_s,lat_e,lon_s,lon_e):
        self.lat_s = lat_s 
        self.lat_e = lat_e  
        self.lon_s = lon_s
        self.lon_e = lon_e                         
  
    def extract_data(self,file_path):
        location = xr.open_dataset(file_path, group='Geolocation Fields')
        Z = xr.open_dataset(file_path, group='Data Fields')
        
        location_variables = ['Latitude', 'Longitude', 'Time']
        tmp1 = location[location_variables]

        location_df = tmp1.to_dataframe().reset_index() # Convert xarray.Dataset to pandas DataFrame
        location_df = location_df[location_variables]   # remove spatial (2048), image (695) indices

        Z_variables = ['ColumnAmountO3','FinalAlgorithmFlags']
        tmp2 = Z[Z_variables]

        Z_df = tmp2.to_dataframe().reset_index()      
        Z_df = Z_df[Z_variables]

        mydata = pd.concat([location_df, Z_df], axis=1)      # both rows are 2048*695
  
        # Close the NetCDF file
        location.close()
        Z.close()
        return mydata
    
    def dropna(self, file_path):
        mydata = self.extract_data(file_path)
        mydata = mydata.dropna(subset=['Latitude', 'Longitude','Time','ColumnAmountO3','FinalAlgorithmFlags'])
        return mydata

    def ORItocsv(self, file_path):

        df = self.dropna(file_path)  
        truncated_df = df[ (df['Latitude']<= self.lat_e) & (df['Latitude']>= self.lat_s) & (df['Longitude']>= self.lon_s) & (df['Longitude']<= self.lon_e) ]
        
        # Cut off missing values
        truncated_df= truncated_df[truncated_df.iloc[:,3]<1000]    

        truncated_df['Time'] = np.mean(truncated_df.iloc[:,2])

        # Convert 'Time' column to datetime type
        # print(df2['Time'])

        truncated_df['Time'] = pd.to_datetime(truncated_df['Time'], unit='h')
        truncated_df['Time'] = truncated_df['Time'].dt.floor('min')  
        
        return truncated_df

class file_path_list:
    def __init__(self, year:int, month:int, computer_path:str):
        self.year = year
        self.month = month
        self.computer_path = computer_path

    def file_names_july24(self):
        if self.month == 2:
            self.day_str = "0128"  # Handle February specifically
        else:
            self.day_str = "0131" if (self.month in [1, 3, 5, 7, 8, 10, 12]) else "0130"

        last_day_range = int(self.day_str[2:])+1
        base_directory = f'{self.year}{self.month:02d}{self.day_str}/'
        file_prefixes = []
        for i in range(1,last_day_range):
            file_prefixes.append(f'{self.year}{self.month:02d}{i:02d}_')
        
        file_paths_list = [f"{self.computer_path}{base_directory}{prefix}{hour:02d}45.nc" for prefix in file_prefixes for hour in range(0, 8)] # 6 for january 8 for else
        return file_paths_list
    
class MonthAggregatedCSV(GemsORITocsvHour):
    def __init__(self, lat_start, lat_end, lon_start, lon_end):
        super().__init__(lat_start, lat_end, lon_start, lon_end)

    def aggregate_july24tocsv(self, file_paths_list):
        aggregated_data = []
        for i, filepath in enumerate(file_paths_list):
            try:
                # Attempt to transform netCDF file into csv for hourly data
                cur_data = self.ORItocsv(filepath)
                aggregated_data.append(cur_data)

            except FileNotFoundError:
                print(f"Warning: File not found - {filepath}. Skipping this file.")
                continue

        # Concatenate all DataFrames at once (more efficient than repeated concat)
        aggregated_df =  pd.concat(aggregated_data, ignore_index=True) if aggregated_data else pd.DataFrame()
        aggregated_df['Hours_elapsed'] = aggregated_df['Time'].astype('int64') // 10**9/3600
        
        acceptable_flags = [0, 2, 4, 128]
        filtered_df = aggregated_df[aggregated_df['FinalAlgorithmFlags'].isin(acceptable_flags)]

        # frequency_table3= gqdata['FinalAlgorithmFlags'].value_counts()
        # print(frequency_table3)
        return filtered_df
    
    def save(self, GoodqualityData, year,month, computer_path):
        # computer_path = config.mac.data_path
        if month == 2:
            day_str = "0128"  # Handle February specifically
        else:
            day_str = "0131" if (month in [1, 3, 5, 7, 8, 10, 12]) else "0130"

        output_file = f'data_{int(str(year)[2:4])}_{month:02d}_{day_str}_N{str(self.lat_s)+str(self.lat_e)}_E{str(self.lon_s)+str(self.lon_e)}.csv' 
        output_csv_path = Path(computer_path)/output_file
        
        # csv_file_path = os.path.join(r"C:\\Users\\joonw\tco\\data_engineering", tmp_path)
        GoodqualityData.to_csv(output_csv_path, index=False)

class MonthAggregatedHashmap(MonthAggregatedCSV):
    def __init__(self, lat_start, lat_end, lon_start, lon_end, years:list=[2024], months:list =list( range(7,8))):
        super().__init__(lat_start, lat_end, lon_start, lon_end)
        self.years = years  
        self.months = months 
    
    def aggregate_july24topickle(self, csvfilepath): # ex) config.mac_data_load_path
        for year in self.years:
            for month in self.months:  
                try:
                    # Construct filenames dynamically
                    month_str = f"{month:02d}"  # Ensure month is zero-padded
                    if month == 2 and year==2023:
                        day_str = "0128"  # Handle February specifically
                    elif month ==2 and year==2024:
                        day_str = "0129"
                    else:
                        day_str = "0131" if (month in [1, 3, 5, 7, 8, 10, 12]) else "0130"
            
                    input_filename = f"data_{year}/data_{str(year)[2:]}_{month_str}_{day_str}_N{self.lat_s}{self.lat_e}_E{self.lon_s}{self.lon_e}.csv"
                    
                    input_filepath = Path(csvfilepath) / input_filename
                    
                    # Read data
                    print(f"Reading file: {input_filepath}")
                    df = pd.read_csv(input_filepath)

                    # Process data
                    instance = center_matching_hour(df, self.lat_s, self.lat_e, self.lon_s, self.lon_e)
                    orbit_map = instance.group_data_by_orbits()
                    output_path = Path(csvfilepath) / f'pickle_{year}'

                    # Ensure output directory exists
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    # Save pickle
                    output_filename = f"orbit_map{str(year)[2:]}_{month_str}.pkl"
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, 'wb') as pickle_file:
                        pickle.dump(orbit_map, pickle_file)
                    
                    print(f"Successfully processed and saved data for year {str(year)[2:]} month {month_str}.")
                except FileNotFoundError:
                    print(f"Warning: File {input_filename} not found. Skipping.")
                except Exception as e:
                    print(f"Error processing file {input_filename}: {e}")
      

class center_matching_hour():
    """
    Processes orbit data by averaging over specified spatial regions and resolutions.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        lat_s (int): Start latitude for spatial averaging.
        lat_e (int): End latitude for spatial averaging.
        lon_s (int): Start longitude for spatial averaging.
        lon_e (int): End longitude for spatial averaging.
        lat_resolution (Optional[float]): Latitude resolution for spatial bins. Default is None.
        lon_resolution (Optional[float]): Longitude resolution for spatial bins. Default is None.
    """
    def __init__(
        self, 
        df:pd.DataFrame=None, 
        lat_s:float =0,
        lat_e:float =5, 
        lon_s:float =123,
        lon_e:float =133, 
        lat_resolution:float=None, 
        lon_resolution:float =None
    ):
        # Input validation
        if df is not None:
            assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"

        if lat_resolution is not None:
            assert isinstance(lat_resolution, float), "lat_resolution must be a float"
        if lon_resolution is not None:
            assert isinstance(lon_resolution, float), "lon_resolution must be a float"
        
        self.df = df
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.lat_s = lat_s
        self.lat_e = lat_e
        self.lon_s = lon_s
        self.lon_e = lon_e

    def group_data_by_orbits(self):
        """
        Groups data into a dictionary based on unique orbit timestamps.

        Returns:
            dict: A dictionary where keys represent formatted orbit identifiers 
                and values are DataFrames corresponding to each orbit.
        """
        orbit_map = {}  
        self.df['Orbit'] = self.df['Time'].str[0:16]
        orbits = self.df['Orbit'].unique()
        for orbit in orbits:
            orbit_key = f'y{orbit[2:4]}m{int(orbit[5:7]):02d}day{ int(orbit[8:10]):02d}_hm{(orbit[11:16])}'
            orbit_map[orbit_key] = self.df.loc[self.df['Orbit'] == orbit].reset_index(drop=True)
        return orbit_map
    
    def make_center_points(self, step_lat:float=0.044, step_lon:float=0.063) -> pd.DataFrame:
        lat_coords = np.arange( self.lat_e-step_lat- 0.0002, self.lat_s -step_lat, -step_lat)
        lon_coords = np.arange( self.lon_e-step_lon- 0.0002, self.lon_s-step_lon, -step_lon)

        # Apply the shift as in the original code
        # These are the unique lat/lon values for the "center_points" grid
        final_lat_values = lat_coords + step_lat 
        final_lon_values = lon_coords + step_lon 
        
        # Create 2D grid with broadcasting
        decrement = 0.00012
        lat_grid = final_lat_values[:, None] + np.arange(len(final_lon_values)) * decrement  # shape: (228, 152)

        # Flatten row-wise (C order)
        center_lats = lat_grid.flatten()

        # Create matching longitude grid
        center_lons = np.tile(final_lon_values, len(final_lat_values))

        # Now you can build your DataFrame
        center_points_df = pd.DataFrame({'lat': center_lats, 'lon': center_lons})
        return center_points_df
    
    def make_center_points_wo_calibration(self, step_lat:float=0.044, step_lon:float=0.063) -> pd.DataFrame:
        lat_coords = np.arange( self.lat_e-step_lat, self.lat_s -step_lat, -step_lat)
        lon_coords = np.arange( self.lon_e-step_lon, self.lon_s-step_lon, -step_lon)

        # Apply the shift as in the original code
        # These are the unique lat/lon values for the "center_points" grid
        final_lat_values = lat_coords + step_lat 
        final_lon_values = lon_coords + step_lon 
        
        # Create 2D grid with broadcasting
        #decrement = 0.00012
        decrement = 0 
        
        lat_grid = final_lat_values[:, None] + np.arange(len(final_lon_values)) * decrement  # shape: (228, 152)

        # Flatten row-wise (C order)
        center_lats = lat_grid.flatten()

        # Create matching longitude grid
        center_lons = np.tile(final_lon_values, len(final_lat_values))

        # Now you can build your DataFrame
        center_points_df = pd.DataFrame({'lat': center_lats, 'lon': center_lons})
        return center_points_df
    

    '''  
    coarse_by_center   allows duplicates while coarse_by_center_unique doesnt.
    '''

    def coarse_by_center(self, orbit_map: dict, center_points: pd.DataFrame) -> dict:
        assert isinstance(orbit_map, dict), "orbit_map must be a dict"
        assert isinstance(center_points, pd.DataFrame), "center_points must be a pd.DataFrame"

        coarse_map = {}
        key_list = sorted(orbit_map)

        # Convert query points (lat, lon) to NumPy array
        query_points = center_points[['lat', 'lon']].to_numpy()
        query_points_rad = np.radians(query_points)  # Haversine을 위해 라디안 변환

        num_center_points = len(center_points)

        for key in key_list:
            cur_data = orbit_map[key].reset_index(drop=True)
            locs = cur_data[['Latitude', 'Longitude']].to_numpy()

            if locs.shape[0] == 0:
                # 데이터가 없는 경우 어쩔 수 없이 NaN (혹은 0 등 특정 값으로 채워야 한다면 수정)
                coarse_map[key] = pd.DataFrame({
                    'Latitude': center_points['lat'],
                    'Longitude': center_points['lon'],
                    'ColumnAmountO3': [np.nan] * num_center_points, # 필요시 0.0 등으로 변경
                    'Hours_elapsed': [np.nan] * num_center_points,
                    'Time': [pd.NaT] * num_center_points,
                })
                continue

            # Use haversine
            locs_rad = np.radians(locs)
            tree = BallTree(locs_rad, metric='haversine')
            
            # -------------------------------------------------------------
            # [수정 포인트] k=1 대신 k=3 사용 (주변 3개 점 탐색)
            # -------------------------------------------------------------
            k_neighbors = 3
            dist, ind = tree.query(query_points_rad, k=k_neighbors) 
            # dist shape: (num_points, 3), ind shape: (num_points, 3)

            # -------------------------------------------------------------
            # IDW (Inverse Distance Weighting) 계산
            # 거리가 가까울수록 가중치를 크게 둠 (Weight = 1 / distance^2)
            # -------------------------------------------------------------
            epsilon = 1e-10  # 0으로 나누기 방지
            weights = 1.0 / (dist + epsilon)**2
            
            # 가중치 정규화 (합이 1이 되도록)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            norm_weights = weights / weights_sum

            # 이웃한 점들의 O3 값 가져오기
            # ind 배열을 이용해 O3 값을 (num_points, 3) 형태로 추출
            neighbor_o3 = cur_data['ColumnAmountO3'].values[ind]
            
            # 가중 평균 계산 (Row-wise sum)
            interpolated_o3 = np.sum(norm_weights * neighbor_o3, axis=1)
            
            # -------------------------------------------------------------
            # 메타 데이터 (Time, Hours_elapsed)
            # -------------------------------------------------------------
            hours_elapsed_val = cur_data['Hours_elapsed'].iloc[0] if not cur_data.empty else np.nan
            time_val = cur_data['Time'].iloc[0] if not cur_data.empty else pd.NaT

            # Source 좌표는 '가장 가까운 점(k=1)' 하나만 남기거나, 굳이 필요 없다면 생략 가능
            # 여기서는 가장 가까운 점의 좌표를 메타데이터로 남김
            nearest_indices = ind[:, 0] 
            source_lat = cur_data.loc[nearest_indices, 'Latitude'].values
            source_lon = cur_data.loc[nearest_indices, 'Longitude'].values

            coarse_map[key] = pd.DataFrame({
                'Latitude': center_points['lat'].values,
                'Longitude': center_points['lon'].values,
                'ColumnAmountO3': interpolated_o3,  # IDW로 부드럽게 채워진 값
                'Hours_elapsed': [hours_elapsed_val] * num_center_points,
                'Time': [time_val] * num_center_points,
                'Source_Latitude': source_lat, # 참고용 (가장 가까운 점)
                'Source_Longitude': source_lon
            })
            
        return coarse_map
    






