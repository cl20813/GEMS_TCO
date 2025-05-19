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
        lat_s:float =5,
        lat_e:float =10, 
        lon_s:float =110,
        lon_e:float =120, 
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

        Input:
            Time: String. When saving data into csv file, time object file becomes string
            beause csv file is a plain text file. 
        
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
    
    def make_center_points(self, step:float=0.05) -> pd.DataFrame:
        assert isinstance(step, float), "step must be a float"
        # Create grid coordinates
        lat_coords = np.arange(self.lat_s, self.lat_e, step)
        lon_coords = np.arange(self.lon_s, self.lon_e, step)
        center_points = []
        for lat in lat_coords:
            for lon in lon_coords:
                center_lat = lat + step / 2
                center_lon = lon + step / 2
                center_points.append([center_lat, center_lon])

        center_points= pd.DataFrame(center_points,columns=['lat','lon'])
        return center_points

    def coarse_by_center(self, orbit_map:dict, center_points:pd.DataFrame) -> dict:
        assert isinstance(orbit_map, dict), "orbit_map must be a dict"
        assert isinstance(center_points, pd.DataFrame), "center_points must be a pd.DataFrame"

        coarse_map = {}
        key_list = sorted(orbit_map)

        res = [0]* len(center_points) 

        for key in key_list:
            cur_data = orbit_map[key].reset_index(drop=True)
            locs = cur_data[['Latitude','Longitude']]
            locs = np.array(locs)
            tree = BallTree(locs, metric='euclidean')
            for i in range(len(center_points)):
                target = center_points.iloc[i,:].to_numpy().reshape(1,-1)
                dist, ind = tree.query(target, k=1)
                res[i] = cur_data.loc[ind[0][0], 'ColumnAmountO3']
            
            res_series = pd.Series(res)

            coarse_map[key] = pd.DataFrame( 
                {
                    'Latitude':center_points.loc[:,'lat'], 
                    'Longitude':center_points.loc[:,'lon'], 
                    'ColumnAmountO3':res_series,  
                    'Hours_elapsed': [cur_data['Hours_elapsed'][0]]* len(center_points), 
                    'Time' : [cur_data['Time'][0]]* len(center_points) 
                }
            )

        return coarse_map




    






