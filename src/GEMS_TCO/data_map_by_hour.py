# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.neighbors import BallTree
import xarray as xr # for netCDF4 
from netCDF4 import Dataset

# from scipy.optimize import minimize
# from scipy.spatial.distance import cdist  # For space and time distance
# from scipy.spatial import distance  # Find closest spatial point

from typing import Callable, Union, Tuple

class gems_ORI_tocsv:          
    def __init__(self, file_path,lat_s,lat_e,lon_s,lon_e):
        self.file_path = file_path       
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

        mydata = pd.concat([location_df, Z_df], axis=1) # both rows are 2048*695
  
        # Close the NetCDF file
        location.close()
        Z.close()
        return mydata
    
    def dropna(self):
        mydata = self.extract_data(self.file_path)
        mydata = mydata.dropna(subset=['Latitude', 'Longitude','Time','ColumnAmountO3','FinalAlgorithmFlags'])
        return mydata

    def result(self):

        df = self.dropna()  
        truncated_df = df[ (df['Latitude']<= self.lat_e) & (df['Latitude']>= self.lat_s) & (df['Longitude']>= self.lon_s) & (df['Longitude']<= self.lon_e) ]
        
        # Cut off missing values
        truncated_df= truncated_df[truncated_df.iloc[:,3]<1000]    

        truncated_df['Time'] = np.mean(truncated_df.iloc[:,2])

        # Convert 'Time' to datetime type
        truncated_df['Time'] = pd.to_datetime(truncated_df['Time'], unit='h')
        truncated_df['Time'] = truncated_df['Time'].dt.floor('min')  
        return truncated_df
    
 
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




    






