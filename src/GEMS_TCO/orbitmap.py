# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.neighbors import BallTree

# from scipy.optimize import minimize
# from scipy.spatial.distance import cdist  # For space and time distance
# from scipy.spatial import distance  # Find closest spatial point

from typing import Callable, Union, Tuple

 
class MakeOrbitdata():
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
        lat_s:int =5,
        lat_e:int =10, 
        lon_s:int =110,
        lon_e:int =120, 
        lat_resolution:float=None, 
        lon_resolution:float =None
    ):
        # Input validation
        if df is not None:
            assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(lat_s, int), "lat_s must be int"
        assert isinstance(lat_e, int), "lat_e must be int"
        assert isinstance(lon_s, int), "lon_s must be int"
        assert isinstance(lon_e, int), "lon_e must be int"
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




    






