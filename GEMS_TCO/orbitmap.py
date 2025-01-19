# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict

import sklearn
from sklearn.neighbors import BallTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist  # For space and time distance
from scipy.spatial import distance  # Find closest spatial point

from typing import Callable, Union, Tuple
from GEMS_TCO.smoothspace import space_average
 
class MakeOrbitdata(space_average):
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
        df:pd.DataFrame, 
        lat_s:int,
        lat_e:int, 
        lon_s:int,
        lon_e:int, 
        lat_resolution:float=None, 
        lon_resolution:float =None
    ):
        # Input validation
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(lat_s, int), "lat_s must be int"
        assert isinstance(lat_e, int), "lat_e must be int"
        assert isinstance(lon_s, int), "lon_s must be int"
        assert isinstance(lon_e, int), "lon_e must be int"
        if lat_resolution is not None:
            assert isinstance(lat_resolution, float), "lat_resolution must be a float"
        if lon_resolution is not None:
            assert isinstance(lon_resolution, float), "lon_resolution must be a float"
        
        super().__init__(df, lat_resolution, lon_resolution, lat_s, lat_e, lon_s, lon_e)

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
    
    
    # make_coarse map by aggregating data
    def make_sparsemap(self, orbit_map, sparsity):
        assert isinstance(orbit_map, dict), "orbit_map must be a dict"
        sparse_map = {}
        key_list = sorted(orbit_map)
        for i in range(len(key_list)):
            cur = orbit_map[key_list[i]]   
            instance =  space_average(cur, sparsity,sparsity,self.lat_s,self.lat_e, self.lon_s,self.lon_e) # 0.2 defines sparsity
            cur = instance.space_avg()

            mask = (cur['ColumnAmountO3'] < 200) | (cur['ColumnAmountO3'] > 500)
            df = cur[~mask]
            mean_value = np.mean(df['ColumnAmountO3'])
            cur.loc[mask, 'ColumnAmountO3'] = mean_value

            sparse_map[key_list[i]]  = cur.reset_index(drop=True)
        return sparse_map

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

    def sparse_by_center(self, orbit_map:dict, center_points:pd.DataFrame) -> dict:
        assert isinstance(orbit_map, dict), "orbit_map must be a dict"
        assert isinstance(center_points, pd.DataFrame), "center_points must be a pd.DataFrame"

        sparse_map = {}
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

            sparse_map[key] = pd.DataFrame( 
                {
                    'Latitude':center_points.loc[:,'lat'], 
                    'Longitude':center_points.loc[:,'lon'], 
                    'ColumnAmountO3':res_series,  
                    'Hours_elapsed': [cur_data['Hours_elapsed'][0]]* len(center_points), 
                    'Time' : [cur_data['Time'][0]]* len(center_points) 
                }
            )

        return sparse_map


    # 2:7432 4:1858, 5:1190, 6:826, 7:607, 8:465
    def coarse_fun(self,df:pd.DataFrame, rho:int)->pd.DataFrame:  # rho has to be integer
        assert isinstance(df, pd.DataFrame), "df must be a pd.DataFrame"
        assert isinstance(rho, int), "rho must be an integer"

        # Sort by latitude, take every rho-th row
        df_sorted_lat = df.sort_values(by='Latitude', ascending=False).iloc[::rho, :]
        # Sort by longitude, take every rho-th row
        df_sorted_lon = df_sorted_lat.sort_values(by='Longitude', ascending=False).iloc[::rho, :]

        mask = df.index.isin(df_sorted_lon.index)
        result = df.loc[mask, :].reset_index(drop=True)
        return result
    
    def make_coarsemap(self,orbit_map:dict, rho:int)->dict:
        assert isinstance(orbit_map, dict), "orbit_map must be a dict"
        assert isinstance(rho, int), "rho must be an integer"
        coarse_map = {}
        key_list = sorted(orbit_map)
        for key in key_list:

            cur = orbit_map[key]
            mask = (cur['ColumnAmountO3'] < 200) | (cur['ColumnAmountO3'] > 500)
            df = cur[~mask]
            mean_value = np.mean(df['ColumnAmountO3'])
            cur.loc[mask, 'ColumnAmountO3'] = mean_value

            instance = self.coarse_fun(cur,rho)

            coarse_map[key] = instance 
        return coarse_map
        
    
    def maxmin_naive(self,dist: np.ndarray, first: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs min-max ordering

        The implementation is naive and will not perform well for large inputs.

        Parameters
        ----------
        dist
            distance matrix
        first
            Index of the observation that should be sorted first

        Returns
        -------
        np.ndarray
            The minmax ordering
        np.ndarray
            Array with the distances to the location preceding in ordering
        """

        n = dist.shape[0]
        ord = np.zeros(n, dtype=np.int64)
        ord[0] = first
        dists = np.zeros(n)
        dists[0] = np.nan
        idx = np.arange(n)


        for i in range(1, n):
            # find min dist for each not selected loccation
            mask = ~np.isin(idx, ord[:i])
            min_d = np.min(dist[mask, :][:, ord[:i]], axis=1)

            # find max idx among those
            idx_max = np.argmax(min_d)

            # record dist
            dists[i] = min_d[idx_max]

            # adjust idx for the prevous removed rows
            idx_max = idx[mask][idx_max]
            ord[i] = idx_max
        return ord, dists

    
    def find_nns_naive(
        self, locs: np.ndarray, dist_fun: Union[Callable, str] = "euclidean", max_nn: int = 10, **kwargs
    ) -> np.ndarray:
        """
        Finds the max_nn nearest neighbors preceding in the ordering.

        The method is naivly implemented and will not perform well for large inputs.

        Parameters
        ----------
        locs
            an n x m array of ordered locations
        dist_fun
            a distance function
        max_nn
            number of nearest neighbours
        kwargs
            supplied dist_func

        Returns
        -------
        np.ndarray
            Returns an n x max_nn array holding the indices of the nearest neighbors
            preceding in the ordering where -1 indicates missing neighbors.
        """
        n = locs.shape[0]
        nns = np.zeros((n, max_nn), dtype=np.int64) - 1
        for i in range(1, n):
            nn = sklearn.neighbors.BallTree(locs[:i], metric=dist_fun, **kwargs)   # dist_fun= 'euclidean'
            k = min(i-1, max_nn)
            nn_res = nn.query(locs[[i], :], k=k, return_distance=False)
            nns[i, :k] = nn_res
        return nns


class databyday_24July:
    def __init__(self):
        pass

    def cutlatitude(self, map, lat_idx):
        keys = sorted(map)
        for key in keys:
            df = map[key]
            cutted_df = df[(df['Latitude']>=lat_idx) & (df['Latitude']< (lat_idx+1))]
            map[key] = cutted_df
        
        return map
    
    def groupbyday(self,map):
        keys = sorted(map)
        grp = defaultdict(list)
        keys_idx = list(range(240))
        grouped_data = [ keys_idx[i:i + 8] for i in range(0, len(keys_idx), 8)]
        for i in range(len(grouped_data)):
            data = None
            for idx in grouped_data[i]:
                if data is not None:
                    data = pd.concat([data, map[keys[idx]]], axis=0)
                else:
                    data = map[keys[idx]]
            grp[i] = data.reset_index(drop=True)
        return grp
    
    def maxmin_ordering(self, coarse_map:dict, mm_cond_number:int):
        assert isinstance(coarse_map, dict), "coarse_map must be a dict"
        assert isinstance(mm_cond_number, int), "mm_cond_number must be an integer"
        keys = sorted(coarse_map)
        sample_data = coarse_map[keys[0]]

        # Extract values
        x1 = sample_data['Longitude'].values
        y1 = sample_data['Latitude'].values 
        coords1 = np.stack((x1, y1), axis=-1)
        # Calculate spatial distances using cdist
        s_dist = cdist(coords1, coords1, 'euclidean')
        # initiate 
        instance = MakeOrbitdata(sample_data, 5,10,110,120) # input is not really here just to use maxmin_naive and finds_nns_naive
        ord_mm, _ = instance.maxmin_naive(s_dist, 0)
        # Construct nearest neighboring set
        
        # Reorder the DataFrame
        sample_data = sample_data.iloc[ord_mm].reset_index(drop=True)  
        coords1_reordered = np.stack((sample_data['Longitude'].values, sample_data['Latitude'].values), axis=-1)
        nns_map = instance.find_nns_naive(locs=coords1_reordered, dist_fun='euclidean', max_nn=mm_cond_number)
        baseset_from_maxmin = ord_mm[:mm_cond_number]

        for key in keys:
            coarse_map[key] = coarse_map[key].iloc[ord_mm].reset_index(drop=True)
        return coarse_map, baseset_from_maxmin, nns_map

    def process(self, coarse_map:dict, lat_idx:int, mm_cond_number:int):
        assert isinstance(coarse_map, dict), "coarse_map must be a dict"
        assert isinstance(lat_idx, int), "lat_idx must be an integer"
        assert isinstance(mm_cond_number, int), "mm_cond_number must be an integer"
        lat_cutted_map = self.cutlatitude(coarse_map, lat_idx)
        lat_cutted_map, baseset_from_maxmin, nns_map = self.maxmin_ordering(lat_cutted_map,mm_cond_number)
        grp = self.groupbyday(lat_cutted_map)
        return grp, baseset_from_maxmin, nns_map 

    def process_nolat(self, coarse_map, mm_cond_number):
        assert isinstance(coarse_map, dict), "coarse_map must be a dict"
        assert isinstance(mm_cond_number, int), "mm_cond_number must be an integer"

        coarse_map, baseset_from_maxmin, nns_map = self.maxmin_ordering(coarse_map, mm_cond_number)
        grp = self.groupbyday(coarse_map)
        return grp, baseset_from_maxmin, nns_map 
    
    def process_forpurespace(self, coarse_map, mm_cond_number):
        assert isinstance(coarse_map, dict), "coarse_map must be a dict"
        assert isinstance(mm_cond_number, int), "mm_cond_number must be an integer"

        coarse_map, _, nns_map = self.maxmin_ordering(coarse_map,mm_cond_number)
        return coarse_map, nns_map







