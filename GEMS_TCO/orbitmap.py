# work environment: jl2815
import pandas as pd
import numpy as np
from collections import defaultdict

from GEMS_TCO.smoothspace import space_average

 
class MakeOrbitdata(space_average):
    def __init__(self, df, lat_resolution, lon_resolution, lat_s,lat_e, lon_s,lon_e ):
        super().__init__(df, lat_resolution, lon_resolution, lat_s, lat_e, lon_s, lon_e)

    def makeorbitmap(self):
        orbit_map = {}  
        self.df['Orbit'] = self.df['Time'].str[0:16]
        orbits = self.df['Orbit'].unique()

        p=0
        for orbit in orbits:
            p+=1
            hour = (p % 8) if (p % 8) != 0 else 8
            orbit_key = f'y{orbit[2:4]}m{int(orbit[5:7]):02d}day{ int(orbit[8:10]):02d}_{hour}'
            orbit_map[orbit_key] = self.df.loc[self.df['Orbit'] == orbit]
        return orbit_map
    
    def makeorbitmap_jan(self):
        orbit_map = {}  
        self.df['Orbit'] = self.df['Time'].str[0:16]
        orbits = self.df['Orbit'].unique()

        p=0
        for orbit in orbits:
            p+=1
            hour = (p % 6) if (p % 6) != 0 else 6
            orbit_key = f'y{orbit[2:4]}m{int(orbit[5:7]):02d}day{ int(orbit[8:10]):02d}_{hour}'
            orbit_map[orbit_key] = self.df.loc[self.df['Orbit'] == orbit]
        return orbit_map
    
        
    def make_sparsemap(self, orbit_map, sparsity):
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
            
            sparse_map[key_list[i]]  = cur
        return sparse_map
    # 2:7432 4:1858, 5:1190, 6:826, 7:607, 8:465
    def coarse_fun(self,df:pd.DataFrame, rho:float)->pd.DataFrame:  # rho has to be integer
        df_copy = df.copy()
        df_copy = df_copy.sort_values(by='Latitude', ascending=False)
        df_copy = df_copy.iloc[::rho,:]
        df_copy = df_copy.sort_values(by='Longitude', ascending=False)
        df_copy = df_copy.iloc[::rho,:]
        mask = df.index.isin(df_copy.index)
        return df.loc[mask,:]
    
    def make_coarsemap(self,orbit_map, rho):
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
        


'''
class difference_data:
    def __init__(self,df: pd.DataFrame):
        self.df = df
    
    def diff_day_to_day(self):

'''





