# work environment: jl2815
import pandas as pd
from collections import defaultdict

from GEMS_TCO import smoothspace

class MakeOrbitdata:
    def __init__(self,df: pd.DataFrame):
        self.df = df

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
            instance =  smoothspace.space_avgerage(cur, sparsity,sparsity, lat_start=30,lon_start=100) # 0.2 defines sparsity

            sparse_map[key_list[i]] = instance.space_avg()
        return sparse_map
        


'''
class difference_data:
    def __init__(self,df: pd.DataFrame):
        self.df = df
    
    def diff_day_to_day(self):

'''





