# work environment: jl2815
import pandas as pd
from collections import defaultdict


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


def makeorbitoutside(df,orbit_map):
    days_list = df['Orbit'].str[8:10].unique()
    year = int(df['Orbit'][0][2:4])
    month = int(df['Orbit'][0][5:7])
    

    for day in days_list:
        for i in range(1,9):
            orbit_key = f'y{year}m{month:02d}day{int(day):02d}_{i:01d}'
            globals()[orbit_key] =  orbit_map[orbit_key]