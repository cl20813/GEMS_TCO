import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from skgstat import Variogram
import numpy as np
from typing import Callable   # nearest neighbor function input type
import sklearn.neighbors  # nearest neighbor

from sklearn.neighbors import BallTree # for space_center function



class space_avg:
    '''
    input
    df: pandas data frame
    lat_resolution: 
    lon_resolution:

    predtermined input:
    lat_min, mat_max : These should align with the spaital domain
    lon_min, lon_max
    '''

    def __init__(self, df, lat_resolution, lon_resolution, lat_start, lon_start):
        self.df = df
   
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.lat_start = lat_start
        self.lon_start = lon_start
        

    def space_avg(self):
        self.latitudes = np.array(self.df['Latitude'])
        self.longitudes =  np.array(self.df['Longitude'])
        self.values =  np.array( self.df['ColumnAmountO3'])

        lat_min, lat_max = self.lat_start, (self.lat_start+5)
        lon_min, lon_max = self.lon_start, (self.lon_start+10)
        # lat_resolution = 0.1
        # lon_resolution = 0.1

        # Create bins for latitude and longitude
        lat_bins = np.linspace(lat_min, lat_max , int((lat_max - lat_min) / self.lat_resolution) + 1)
        lon_bins = np.linspace(lon_min, lon_max , int((lon_max - lon_min) / self.lon_resolution) + 1)

        # lat_bins = np.arange(lat_min, lat_max , lat_resolution)
        # lon_bins = np.arange(lon_min, lon_max , lon_resolution)

        lat_indices = np.digitize(self.latitudes, lat_bins) - 1  # Get bin index for latitudes # np.digitize returns 1 based indice
        lon_indices = np.digitize(self.longitudes, lon_bins) - 1 # this is why we subtract one at the end

        smoothed_grid = np.full( (len(lat_bins) - 1, len(lon_bins) - 1), np.nan) # 

        # Create a 2D array to count the number of points per cell
        counts = np.zeros_like(smoothed_grid)

        # Perform spatial averaging
        for i in range(len(self.values)):
            if 0 <= lat_indices[i] < smoothed_grid.shape[0] and 0 <= lon_indices[i] < smoothed_grid.shape[1]:
                if np.isnan(smoothed_grid[lat_indices[i], lon_indices[i]]):
                    smoothed_grid[lat_indices[i], lon_indices[i]] = self.values[i]
                else:
                    smoothed_grid[lat_indices[i], lon_indices[i]] += self.values[i]
                counts[lat_indices[i], lon_indices[i]] += 1

        # Divide by the counts to get the average in each cell
        smoothed_grid = np.divide(smoothed_grid, counts, where=(counts != 0))

        # Replace NaNs (empty cells) with 0 or other fill values, if necessary
        smoothed_grid = np.nan_to_num(smoothed_grid)


        lat_centers = lat_bins[:-1] + (lat_bins[1] - lat_bins[0]) / 2  # Midpoints of latitude bins
        lon_centers = lon_bins[:-1] + (lon_bins[1] - lon_bins[0]) / 2  # Midpoints of longitude bins

        # Create meshgrid for the latitudes and longitudes to match the shape of smoothed_grid
        lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

        # Flatten the grids and smoothed values
        lat_flat = lat_mesh.flatten()
        lon_flat = lon_mesh.flatten()
        values_flat = np.floor(smoothed_grid.flatten() )

    
        # Create a DataFrame with three columns: latitude, longitude, and values
        df_long_format = pd.DataFrame({
            'Latitude': lat_flat,
            'Longitude': lon_flat,
            'ColumnAmountO3': values_flat
        })

        # Optionally remove rows where values are NaN (if you only want grid cells with data)
        df_long_format = df_long_format.dropna(subset=['ColumnAmountO3'])

        # add time indices from original dataframe
        df_long_format['Orbit'] = self.df['Orbit'][:len(df_long_format)].reset_index(drop=True)
        df_long_format['Hours_elapsed'] = self.df['Hours_elapsed'][:len(df_long_format)].reset_index(drop=True)
        return df_long_format
    


    def space_center(df: pd.DataFrame) -> np.ndarray:
        locs = np.array( df[['Latitude','Longitude']] )
        lat_min, lat_max = 30, 35
        lon_min, lon_max = 100, 110

        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        center = np.array([[lat_center, lon_center]]) # query expect 2-d array input

        tree = BallTree(locs, metric='euclidean')
        
        dist, ind = tree.query(center, k=1)  
            
        # Return the index of the nearest point

        return locs[ind[0][0]]