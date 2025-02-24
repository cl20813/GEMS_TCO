import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from skgstat import Variogram
import numpy as np
from typing import Callable   # nearest neighbor function input type
import sklearn.neighbors  # nearest neighbor
from sklearn.neighbors import BallTree # for space_center function


class space_average:

    """
    A class for spatial averaging and center-point operations within a defined spatial domain.

    Methods:
        space_avg():
            Performs spatial averaging of data into a grid.
        space_center(df: pd.DataFrame) -> np.ndarray:
            Finds the nearest data point to the center of the spatial domain.

    Parameters:
        df (pd.DataFrame): Input pandas DataFrame containing spatial data.
        lat_resolution (float): Latitude resolution for spatial bins.
        lon_resolution (float): Longitude resolution for spatial bins.
        lat_s (float): Start latitude for the spatial domain.
        lat_e (float): End latitude for the spatial domain.
        lon_s (float): Start longitude for the spatial domain.
        lon_e (float): End longitude for the spatial domain.
    """

    def __init__(self, df=None, lat_resolution=None, lon_resolution=None, lat_s=None,lat_e=None, lon_s=None,lon_e=None ):
        self.df = df
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.lat_s = lat_s
        self.lat_e = lat_e
        self.lon_s = lon_s
        self.lon_e = lon_e

    def space_avg(self):
        """
        Performs spatial averaging of data into a grid based on latitude and longitude bins.

        Returns:
            pd.DataFrame: A DataFrame in long format with averaged 'ColumnAmountO3' values, 
                          latitude, longitude, orbit, and hours elapsed.
        """

        # Extract columns as numpy arrays
        latitudes = np.array(self.df['Latitude'])
        longitudes =  np.array(self.df['Longitude'])
        self.values =  np.array( self.df['ColumnAmountO3'])

        # Define spatial domain
        lat_min, lat_max = self.lat_s, self.lat_e
        lon_min, lon_max = self.lon_s, self.lon_e

        # Create bins for latitude and longitude
        lat_bins = np.linspace(lat_min, lat_max , int((lat_max - lat_min) / self.lat_resolution) + 1)
        lon_bins = np.linspace(lon_min, lon_max , int((lon_max - lon_min) / self.lon_resolution) + 1)

        # Digitize coordinates into bins
        lat_indices = np.digitize(latitudes, lat_bins) - 1  # Get bin indices for latitudes. 
        lon_indices = np.digitize(longitudes, lon_bins) - 1 # np.digitize returns 1 based indice, so subtract 1

        # Initialize the smoothed grid and counts
        smoothed_grid = np.full( (len(lat_bins) - 1, len(lon_bins) - 1), np.nan) # 

        # Create a 2D array to count the number of points per cell
        counts = np.zeros_like(smoothed_grid)

        # Aggregate data into bins
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

        # Compute bin centers
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
    
    def space_center(self, df: pd.DataFrame) -> np.ndarray:
        """
        Finds the nearest data point to the center of the spatial domain.

        Returns:
            np.ndarray: A 2D array containing the latitude and longitude of the nearest point to the center.
        """
        # Extract locations from the DataFrame
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            raise ValueError("Input DataFrame must contain 'Latitude' and 'Longitude' columns.")
        
        locs = np.array( df[['Latitude','Longitude']] )
        lat_min, lat_max = self.lat_s, self.lat_e
        lon_min, lon_max = self.lon_s, self.lon_e

        # Compute the center of the spatial domain
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        center = np.array([[lat_center, lon_center]]) # query expect 2-d array input

        # Build a BallTree for efficient nearest neighbor search
        tree = BallTree(locs, metric='euclidean')
        
        # Find the nearest point to the center
        dist, ind = tree.query(center, k=1)  
            
        # Return the index of the nearest point
        return locs[ind[0][0]]