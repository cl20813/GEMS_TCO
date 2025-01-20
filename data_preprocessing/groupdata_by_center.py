# work environment: jl2815
import pandas as pd
import pickle
import sys 
import os
#  sys.path
# !pip install numpy==2.0

from GEMS_TCO import orbitmap
from GEMS_TCO.smoothspace import space_average


# Base file path and settings
base_path = "/home/jl2815/tco/data/pickle_data"
lat_start, lat_end, lon_start, lon_end = 5, 10, 110, 120

years = [2023,2024]
# Loop through months
for year in years:
    for month in range(1, 13):  
        try:
            # Construct filenames dynamically
            month_str = f"{month:02d}"  # Ensure month is zero-padded
            if month == 2 and year==2023:
                day_str = "0128"  # Handle February specifically
            elif month ==2 and year==2024:
                day_str = "0129"
            else:
                day_str = "0131" if (month in [1, 3, 5, 7, 8, 10, 12]) else "0130"
    
            input_filename = f"data_{year}/data_{str(year)[2:]}_{month_str}_{day_str}_N510_E110120.csv"
            input_filepath = os.path.join(base_path, input_filename)
            
            # Read data
            print(f"Reading file: {input_filepath}")
            df = pd.read_csv(input_filepath)
            
            # Process data
            instance = orbitmap.MakeOrbitdata(df, lat_start, lat_end, lon_start, lon_end)
            orbit_map = instance.group_data_by_orbits()
            
            output_path = os.path.join(base_path, f'pickle_{year}')

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


