# work environment: jl2815
import pandas as pd
import pickle
import sys 
import os
from pathlib import Path
#  sys.path
# !pip install numpy==2.0

from GEMS_TCO import configuration as config
from GEMS_TCO import data_preprocess as dmbh

'''
1. Transfer csv file into amarel
2. Transform csv into ORI data map
3. Transform ORI denset data map into coarse data map
'''
# "/home/jl2815/tco/data/pickle_data/" this is amarel_data_load_path
amarel_data_path =  config.amarel_data_load_path
lat_start, lat_end, lon_start, lon_end = 0, 5, 123, 133

years, months = [2024], list( range(7,8))  # years = [2023,2024]

# save ORI dense data map in pickle file
instance = dmbh.MonthAggregatedHashmap(lat_start, lat_end, lon_start, lon_end, years, months)
instance.aggregate_july24topickle(csvfilepath = amarel_data_path)


# transform ORI dense data into coarse map
step_lat, step_lon = 0.044, 0.063   #may 2024 setting
df = pd.read_csv( Path(amarel_data_path) /f'data_2024/data_24_07_0131_N05_E123133.csv')  # MAC
instance = dmbh.center_matching_hour(df, lat_start, lat_end, lon_start, lon_end)  

for year in years:
    for month in months:
        month_str = f"{month:02d}" 
        try:
            # load pickle (dense ORI data)
            pickle_path = os.path.join(amarel_data_path, f'pickle_{year}')
            input_filename = f"orbit_map{str(year)[2:]}_{month_str}.pkl"
            input_filepath = os.path.join(pickle_path, input_filename)
            with open(input_filepath, 'rb') as pickle_file:
                loaded_map = pickle.load(pickle_file)
            center_points = instance.make_center_points(step_lat = step_lat, step_lon= step_lon)
            coarse_cen_map = instance.coarse_by_center(loaded_map, center_points)

            # Save pickle (coarse data)
            output_filename = f"coarse_cen_map{str(year)[2:]}_{month_str}.pkl"
            output_filepath = os.path.join(pickle_path, output_filename)
            with open(output_filepath, 'wb') as pickle_file:
                pickle.dump(coarse_cen_map, pickle_file)
            
            print(f"Successfully processed and saved data for year {str(year)[2:]} month {month_str}.")
        except FileNotFoundError:
            print(f"Warning: File {input_filename} not found. Skipping.")
        except Exception as e:
            print(f"Error processing file {input_filename}: {e}")


### Now make coarse set



