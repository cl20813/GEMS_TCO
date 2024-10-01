
# work environment: jl2815
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from skgstat import Variogram
import numpy as np
from typing import Callable   # nearest neighbor function input type
import sklearn.neighbors  # nearest neighbor
from sklearn.neighbors import BallTree # for space_center function
import sys 

from GEMS_TCO import orbitmap
from GEMS_TCO import smoothspace
'''
take orbit_map from 
instance = orbitmap.MakeOrbitdata(df)
orbit_map = instance.makeorbitmap()

orbit_map has a data of form 'y23m07day01_1'
'''



import numpy as np
from scipy.spatial.distance import pdist, squareform

class timeseries:
    def __init__(self, map: dict):
        self.map = map

    def timeseries_var_23_4(self):
        ori_var_timeseries = []
        key_list = sorted(self.map)

        for i in range(len(key_list)):
            cur = self.map[key_list[i]]
            ori_var_timeseries.append(np.var(cur['ColumnAmountO3']))

        ori_var_timeseries = [i for i in ori_var_timeseries]
        ori_var_timeseries = pd.Series(ori_var_timeseries)

        x_with_gaps = []

        for i in range(19):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ [456,457,458,459,460, 461,462]

        x_with_gaps3 = []
        for i in range(20, 240 // 8):
            x_with_gaps3.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ x_with_gaps3
     

        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Variance')
        plt.title('Variance of Ozone values within each orbit')

        day_labels=[]
        day_positions = []

        for i in range(0, 20*8, 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
                        # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.axvline(x= x_with_gaps[ (160-1)], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
        day_labels.append(f"d{21}")
        day_positions.append(x_with_gaps[ (160-1)])

        for i in range(167, len(ori_var_timeseries), 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            day_labels.append(f"d{ (i+1) // 8 + 1}")
            day_positions.append(x_with_gaps[i])
        # time = list(range())
        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)

        # Show plot
        # plt.legend()
        self.ori_var_timeseries = ori_var_timeseries
        plt.show()
        return ori_var_timeseries

    def timeseries_var_23_7(self):
        ori_var_timeseries = []
        key_list = sorted(self.map)

        for i in range(len(key_list)):
            cur = self.map[key_list[i]]
            ori_var_timeseries.append(np.var(cur['ColumnAmountO3']))

        ori_var_timeseries = [i for i in ori_var_timeseries]
        ori_var_timeseries = pd.Series(ori_var_timeseries)

        x_with_gaps = []

        for i in range(12):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ [288,289,290,291,292,293,294]

        x_with_gaps3 = []
        for i in range(13, 240 // 8):
            x_with_gaps3.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ x_with_gaps3
     

        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Variance')
        plt.title('Variance of Ozone values within each orbit')

        day_labels=[]
        day_positions = []

        for i in range(0, 13*8, 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
                        # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.axvline(x= x_with_gaps[ 104-1], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
        day_labels.append(f"d{14}")
        day_positions.append(x_with_gaps[104-1])

        for i in range(111, len(ori_var_timeseries), 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            day_labels.append(f"d{ (i+1) // 8 + 1}")
            day_positions.append(x_with_gaps[i])
        # time = list(range())
        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        self.ori_var_timeseries = ori_var_timeseries
        plt.show()
        return ori_var_timeseries
    
    def timeseries_var_24_7(self):
        ori_var_timeseries = []
        key_list = sorted(self.map)

        for i in range(len(key_list)):
            cur = self.map[key_list[i]]
            ori_var_timeseries.append(np.var(cur['ColumnAmountO3']))

        ori_var_timeseries = [i for i in ori_var_timeseries]
        ori_var_timeseries = pd.Series(ori_var_timeseries)

        x_with_gaps = []
        for i in range( len(ori_var_timeseries) // 8):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))


    
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Variance')
        plt.title('Variance of Ozone values within each orbit')

        # time = list(range())
        # Loop to add horizontal lines for every 8th index
        
        day_labels = []
        day_positions = []

        for i in range(0, len(x_with_gaps), 8):
            plt.axvline(x=x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            
            # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        # Set custom x-ticks to display only the day labels and remove other numbers
        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)

            
        # Show plot
        # plt.legend()
        self.ori_var_timeseries = ori_var_timeseries
        plt.show()
        return ori_var_timeseries

    def timeseries_variance_table(self,ori_var_timeseries):
        for t in range(1,31):
            a= ori_var_timeseries[8*(t-1): 8*(t-1)+8 ]
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )

    def timeseries_semivariogram_23_4(self, target_distance,tolerance):
        ori_semi_var_timeseries = []
        # target_distance = 0.3 # The specific distance you're interested in

 
        key_list = sorted(self.map)
        for i in range(len(key_list)):
            cur_data = self.map[key_list[i]]
            coordinates = np.array(cur_data[['Latitude', 'Longitude']])
            values = np.array(cur_data['ColumnAmountO3'])
                
                # Calculate the pairwise distances between all points
            pairwise_distances = squareform(pdist(coordinates))
                
                # Calculate the semivariance for the pairs with distances near the target distance
            # tolerance = 0.03  # Small tolerance to capture points near the target distance
            valid_pairs = np.where((pairwise_distances >= target_distance - tolerance) & 
                                        (pairwise_distances <= target_distance + tolerance))
                
            if len(valid_pairs[0]) == 0:
                print(f"No valid pairs found for t{j+1:02d}_{i+1} at distance {target_distance}")
                ori_semi_var_timeseries.append(np.nan)
                continue
            
            # Compute the semivariance for those valid pairs
            semivariances = 0.5 * np.mean((values[valid_pairs[0]] - values[valid_pairs[1]])**2)
            
            # Normalize the semivariance
            variance_of_data = np.var(values)
            normalized_semivariance = semivariances / variance_of_data
            
            # Append the normalized semivariance to the timeseries
            ori_semi_var_timeseries.append(normalized_semivariance)

        x_with_gaps = []

        for i in range(19):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ [456,457,458,459,460, 461,462]

        x_with_gaps3 = []
        for i in range(20, 240 // 8):
            x_with_gaps3.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ x_with_gaps3


        ori_semi_var_timeseries = [i for i in ori_semi_var_timeseries]
        ori_semi_var_timeseries = pd.Series(ori_semi_var_timeseries)
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_semi_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Semivariogram')
        plt.title(f'Semivariograms within each orbit (lag {target_distance})')


        day_labels=[]
        day_positions = []

        # Loop to add horizontal lines for every 8th index
        for i in range(0, 20*8, 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
                        # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.axvline(x= x_with_gaps[160-1], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
        day_labels.append(f"d{21}")
        day_positions.append(x_with_gaps[160-1])  

        for i in range(167, len(ori_semi_var_timeseries), 8):
            plt.axvline(x= x_with_gaps[i]-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            day_labels.append(f"d{ (i+1) // 8 + 1}")
            day_positions.append(x_with_gaps[i])
        # time = list(range())
        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        plt.show()
        return ori_semi_var_timeseries

    def timeseries_semivariogram_23_7(self, target_distance,tolerance):
        ori_semi_var_timeseries = []
        # target_distance = 0.3 # The specific distance you're interested in

 
        key_list = sorted(self.map)
        for i in range(len(key_list)):
            cur_data = self.map[key_list[i]]
            coordinates = np.array(cur_data[['Latitude', 'Longitude']])
            values = np.array(cur_data['ColumnAmountO3'])
                
                # Calculate the pairwise distances between all points
            pairwise_distances = squareform(pdist(coordinates))
                
                # Calculate the semivariance for the pairs with distances near the target distance
            # tolerance = 0.03  # Small tolerance to capture points near the target distance
            valid_pairs = np.where((pairwise_distances >= target_distance - tolerance) & 
                                        (pairwise_distances <= target_distance + tolerance))
                
            if len(valid_pairs[0]) == 0:
                print(f"No valid pairs found for t{j+1:02d}_{i+1} at distance {target_distance}")
                ori_semi_var_timeseries.append(np.nan)
                continue
            
            # Compute the semivariance for those valid pairs
            semivariances = 0.5 * np.mean((values[valid_pairs[0]] - values[valid_pairs[1]])**2)
            
            # Normalize the semivariance
            variance_of_data = np.var(values)
            normalized_semivariance = semivariances / variance_of_data
            
            # Append the normalized semivariance to the timeseries
            ori_semi_var_timeseries.append(normalized_semivariance)

        x_with_gaps = []

        for i in range(12):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ [288,289,290,291,292,293,294]  # not 272... 278

        x_with_gaps3 = []
        for i in range(13, 240 // 8):
            x_with_gaps3.extend(range(i * 24, i * 24 + 8))

        x_with_gaps = x_with_gaps+ x_with_gaps3


        ori_semi_var_timeseries = [i for i in ori_semi_var_timeseries]
        ori_semi_var_timeseries = pd.Series(ori_semi_var_timeseries)
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_semi_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Semivariogram')
        plt.title(f'Semivariograms within each orbit (lag {target_distance})')


        day_labels=[]
        day_positions = []

        # Loop to add horizontal lines for every 8th index
        for i in range(0, 13*8, 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
                # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.axvline(x= x_with_gaps[104-1], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
        day_labels.append(f"d{14}")
        day_positions.append(x_with_gaps[104-1])

        for i in range(111, len(ori_semi_var_timeseries), 8):
            plt.axvline(x= x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            day_labels.append(f"d{ (i+1) // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        plt.show()
        return ori_semi_var_timeseries

    def timeseries_semivariogram_24_7(self, target_distance,tolerance):
        ori_semi_var_timeseries = []
        # target_distance = 0.3 # The specific distance you're interested in

 
        key_list = sorted(self.map)
        for i in range(len(key_list)):
            cur_data = self.map[key_list[i]]
            coordinates = np.array(cur_data[['Latitude', 'Longitude']])
            values = np.array(cur_data['ColumnAmountO3'])
                
                # Calculate the pairwise distances between all points
            pairwise_distances = squareform(pdist(coordinates))
                
                # Calculate the semivariance for the pairs with distances near the target distance
            # tolerance = 0.03  # Small tolerance to capture points near the target distance
            valid_pairs = np.where((pairwise_distances >= target_distance - tolerance) & 
                                        (pairwise_distances <= target_distance + tolerance))
                
            if len(valid_pairs[0]) == 0:
                print(f"No valid pairs found for t{j+1:02d}_{i+1} at distance {target_distance}")
                ori_semi_var_timeseries.append(np.nan)
                continue
            
            # Compute the semivariance for those valid pairs
            semivariances = 0.5 * np.mean((values[valid_pairs[0]] - values[valid_pairs[1]])**2)
            
            # Normalize the semivariance
            variance_of_data = np.var(values)
            normalized_semivariance = semivariances / variance_of_data
            
            # Append the normalized semivariance to the timeseries
            ori_semi_var_timeseries.append(normalized_semivariance)

        x_with_gaps = []

        for i in range( len(ori_semi_var_timeseries) // 8):
            x_with_gaps.extend(range(i * 24, i * 24 + 8))



        ori_semi_var_timeseries = [i for i in ori_semi_var_timeseries]
        ori_semi_var_timeseries = pd.Series(ori_semi_var_timeseries)
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_semi_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Semivariogram')
        plt.title(f'Semivariograms within each orbit (lag {target_distance})')


        day_labels=[]
        day_positions = []
        # Loop to add horizontal lines for every 8th index
        for i in range(0, len(ori_semi_var_timeseries), 8):
            plt.axvline(x= x_with_gaps[i]-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        plt.show()
        return ori_semi_var_timeseries
    
        '''
        note that january hours span from 00 to 05 no 06 or 07.
        '''

    
    def timeseries_semivariogram_24_jan(self, target_distance,tolerance):  
        ori_semi_var_timeseries = []
        # target_distance = 0.3 # The specific distance you're interested in

 
        key_list = sorted(self.map)
        for i in range(len(key_list)):
            cur_data = self.map[key_list[i]]
            coordinates = np.array(cur_data[['Latitude', 'Longitude']])
            values = np.array(cur_data['ColumnAmountO3'])
                
                # Calculate the pairwise distances between all points
            pairwise_distances = squareform(pdist(coordinates))
                
                # Calculate the semivariance for the pairs with distances near the target distance
            # tolerance = 0.03  # Small tolerance to capture points near the target distance
            valid_pairs = np.where((pairwise_distances >= target_distance - tolerance) & 
                                        (pairwise_distances <= target_distance + tolerance))
                
            if len(valid_pairs[0]) == 0:
                print(f"No valid pairs found for t{j+1:02d}_{i+1} at distance {target_distance}")
                ori_semi_var_timeseries.append(np.nan)
                continue
            
            # Compute the semivariance for those valid pairs
            semivariances = 0.5 * np.mean((values[valid_pairs[0]] - values[valid_pairs[1]])**2)
            
            # Normalize the semivariance
            variance_of_data = np.var(values)
            normalized_semivariance = semivariances / variance_of_data
            
            # Append the normalized semivariance to the timeseries
            ori_semi_var_timeseries.append(normalized_semivariance)

        x_with_gaps = []

        for i in range( len(ori_semi_var_timeseries) // 6):
            x_with_gaps.extend(range(i * 24, i * 24 + 6))



        ori_semi_var_timeseries = [i for i in ori_semi_var_timeseries]
        ori_semi_var_timeseries = pd.Series(ori_semi_var_timeseries)
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_semi_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Semivariogram')
        plt.title(f'Semivariograms within each orbit (lag {target_distance})')


        day_labels=[]
        day_positions = []
        # Loop to add horizontal lines for every 8th index
        for i in range(0, len(ori_semi_var_timeseries), 6):
            plt.axvline(x= x_with_gaps[i]-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 6 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        plt.show()
        return ori_semi_var_timeseries
    
    def timeseries_semivariogram_table(self,ori_semi_var_timeseries):
        for t in range(1,31):
            a= ori_semi_var_timeseries[8*(t-1): 8*(t-1)+8 ]
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )

    def timeseries_semivariogram_table_23_4(self,ori_semi_var_timeseries):
        for t in range(1,20):
            a= ori_semi_var_timeseries[8*(t-1): 8*(t-1)+8 ]
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )
        
        a = ori_semi_var_timeseries[152:159]
        print(''.join([ f'{np.round(x,2)}&' for x in a]) )

        for t in range(21,31):
            a= ori_semi_var_timeseries[8*(t-1)-1: 8*(t-1)-1+8 ] ##  tricky here
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )

    def timeseries_semivariogram_table_23_7(self,ori_semi_var_timeseries):
        for t in range(1,13):
            a= ori_semi_var_timeseries[8*(t-1): 8*(t-1)+8 ]
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )

        a = ori_semi_var_timeseries[96:103]
        print(''.join([ f'{np.round(x,2)}&' for x in a]) )

        for t in range(14,31):
            a= ori_semi_var_timeseries[8*(t-1)-1: 8*(t-1)-1+8 ] ##  tricky here
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )


    def timeseries_semivariogram_table_jan(self,ori_semi_var_timeseries):
        for t in range(1,31):
            a= ori_semi_var_timeseries[6*(t-1): 6*(t-1)+6 ]
            print(''.join([ f'{np.round(x,2)}&' for x in a]) )
    



class diff_timeseries(timeseries):
    def __init__(self, map: dict):
        super().__init__(map)
 

    def korbitalg(self, k): # year last two digit
        tmp = list( self.map.keys())[0]
        year = int(tmp[1:3])
        month = int(tmp[4:6])

        globals()[f'sparse_map_{year}_{month:02d}_{k}_orbitlag']={}
        for j in range(1,31):
            for i in range(1,9-k):
                cur_key = f'y{year}m{month:02d}day{j:02d}_{i+k}'
                korbitlag_key = f'y{year}m{month:02d}day{j:02d}_{i}'
                if cur_key != 'y{year}m07day13_8' and korbitlag_key != 'y{year}m07day13_8':
                    differenced_data = self.map[cur_key]['ColumnAmountO3'] - self.map[korbitlag_key]['ColumnAmountO3']
                globals()[f'sparse_map_{year}_{month:02d}_{k}_orbitlag'][f'y{year}m{month:02d}day{j:02d}_{i}_{k}orbitlag'] = pd.DataFrame({'Latitude':self.map[cur_key]['Latitude'], 'Longitude':self.map[cur_key]['Longitude'],'ColumnAmountO3': differenced_data})
        return globals()[f'sparse_map_{year}_{month:02d}_{k}_orbitlag']
    

    def kdaylag(self,k):
        tmp = list( self.map.keys())[0]
        year = int(tmp[1:3])
        month = int(tmp[4:6])

        globals()[f'sparse_map_{year}_{month:02d}_{k}_daylag']={}
        for j in range(1,(31-k)):
            for i in range(1,9):
                cur_key = f'y{year}m{month:02d}day{j+k:02d}_{i}'
                korbitlag_key = f'y{year}m{month:02d}day{j:02d}_{i}'
                if cur_key != 'y{year}m07day13_8' and korbitlag_key != 'y{year}m07day13_8':
                    differenced_data = self.map[cur_key]['ColumnAmountO3'] - self.map[korbitlag_key]['ColumnAmountO3']
                globals()[f'sparse_map_{year}_{month:02d}_{k}_daylag'][f'y{year}m{month:02d}day{j:02d}_{i}_{k}daylag'] = pd.DataFrame({'Latitude':self.map[cur_key]['Latitude'], 'Longitude':self.map[cur_key]['Longitude'],'ColumnAmountO3': differenced_data})
 
        return globals()[f'sparse_map_{year}_{month:02d}_{k}_daylag']

    def timeseries_var_24_orbitdiff(self,k):
        ori_var_timeseries = []
        key_list = sorted(self.map)

        for i in range(len(key_list)):
            cur = self.map[key_list[i]]
            ori_var_timeseries.append(np.var(cur['ColumnAmountO3']))

        ori_var_timeseries = [i for i in ori_var_timeseries]
        ori_var_timeseries = pd.Series(ori_var_timeseries)

        x_with_gaps = []
        for i in range( len(ori_var_timeseries) // (8-k)):
            x_with_gaps.extend(range(i * 24, i * 24 + (8-k)))

        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Variance')
        plt.title('Variance of Ozone values within each orbit')

        # time = list(range())
        # Loop to add horizontal lines for every 8th index
        
        day_labels=[]
        day_positions = []
        for i in range(0, len(x_with_gaps), (8-k)):
            plt.axvline(x=x_with_gaps[i], color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])

        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)
        # Show plot
        # plt.legend()
        self.ori_var_timeseries = ori_var_timeseries
        plt.show()
        return ori_var_timeseries
    
    def timeseries_semivariogram_24_orbitdiff(self, target_distance,tolerance,k):
        ori_semi_var_timeseries = []
        # target_distance = 0.3 # The specific distance you're interested in

 
        key_list = sorted(self.map)
        for i in range(len(key_list)):
            cur_data = self.map[key_list[i]]
            coordinates = np.array(cur_data[['Latitude', 'Longitude']])
            values = np.array(cur_data['ColumnAmountO3'])
                
                # Calculate the pairwise distances between all points
            pairwise_distances = squareform(pdist(coordinates))
                
                # Calculate the semivariance for the pairs with distances near the target distance
            # tolerance = 0.03  # Small tolerance to capture points near the target distance
            valid_pairs = np.where((pairwise_distances >= target_distance - tolerance) & 
                                        (pairwise_distances <= target_distance + tolerance))
                
            if len(valid_pairs[0]) == 0:
                print(f"No valid pairs found for t{j+1:02d}_{i+1} at distance {target_distance}")
                ori_semi_var_timeseries.append(np.nan)
                continue
            
            # Compute the semivariance for those valid pairs
            semivariances = 0.5 * np.mean((values[valid_pairs[0]] - values[valid_pairs[1]])**2)
            
            # Normalize the semivariance
            variance_of_data = np.var(values)
            normalized_semivariance = semivariances / variance_of_data
            
            # Append the normalized semivariance to the timeseries
            ori_semi_var_timeseries.append(normalized_semivariance)

        x_with_gaps = []




        for i in range( len(ori_semi_var_timeseries) // (8-k)):
            x_with_gaps.extend(range(i * 24, i * 24 + (8-k)))




        ori_semi_var_timeseries = [i for i in ori_semi_var_timeseries]
        ori_semi_var_timeseries = pd.Series(ori_semi_var_timeseries)
        plt.figure(figsize=(10,3))
        plt.scatter(x_with_gaps, ori_semi_var_timeseries, marker='o', s= 2)


        plt.xlabel('Orbit Indices (Red lines separate different days)')
        plt.ylabel('Semivariogram')
        plt.title(f'Semivariograms within each orbit (lag {target_distance})')

        day_labels=[]
        day_positions = []
        # Loop to add horizontal lines for every 8th index
        for i in range(0, len(ori_semi_var_timeseries), (8-k)):
            plt.axvline(x= x_with_gaps[i]-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5, )  # Adding a vertical line at every 8th index
                # Collect day labels and positions for xticks
            day_labels.append(f"d{i // 8 + 1}")
            day_positions.append(x_with_gaps[i])
        plt.xticks(ticks=day_positions, labels=day_labels, fontsize = 9)   
        # Show plot
        # plt.legend()
        plt.show()
        return ori_semi_var_timeseries