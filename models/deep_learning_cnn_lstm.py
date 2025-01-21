# Data manipulation and analysis
import pandas as pd
import pickle
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Custom imports
from GEMS_TCO import orbitmap
from GEMS_TCO import kernels
# Initiate instance

print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")


def main():
    df = pd.read_csv('/home/jl2815/tco/data/pickle_data/data_2024/data_24_07_0131_N510_E110120.csv')
    # df = pd.read_csv("C:\\Users\\joonw\\TCO\\data_engineering\\data_2024\\data_24_07_0131_N510_E110120.csv")
    instance = orbitmap.MakeOrbitdata(df, 5,10,110,120)

    # Load the one dictionary to set spaital coordinates
    filepath = "/home/jl2815/tco/data/pickle_data/pickle_2023/coarse_cen_map23_01.pkl"

    with open(filepath, 'rb') as pickle_file:
        coarse_dict_24_1 = pickle.load(pickle_file)

    print(coarse_dict_24_1.keys())
    sample_df = coarse_dict_24_1['y24m01day01_hm02:12']

    sample_key = coarse_dict_24_1.get('y24m01day01_hm02:12')
    if sample_key is None:
        print("Key 'y24m01day01_hm02:12' not found in the dictionary.")

    rho_lat = 20
    rho_lon = 20
    lat_n = sample_df['Latitude'].unique()[::rho_lat]
    lon_n = sample_df['Longitude'].unique()[::rho_lon]

    # Set spatial coordinates for each dataset
    coarse_dicts = {}

    years = ['2023','2024']
    for year in years:
        for month in range(1, 13):  # Iterate over all months
            filepath = f"/home/jl2815/tco/data/pickle_data/pickle_{year}\\coarse_cen_map{year[2:]}_{month:02d}.pkl"
            with open(filepath, 'rb') as pickle_file:
                loaded_map = pickle.load(pickle_file)
                for key in loaded_map:
                    tmp_df = loaded_map[key]
                    coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                    coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)

            print(f"Finished processing {year} {month}.")

    # now aggregate data into a single dataframe

    df_list = []

    for key in coarse_dicts:
        df_list.append(coarse_dicts[key])

    df_entire = pd.concat(df_list, axis=0, ignore_index=True)


    # CNN Model for Spatial Feature Extraction
    class FeatureExtractorCNN(nn.Module):
        def __init__(self, cnn_channels, output_size):
            super(FeatureExtractorCNN, self).__init__()
            self.conv1 = nn.Conv2d(cnn_channels, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(64 * 5 * 10, output_size)  # Adjust based on grid size (5x10 here)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x

    # Custom Dataset for Ozone Data
    class OzoneDataset(Dataset):
        def __init__(self, data, num_latitude, num_longitude, daily_cycle_len, monthly_cycle_len, three_month_cycle_len):
            self.data = data
            self.num_latitude = num_latitude
            self.num_longitude = num_longitude
            self.daily_cycle_len = daily_cycle_len
            self.monthly_cycle_len = monthly_cycle_len
            self.three_month_cycle_len = three_month_cycle_len

            # Cache unique latitude/longitude indices for faster grid construction
            self.latitudes = sorted(data['Latitude'].unique())
            self.longitudes = sorted(data['Longitude'].unique())
            self.time_steps = sorted(data['Hours_elapsed'].unique())
            self.lat_lon_map = {
                (row['Latitude'], row['Longitude']): (self.latitudes.index(row['Latitude']), self.longitudes.index(row['Longitude']))
                for _, row in data.iterrows()
            }

            # Prepare data for sequences
            self.prepared_data = self.prepare_data()

        def prepare_data(self):
            X_daily, X_monthly, X_three_month, y = [], [], [], []
            for i in range(len(self.time_steps) - self.three_month_cycle_len):
                daily_seq, monthly_seq, three_month_seq = [], [], []

                for t in range(i, i + self.three_month_cycle_len):
                    sub_data = self.data[self.data['Hours_elapsed'] == self.time_steps[t]]
                    grid = np.zeros((self.num_latitude, self.num_longitude))
                    
                    for _, row in sub_data.iterrows():
                        lat_idx, lon_idx = self.lat_lon_map[(row['Latitude'], row['Longitude'])]
                        grid[lat_idx, lon_idx] = row['ColumnAmountO3']

                    if t < i + self.daily_cycle_len:
                        daily_seq.append(grid)
                    if t < i + self.monthly_cycle_len:
                        monthly_seq.append(grid)
                    three_month_seq.append(grid)

                X_daily.append(torch.tensor(np.array(daily_seq), dtype=torch.float32))
                X_monthly.append(torch.tensor(np.array(monthly_seq), dtype=torch.float32))
                X_three_month.append(torch.tensor(np.array(three_month_seq), dtype=torch.float32))
                y.append(torch.tensor(np.array(three_month_seq[-1]), dtype=torch.float32))  # Target is last grid in the sequence

            return X_daily, X_monthly, X_three_month, y

        def __len__(self):
            return len(self.prepared_data[0])

        def __getitem__(self, idx):
            X_daily, X_monthly, X_three_month, y = self.prepared_data
            return X_daily[idx], X_monthly[idx], X_three_month[idx], y[idx]

    # Multi-Scale LSTM for Temporal Modeling
    class MultiScaleLSTM(nn.Module):
        def __init__(self, cnn_channels, cnn_output_size, lstm_hidden_size, lstm_num_layers=1, lstm_dropout=0.25):
            super(MultiScaleLSTM, self).__init__()
            self.cnn = FeatureExtractorCNN(cnn_channels, cnn_output_size)
            self.lstm_daily = nn.LSTM(cnn_output_size, lstm_hidden_size, num_layers=lstm_num_layers, dropout=lstm_dropout, batch_first=True)
            self.lstm_monthly = nn.LSTM(cnn_output_size, lstm_hidden_size, num_layers=lstm_num_layers, dropout=lstm_dropout, batch_first=True)
            self.lstm_three_month = nn.LSTM(cnn_output_size, lstm_hidden_size, num_layers=lstm_num_layers, dropout=lstm_dropout, batch_first=True)
            # self.fc = nn.Linear(lstm_hidden_size * 3, 1)
            self.fc = nn.Linear(lstm_hidden_size * 3, 5 * 10)

        def forward(self, X_daily, X_monthly, X_three_month):
            def extract_features(X_seq):
                batch_size, seq_len, height, width = X_seq.shape
                # Process each timestep with CNN
                features = [self.cnn(X_seq[:, t].unsqueeze(1)) for t in range(seq_len)]
                return torch.stack(features, dim=1)  # Shape: [batch_size, seq_len, cnn_output_size]

            # Extract spatial features for each sequence
            daily_features = extract_features(X_daily)
            monthly_features = extract_features(X_monthly)
            three_month_features = extract_features(X_three_month)

            # Process temporal features using LSTMs
            lstm_out_daily, _ = self.lstm_daily(daily_features)
            lstm_out_monthly, _ = self.lstm_monthly(monthly_features)
            lstm_out_three_month, _ = self.lstm_three_month(three_month_features)

            # Concatenate the final outputs from each LSTM
            combined_features = torch.cat((lstm_out_daily[:, -1, :], lstm_out_monthly[:, -1, :], lstm_out_three_month[:, -1, :]), dim=1)

            output = self.fc(combined_features)
            output = output.view(-1, 5, 10)  # Reshape to grid dimensions

            # Final prediction
            return output

    # Parameters
    num_latitude = 5
    num_longitude = 10
    cnn_channels = 1  # Grayscale-like input
    cnn_output_size = 64  # Number of features extracted by CNN
    daily_cycle_len = 8
    monthly_cycle_len = 24    #240
    three_month_cycle_len = 60  # 720 tmp for week
    lstm_hidden_size = 64 # 128

    # Load dataset (example)
    data = df_entire
    dataset = OzoneDataset(data, num_latitude, num_longitude, daily_cycle_len, monthly_cycle_len, three_month_cycle_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    # model = MultiScaleLSTM(cnn_channels, cnn_output_size, lstm_hidden_size)
    # model.train()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiScaleLSTM(cnn_channels, cnn_output_size, lstm_hidden_size)


    # Loss and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        for X_daily, X_monthly, X_three_month, y in dataloader:
            # Move data to GPU
            X_daily = X_daily.to(device)
            X_monthly = X_monthly.to(device)
            X_three_month = X_three_month.to(device)
            y = y.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_daily, X_monthly, X_three_month)
            loss = criterion(predictions,   y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    main()
