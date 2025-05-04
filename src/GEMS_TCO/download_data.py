import requests
import os
from tqdm import tqdm  # Progress bar


class Download_file:
    def __init__(self, year:int,month:int, days:list):
        self.year = year
        self.month = month
        self.days = days
        self.hours = [11] + list(range(0,8))
        self.base_url = 'https://nesc.nier.go.kr:38032/api/GK2/L2/O3T/data/getFileItem.do'
        self.api_key = 'api-14a8f20bdde3413fa75c03de34659294'
        self.dates = [f'{self.year}{self.month:02d}{day:02d}{hour:02d}45' for day in self.days for hour in self.hours]
        
        self.output_folder = f'D:\\GEMS_UNZIPPED\\{self.year}{self.month:02d}{self.days[0]:02d}{self.days[-1]:02d}' if len(self.days) > 1 else f'D:\\GEMS_UNZIPPED\\{self.year}{self.month:02d}{self.days[0]:02d}'

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def download_file(self, url, file_name):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(self.output_folder, file_name)
            with open(file_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(1024), desc=f"Downloading {file_name}"):
                    f.write(chunk)
            print(f"Downloaded {file_name}")
        else:
            print(f"Failed to download {file_name}, status code: {response.status_code}")
    
    def run(self):
        for date in self.dates:
            url = f'{self.base_url}?date={date}&key={self.api_key}'
            file_name = f'{date[:8]}_{date[8:]}.nc'  # name of the file to be saved
            self.download_file(url, file_name)
