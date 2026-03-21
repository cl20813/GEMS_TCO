import requests
import os
import time
from tqdm import tqdm  # Progress bar


class Download_file:
    '''
    This class is intended to download GEMS data. 
    '''
    def __init__(self, year:int,month:int, days:list, ouptut_folder):
        self.year = year
        self.month = month
        self.days = days
        self.hours = [11] + list(range(0,8))
        self.base_url = 'https://nesc.nier.go.kr:38032/api/GK2/L2/O3T/data/getFileItem.do'

        self.api_key = 'api-799aa3c4e040444789658e94e54e1dd3'
        
        self.dates = [f'{self.year}{self.month:02d}{day:02d}{hour:02d}45' for day in self.days for hour in self.hours]
        
        # self.output_folder = f'D:\\GEMS_UNZIPPED\\{self.year}{self.month:02d}{self.days[0]:02d}{self.days[-1]:02d}' if len(self.days) > 1 else f'D:\\GEMS_UNZIPPED\\{self.year}{self.month:02d}{self.days[0]:02d}'
        self.output_folder = ouptut_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def download_file(self, url:str, file_name:str, max_retries:int=5):
        file_path = os.path.join(self.output_folder, file_name)

        # Check expected size from HEAD request and skip if already complete
        try:
            head = requests.head(url, timeout=30)
            expected = int(head.headers.get('Content-Length', 0))
            if expected > 0 and os.path.exists(file_path) and os.path.getsize(file_path) == expected:
                print(f"Already complete, skipping: {file_name}")
                return
        except Exception:
            pass

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=120)
                if response.status_code != 200:
                    print(f"Failed {file_name}: status {response.status_code}")
                    return
                with open(file_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(1024), desc=f"Downloading {file_name}"):
                        f.write(chunk)
                print(f"Downloaded {file_name}")
                return
            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed for {file_name}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)   # remove incomplete file before retry
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # exponential backoff: 2, 4, 8, 16s
        print(f"Giving up on {file_name} after {max_retries} attempts")
    
    def run(self):
        for date in self.dates:
            url = f'{self.base_url}?date={date}&key={self.api_key}'
            file_name = f'{date[:8]}_{date[8:]}.nc'  # name of the file to be saved
            self.download_file(url, file_name)
