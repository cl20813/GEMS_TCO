"""
step1_download_gemsdata_api_122525.py

Download GEMS TCO data via API.

Usage:
    python step1_download_gemsdata_api_122525.py --year 2025 --month 5 --days 1-31
    python step1_download_gemsdata_api_122525.py --year 2025 --month 6 --days 1-30
    python step1_download_gemsdata_api_122525.py --year 2025 --month 8 --days 1-31
    python step1_download_gemsdata_api_122525.py --year 2025 --month 9 --days 1-30


Run with caffeinate to prevent sleep:
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st 

chmod +x run_download.sh                                                
caffeinate -i bash run_download.sh   
"""
import sys
import argparse

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from GEMS_TCO import download_data
from GEMS_TCO import configuration as config


def parse_days(days_str: str):
    """Parse days argument: '1-31' → range, '1,2,5' → list."""
    if '-' in days_str:
        start, end = days_str.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(d) for d in days_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description="Download GEMS TCO data")
    parser.add_argument('--year',  type=int, required=True,  help="Year (e.g. 2022)")
    parser.add_argument('--month', type=int, required=True,  help="Month (e.g. 7)")
    parser.add_argument('--days',  type=str, required=True,
                        help="Days as range '1-31' or list '1,2,5'")
    args = parser.parse_args()

    days_list = parse_days(args.days)
    print(f"Downloading: year={args.year}  month={args.month}  days={days_list}")
    print(f"Output folder: {config.portable_disk_path}")

    instance = download_data.Download_file(
        year=args.year,
        month=args.month,
        days=days_list,
        ouptut_folder=config.portable_disk_path
    )
    instance.run()
    print("Done.")


if __name__ == "__main__":
    main()
