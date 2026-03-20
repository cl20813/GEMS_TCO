#!/bin/bash
python step1_download_gemsdata_api_122525.py --year 2025 --month 5 --days 1-31 && \
python step1_download_gemsdata_api_122525.py --year 2025 --month 6 --days 1-30 && \
python step1_download_gemsdata_api_122525.py --year 2025 --month 8 --days 1-31 && \
python step1_download_gemsdata_api_122525.py --year 2025 --month 9 --days 1-30
