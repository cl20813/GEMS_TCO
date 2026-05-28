# Micronesia July Step 5 SCP Commands

Target region:

- Latitude: `1N to 6N`
- Longitude: `129E to 139E`
- File tag: `lat1to6_lon129to139`
- Month: July only
- Years: `2022 2023 2024 2025`

Local base:

```bash
/Users/joonwonlee/Documents/GEMS_DATA
```

Amarel base:

```bash
/home/jl2815/tco/data
```

## 1. Test SSH Login

Run this first. If it asks for Duo, password, or hardware-key touch, complete that prompt.

```bash
ssh jl2815@amarel.rutgers.edu
```

Exit after login succeeds:

```bash
exit
```

If SSH key authentication is not loaded locally:

```bash
ssh-add ~/.ssh/id_rsa
```

## 2. Create Remote Directories

```bash
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/data/pickle_2022 /home/jl2815/tco/data/pickle_2023 /home/jl2815/tco/data/pickle_2024 /home/jl2815/tco/data/pickle_2025 /home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139'
```

## 3. Transfer Monthly Grid Pickles

These are the step3 outputs.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat1to6_lon129to139_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat1to6_lon129to139_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat1to6_lon129to139_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat1to6_lon129to139_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
```

## 4. Transfer Merged July Outputs

These are the step4 outputs. The directory name still starts with `Apr_to_Sep` because the existing preprocessing convention uses that folder name, but this run contains July only.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2022.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2023.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2024.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2025.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/

scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/day_index_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/monthly_means_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
```

## 5. One-Shot Transfer Alternative

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat1to6_lon129to139_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat1to6_lon129to139_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat1to6_lon129to139_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat1to6_lon129to139_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/

scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat1to6_lon129to139/* jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/
```

## 6. Verify On Amarel

```bash
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/pickle_2022/tco_grid_lat1to6_lon129to139_22_07.pkl /home/jl2815/tco/data/pickle_2023/tco_grid_lat1to6_lon129to139_23_07.pkl /home/jl2815/tco/data/pickle_2024/tco_grid_lat1to6_lon129to139_24_07.pkl /home/jl2815/tco/data/pickle_2025/tco_grid_lat1to6_lon129to139_25_07.pkl'
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/Apr_to_Sep_lat1to6_lon129to139/'
```

Expected local files created by steps 2-4:

- `pickle_2022/tco_grid_lat1to6_lon129to139_22_07.pkl`
- `pickle_2023/tco_grid_lat1to6_lon129to139_23_07.pkl`
- `pickle_2024/tco_grid_lat1to6_lon129to139_24_07.pkl`
- `pickle_2025/tco_grid_lat1to6_lon129to139_25_07.pkl`
- `Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2022.pkl`
- `Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2023.pkl`
- `Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2024.pkl`
- `Apr_to_Sep_lat1to6_lon129to139/tco_grid_apr_sep_2025.pkl`
- `Apr_to_Sep_lat1to6_lon129to139/day_index_apr_sep_2022_2025.csv`
- `Apr_to_Sep_lat1to6_lon129to139/monthly_means_apr_sep_2022_2025.csv`

Notes from this preprocessing run:

- Step2 rows by year: `2022=3,614,698`, `2023=3,731,892`, `2024=5,098,640`, `2025=5,187,485`.
- Step3 time steps by year: `2022=240`, `2023=248`, `2024=248`, `2025=247`.
- Step3 first-hour finite coverage: latitude approximately `1.03N to 6.00N`, longitude approximately `129.05E to 138.75E`.
- Step4 day index has `123` days, `full=122`, `partial=1`; only `2025-07-24` has `7` hours.
- 2022 July raw denominator check gave `92.64%` step2-good pixels and `7.36%` unusable-for-step2 pixels.
