# Philippine Sea July Step 5 SCP Commands

Target region:

- Latitude: `15N to 20N`
- Longitude: `121E to 131E`
- File tag: `lat15to20_lon121to131`
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
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/data/pickle_2022 /home/jl2815/tco/data/pickle_2023 /home/jl2815/tco/data/pickle_2024 /home/jl2815/tco/data/pickle_2025 /home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131'
```

## 3. Transfer Monthly Grid Pickles

These are the step3 outputs.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat15to20_lon121to131_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat15to20_lon121to131_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat15to20_lon121to131_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat15to20_lon121to131_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
```

## 4. Transfer Merged July Outputs

These are the step4 outputs. The directory name still starts with `Apr_to_Sep` because the existing preprocessing convention uses that folder name, but this run contains July only.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2022.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2023.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2024.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2025.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/day_index_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/monthly_means_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
```

## 5. One-Shot Transfer Alternative

This sends all Philippine Sea July outputs in fewer commands.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat15to20_lon121to131_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat15to20_lon121to131_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat15to20_lon121to131_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat15to20_lon121to131_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon121to131/* jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/
```

## 6. Verify On Amarel

```bash
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/pickle_2022/tco_grid_lat15to20_lon121to131_22_07.pkl /home/jl2815/tco/data/pickle_2023/tco_grid_lat15to20_lon121to131_23_07.pkl /home/jl2815/tco/data/pickle_2024/tco_grid_lat15to20_lon121to131_24_07.pkl /home/jl2815/tco/data/pickle_2025/tco_grid_lat15to20_lon121to131_25_07.pkl'
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon121to131/'
```

Expected local files created by steps 2-4:

- `pickle_2022/tco_grid_lat15to20_lon121to131_22_07.pkl`
- `pickle_2023/tco_grid_lat15to20_lon121to131_23_07.pkl`
- `pickle_2024/tco_grid_lat15to20_lon121to131_24_07.pkl`
- `pickle_2025/tco_grid_lat15to20_lon121to131_25_07.pkl`
- `Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2022.pkl`
- `Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2023.pkl`
- `Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2024.pkl`
- `Apr_to_Sep_lat15to20_lon121to131/tco_grid_apr_sep_2025.pkl`
- `Apr_to_Sep_lat15to20_lon121to131/day_index_apr_sep_2022_2025.csv`
- `Apr_to_Sep_lat15to20_lon121to131/monthly_means_apr_sep_2022_2025.csv`

Notes from this preprocessing run:

- `2022-07-29` has no source hourly files, so 2022 has `240` time steps.
- `2025-07-24 02:45` is missing, so 2025 has `247` time steps and one partial day in the day index.
- 2023 and 2024 each have `248` time steps.
