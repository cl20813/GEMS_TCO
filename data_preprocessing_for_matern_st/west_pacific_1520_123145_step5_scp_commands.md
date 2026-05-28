# West Pacific July Step 5 SCP Commands

Target region:

- Latitude: `15N to 20N`
- Longitude: `123E to 145E`
- File tag: `lat15to20_lon123to145`
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
ssh jl2815@amarel.rutgers.edu 'mkdir -p /home/jl2815/tco/data/pickle_2022 /home/jl2815/tco/data/pickle_2023 /home/jl2815/tco/data/pickle_2024 /home/jl2815/tco/data/pickle_2025 /home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145'
```

## 3. Transfer Monthly Grid Pickles

These are the step3 outputs.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat15to20_lon123to145_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat15to20_lon123to145_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat15to20_lon123to145_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat15to20_lon123to145_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/
```

## 4. Transfer Merged July Outputs

These are the step4 outputs. The directory name still starts with `Apr_to_Sep` because the existing preprocessing convention uses that folder name, but this run contains July only.

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2022.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2023.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2024.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/
scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2025.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/


scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/day_index_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/


scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/monthly_means_apr_sep_2022_2025.csv jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/
```

## 5. One-Shot Transfer Alternative

```bash
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2022/tco_grid_lat15to20_lon123to145_22_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2022/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2023/tco_grid_lat15to20_lon123to145_23_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2023/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_lat15to20_lon123to145_24_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2024/
scp /Users/joonwonlee/Documents/GEMS_DATA/pickle_2025/tco_grid_lat15to20_lon123to145_25_07.pkl jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/pickle_2025/


scp /Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat15to20_lon123to145/* jl2815@amarel.rutgers.edu:/home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/
```

## 6. Verify On Amarel

```bash
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/pickle_2022/tco_grid_lat15to20_lon123to145_22_07.pkl /home/jl2815/tco/data/pickle_2023/tco_grid_lat15to20_lon123to145_23_07.pkl /home/jl2815/tco/data/pickle_2024/tco_grid_lat15to20_lon123to145_24_07.pkl /home/jl2815/tco/data/pickle_2025/tco_grid_lat15to20_lon123to145_25_07.pkl'
ssh jl2815@amarel.rutgers.edu 'ls -lh /home/jl2815/tco/data/Apr_to_Sep_lat15to20_lon123to145/'
```

Expected local files created by steps 2-4:

- `pickle_2022/tco_grid_lat15to20_lon123to145_22_07.pkl`
- `pickle_2023/tco_grid_lat15to20_lon123to145_23_07.pkl`
- `pickle_2024/tco_grid_lat15to20_lon123to145_24_07.pkl`
- `pickle_2025/tco_grid_lat15to20_lon123to145_25_07.pkl`
- `Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2022.pkl`
- `Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2023.pkl`
- `Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2024.pkl`
- `Apr_to_Sep_lat15to20_lon123to145/tco_grid_apr_sep_2025.pkl`
- `Apr_to_Sep_lat15to20_lon123to145/day_index_apr_sep_2022_2025.csv`
- `Apr_to_Sep_lat15to20_lon123to145/monthly_means_apr_sep_2022_2025.csv`

Notes from this preprocessing run:

- Step2 rows by year: `2022=8,057,344`, `2023=8,222,466`, `2024=9,525,025`, `2025=9,231,801`.
- Step3 finite longitude is approximately `123.01E to 139.52E`; this confirms that the requested `139.5E to 145E` part does not contribute useful GEMS coverage.
- Step2 time steps by year: `2022=240`, `2023=248`, `2024=248`, `2025=247`.
- Step4 day index has `123` days, `full=122`, `partial=1`; only `2025-07-24` has `7` hours.
- 2022 July raw denominator check gave `95.93%` step2-good pixels and `4.07%` unusable-for-step2 pixels.
