# Expanded Bounds Preprocessing Runbook

Target domain:

```text
latitude  [-3, 7]
longitude [111, 131]
month     July
years     2022, 2023, 2024, 2025
```

The `.py` files are the canonical pipeline.  They support both the historical
narrow domain `[-3, 2] x [121, 131]` and the expanded domain through `--bounds`.
The notebooks are now optional interactive records only.

## 0. Setup

```bash
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
```

## 1. Step 2: Expanded CSV To Orbit Map

This reads the already-created expanded CSV files:

```text
/Users/joonwonlee/Documents/GEMS_DATA/data_YYYY/data_YY_07_0131_N-37_E111131.csv
```

and writes bounds-aware orbit maps:

```text
/Users/joonwonlee/Documents/GEMS_DATA/pickle_YYYY/orbit_map_lat-3to7_lon111to131_YY_07.pkl
```

Run:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step2_truncate_cvs_pickle_by_monthyear_020226.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --bounds=-3,7,111,131 \
  --make-orbit-map \
  --overwrite
```

## 2. Step 3: Expanded Orbit Map To Regular Grid

This reads:

```text
orbit_map_lat-3to7_lon111to131_YY_07.pkl
```

and writes:

```text
tco_grid_lat-3to7_lon111to131_YY_07.pkl
```

Run:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step3_enforce_regular_grid_031726.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --bounds=-3,7,111,131 \
  --steps=0.044,0.063 \
  --overwrite
```

## 3. Validate Expanded Coverage

Run this before transferring to Amarel:

```bash
/opt/anaconda3/envs/faiss_env/bin/python - <<'PY'
import pandas as pd
from pathlib import Path

for year in [2022, 2023, 2024, 2025]:
    yy = str(year)[2:]
    p = Path(f"/Users/joonwonlee/Documents/GEMS_DATA/pickle_{year}/tco_grid_lat-3to7_lon111to131_{yy}_07.pkl")
    obj = pd.read_pickle(p)
    k = sorted(obj)[0]
    df = obj[k]
    finite = df.dropna(subset=["ColumnAmountO3"])
    print(year, k, "finite_n", len(finite))
    print("  lat", float(finite.Latitude.min()), float(finite.Latitude.max()))
    print("  lon", float(finite.Longitude.min()), float(finite.Longitude.max()))
PY
```

Expected result: each year should show finite values covering approximately:

```text
lat [-3, 7]
lon [111, 131]
```

If it shows `lat [-3, 2]` and `lon [121, 131]`, then the grid was rebuilt from
the old narrow `orbit_mapYY_07.pkl` rather than the expanded orbit map.

## 4. Step 4: Optional Merge

This is only needed if downstream code expects the merged Apr-to-Sep style
folder.  For the current July-only expanded run, it creates:

```text
/Users/joonwonlee/Documents/GEMS_DATA/Apr_to_Sep_lat-3to7_lon111to131/
```

Run:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step4_merge_apr_sep_multiyear.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --lat-lon-bounds=-3,7,111,131
```

## 5. Step 5: Transfer To Amarel

For the fitting script that reads monthly pkl files directly, this is the
important transfer:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step5_transfer_to_amarel_032626.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --only-extra-bounds
```

If the merged folder is also needed:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step5_transfer_to_amarel_032626.py \
  --years 2022 2023 2024 2025 \
  --only-extra-merged
```

## Notes

- Do not use the old untagged `orbit_mapYY_07.pkl` for expanded fitting.
- The expanded orbit map filename must include:
  `lat-3to7_lon111to131`.
- The expanded fitting script should keep `--require-region-cover` enabled so
  accidentally narrow files fail early.

## Returning To The Narrow Domain

Use the same scripts with the historical bounds:

```bash
/opt/anaconda3/envs/faiss_env/bin/python step2_truncate_cvs_pickle_by_monthyear_020226.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --bounds=-3,2,121,131 \
  --make-orbit-map

/opt/anaconda3/envs/faiss_env/bin/python step3_enforce_regular_grid_031726.py \
  --years 2022 2023 2024 2025 \
  --months 7 \
  --bounds=-3,2,121,131 \
  --steps=0.044,0.063
```

Those commands read/write the original untagged names:

```text
orbit_mapYY_07.pkl
tco_grid_YY_07.pkl
```
