"""
step4_merge_apr_sep_multiyear.py
=================================
Merge monthly tco_grid pkl files (April–September) into per-year hashmaps.

Hours_elapsed epoch: Unix (1970-01-01).  477 700 h ≈ 2024-07-01 ✓

Input  : GEMS_DATA/pickle_{year}/tco_grid_{yy}_{mm}.pkl   (from step 3)

Output (saved to GEMS_DATA/):
  Per-year hashmaps (one pkl per year, ~1.4 GB each):
    Apr_to_Sep/tco_grid_apr_sep_2022.pkl   {time_key: DataFrame}
    Apr_to_Sep/tco_grid_apr_sep_2023.pkl
    Apr_to_Sep/tco_grid_apr_sep_2024.pkl
    Apr_to_Sep/tco_grid_apr_sep_2025.pkl

  Shared index files (small, load once):
    Apr_to_Sep/day_index_apr_sep_2022_2025.csv
        date_str, year, month, day, n_hours, key_h0 … key_h7
        NaN in key_hN means that hour is missing for that day.

    monthly_means_apr_sep_2022_2025.csv
        year, month, monthly_mean   (ColumnAmountO3, for centering)

Usage
-----
conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/data_preprocessing_for_matern_st
python step4_merge_apr_sep_multiyear.py

# Subset:
python step4_merge_apr_sep_multiyear.py --years 2024 2025 --months 7 8 9

Loading a specific day downstream
----------------------------------
import pickle, pandas as pd
from pathlib import Path

DATA = Path("/Users/joonwonlee/Documents/GEMS_DATA") / "Apr_to_Sep"

# Load index once at the start of your script
idx_df    = pd.read_csv(DATA / "day_index_apr_sep_2022_2025.csv")
day_index = {
    row["date_str"]: [row[f"key_h{i}"] for i in range(8)
                      if pd.notna(row[f"key_h{i}"])]
    for _, row in idx_df.iterrows()
}
mm_df = pd.read_csv(DATA / "monthly_means_apr_sep_2022_2025.csv")
mm_lookup = {(int(r.year), int(r.month)): r.monthly_mean
             for _, r in mm_df.iterrows()}

# Load one day
date     = "2024-07-02"
year     = int(date[:4])
month    = int(date[5:7])
day_keys = day_index[date]                  # ≤8 actual keys, sorted by hour

merged   = pickle.load(open(DATA / f"tco_grid_apr_sep_{year}.pkl", "rb"))
day_dict = {k: merged[k] for k in day_keys}

data_map, agg = loader.load_working_data(
    coarse_dicts    = day_dict,
    monthly_mean    = mm_lookup[(year, month)],
    idx_for_datamap = [0, len(day_keys)],   # sub-dict has only this day's hours
)
"""

import sys
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
from GEMS_TCO import configuration as config

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_YEARS  = [2022, 2023, 2024, 2025]
DEFAULT_MONTHS = [4, 5, 6, 7, 8, 9]

BASE_PATH = Path(config.mac_data_load_path)
OUT_DIR   = BASE_PATH / "Apr_to_Sep"

_EPOCH = datetime.datetime(1970, 1, 1)   # Hours_elapsed origin


def _hrs_to_date(hours_elapsed: float) -> str:
    dt = _EPOCH + datetime.timedelta(hours=float(hours_elapsed))
    return dt.strftime("%Y-%m-%d")


# ── Load one year-month pkl ───────────────────────────────────────────────────
def load_year_month(year: int, month: int, base_path: Path):
    """
    Load tco_grid_{yy}_{mm}.pkl and return {global_key: DataFrame}.
    Returns empty dict if file not found.
    """
    fname = f"tco_grid_{str(year)[2:]}_{month:02d}.pkl"
    fpath = base_path / f"pickle_{year}" / fname
    if not fpath.exists():
        print(f"  [Skip] not found: {fpath}")
        return {}

    with open(fpath, "rb") as f:
        month_map = pickle.load(f)

    result = {}
    for orig_key, df in month_map.items():
        global_key = f"{year}_{month:02d}_{orig_key}"
        result[global_key] = df.reset_index(drop=True)

    print(f"  [Load] {fname}  ({len(result)} time steps)")
    return result


# ── Per-year merge & save ─────────────────────────────────────────────────────
def build_and_save_year(year: int, months: list, base_path: Path, out_dir: Path):
    """
    Merge all target months for one year → sort → save as single pkl.
    Returns (year_dict, mm_table_for_year).
    """
    year_dict = {}
    mm_table  = {}

    for month in months:
        month_data = load_year_month(year, month, base_path)
        if not month_data:
            continue

        year_dict.update(month_data)

        # Per-month O3 mean
        vals = [pd.to_numeric(df["ColumnAmountO3"], errors="coerce").values
                for df in month_data.values()]
        if vals:
            mm_table[(year, month)] = float(np.nanmean(np.concatenate(vals)))

    if not year_dict:
        print(f"  [Warn] No data for year {year}, skipping pkl save.")
        return {}, {}

    year_dict = dict(sorted(year_dict.items()))

    out_pkl = out_dir / f"tco_grid_apr_sep_{year}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(year_dict, f)
    print(f"  [Saved] {out_pkl.name}  ({len(year_dict)} total time steps)")

    return year_dict, mm_table


# ── Day index ─────────────────────────────────────────────────────────────────
def build_day_index(all_year_dicts: dict):
    """
    Group every time_key by its calendar date (from Hours_elapsed).

    Parameters
    ----------
    all_year_dicts : {year: {time_key: DataFrame}}

    Returns
    -------
    day_index : {date_str: [key_t0, key_t1, ...]}
                Keys within a date are sorted by actual Hours_elapsed.
                Missing hours are absent — no placeholder.
    """
    date_to_keys = defaultdict(list)   # date_str → [(median_hrs, key)]

    for year_dict in all_year_dicts.values():
        for key, df in year_dict.items():
            if "Hours_elapsed" not in df.columns:
                # Fallback: parse year/month from key prefix
                parts = key.split("_")
                date_str = f"{parts[0]}-{parts[1]}-01"
                date_to_keys[date_str].append((0.0, key))
                continue

            hrs = pd.to_numeric(df["Hours_elapsed"], errors="coerce").dropna()
            if hrs.empty:
                parts = key.split("_")
                date_str = f"{parts[0]}-{parts[1]}-01"
                date_to_keys[date_str].append((0.0, key))
                continue

            med_hrs  = float(hrs.median())
            date_str = _hrs_to_date(med_hrs)
            date_to_keys[date_str].append((med_hrs, key))

    day_index = {}
    for date_str in sorted(date_to_keys.keys()):
        sorted_pairs = sorted(date_to_keys[date_str], key=lambda x: x[0])
        day_index[date_str] = [k for _, k in sorted_pairs]

    return day_index


def day_index_to_df(day_index: dict) -> pd.DataFrame:
    """Convert day_index to a human-readable DataFrame (CSV-ready)."""
    rows = []
    for date_str, keys in sorted(day_index.items()):
        dt  = pd.to_datetime(date_str)
        row = {"date_str": date_str, "year": dt.year, "month": dt.month,
               "day": dt.day, "n_hours": len(keys)}
        for i, k in enumerate(keys):
            row[f"key_h{i}"] = k
        for i in range(len(keys), 8):     # NaN for missing hours
            row[f"key_h{i}"] = np.nan
        rows.append(row)
    cols = (["date_str", "year", "month", "day", "n_hours"] +
            [f"key_h{i}" for i in range(8)])
    return pd.DataFrame(rows, columns=cols)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(years, months):
    print(f"\n{'='*60}")
    print(f"  step4_merge_apr_sep_multiyear")
    print(f"  years  : {years}")
    print(f"  months : {months}")
    print(f"  base   : {BASE_PATH}")
    print(f"{'='*60}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_year_dicts = {}
    all_mm_tables  = {}

    # Process each year independently → separate pkl
    for year in years:
        print(f"\n--- Year {year} ---")
        year_dict, mm_table = build_and_save_year(year, months, BASE_PATH, OUT_DIR)
        if year_dict:
            all_year_dicts[year] = year_dict
            all_mm_tables.update(mm_table)

    if not all_year_dicts:
        print("\n[Error] No data loaded at all.")
        return

    # Monthly means CSV
    mm_rows = [{"year": y, "month": m, "monthly_mean": v}
               for (y, m), v in sorted(all_mm_tables.items())]
    mm_df  = pd.DataFrame(mm_rows)
    mm_csv = OUT_DIR / "monthly_means_apr_sep_2022_2025.csv"
    mm_df.to_csv(mm_csv, index=False)
    print(f"\n[Saved] {mm_csv.name}")
    print(mm_df.to_string(index=False))

    # Day index CSV
    print("\nBuilding day index from Hours_elapsed ...")
    day_index  = build_day_index(all_year_dicts)
    idx_df     = day_index_to_df(day_index)
    idx_csv    = OUT_DIR / "day_index_apr_sep_2022_2025.csv"
    idx_df.to_csv(idx_csv, index=False)

    n_total = len(idx_df)
    n_full  = (idx_df["n_hours"] == 8).sum()
    n_part  = n_total - n_full
    print(f"[Saved] {idx_csv.name}  ({n_total} days, full={n_full}, partial={n_part})")
    if n_part > 0:
        print("  Partial days (n_hours ≠ 8):")
        print(idx_df[idx_df["n_hours"] != 8][["date_str", "n_hours"]].to_string(index=False))

    print("\n[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",  type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    args = parser.parse_args()
    main(args.years, args.months)
