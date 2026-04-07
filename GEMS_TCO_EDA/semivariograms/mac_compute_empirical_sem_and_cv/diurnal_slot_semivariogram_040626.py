"""
diurnal_slot_semivariogram_040626.py

Diurnal cycle check via slot-wise empirical semivariogram.

Goal: Do short-lag spatial variability structures differ by time-of-day (h0-h7)?
      → Pool pairs across all days within each (year, month) for each slot.

Approach:
  - For each slot h in {0,...,7}:
      Pool all observation pairs (i,j) across all days for that slot
      γ_h(lag) = 0.5 * mean over pooled days of (Z_i(d,h) - Z_j(d,h))^2
  - Only first N_SHORT_LAGS lags (short-range focus)
  - Runs locally: N×N pair precompute once per month (~2.6 GB RAM for N=18126)

Output:
  diurnal_slot_sem_{tag}.pkl
    dict: {
      'lags_lat': [(dlat, dlon), ...],
      'lags_lon': [(dlat, dlon), ...],
      'lat': {slot: [gamma_lag0, gamma_lag1, ...], ...},   # 8 slots × N_SHORT_LAGS
      'lon': {slot: [gamma_lag0, gamma_lag1, ...], ...},
      'meta': {'years': [...], 'months': [...], 'n_days_per_slot': {slot: n}}
    }

Usage:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA/semivariograms/mac_compute_empirical_sem_and_cv
  python diurnal_slot_semivariogram_040626.py --years "2022,2023,2024,2025" --months "7"
  python diurnal_slot_semivariogram_040626.py --years "2022,2023,2024,2025" --months "4,5,6,7,8,9"
"""

import sys
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import typer

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# ── Config ────────────────────────────────────────────────────────────────────
N_SLOTS       = 8
N_SHORT_LAGS  = 3        # Only first 3 spatial lags in each direction
LAT_RANGE     = [-3.0, 2.0]
LON_RANGE     = [121.0, 131.0]
TOLERANCE     = 0.015    # degrees — pair matching tolerance

# Short-lag delta grids (lat direction: vary lat, lon=0)
SHORT_LAGS_LAT = [(round(v, 3), 0.0) for v in [0.044, 0.088, 0.132]]   # 1,2,3 grid steps
SHORT_LAGS_LON = [(0.0, round(v, 3)) for v in [0.063, 0.126, 0.189]]   # 1,2,3 grid steps


def precompute_pairs(coords: np.ndarray, deltas: list, tolerance: float, device: str):
    """
    Precompute (row_idx, col_idx) pairs for each delta lag.
    coords: (N, 2) array of [lat, lon]
    Returns list of (row_idx, col_idx) tensors, one per delta.
    """
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    lat_diffs = coords_t[:, None, 0] - coords_t[None, :, 0]  # (N, N)
    lon_diffs = coords_t[:, None, 1] - coords_t[None, :, 1]  # (N, N)

    cache = []
    for (dlat, dlon) in deltas:
        mask = (torch.abs(lat_diffs - dlat) <= tolerance) & \
               (torch.abs(lon_diffs - dlon) <= tolerance)
        pairs = torch.nonzero(mask, as_tuple=True)
        cache.append(pairs)
        print(f"  delta=({dlat:.3f},{dlon:.3f}): {len(pairs[0])} pairs")

    del lat_diffs, lon_diffs
    if device == 'cuda':
        torch.cuda.empty_cache()
    return cache


def compute_slot_semivariogram(
    hourly_maps: dict,
    pair_cache: list,
    deltas: list,
    device: str,
) -> dict:
    """
    Compute pooled-across-days semivariogram for each slot.

    hourly_maps: {key: tensor (N,11)} — all hours across all days, sorted by key
    pair_cache: list of (row_idx, col_idx) per lag
    Returns: {slot: [gamma_lag0, gamma_lag1, ...]}
    """
    key_list = sorted(hourly_maps.keys())
    n_days   = len(key_list) // N_SLOTS

    # slot -> list of per-day semivariance arrays
    slot_sum   = {h: np.zeros(len(deltas)) for h in range(N_SLOTS)}
    slot_count = {h: np.zeros(len(deltas), dtype=int) for h in range(N_SLOTS)}

    for day_idx in range(n_days):
        for slot in range(N_SLOTS):
            list_idx = day_idx * N_SLOTS + slot
            if list_idx >= len(key_list):
                break

            key  = key_list[list_idx]
            data = hourly_maps[key]                          # (N, 4)
            vals = torch.tensor(data[:, 2].astype(np.float32), device=device)

            # Drop NaN before centering
            finite_mask = torch.isfinite(vals)
            if finite_mask.sum() < 10:
                continue
            vals = torch.where(finite_mask, vals, torch.zeros_like(vals))
            vals = vals - torch.nanmean(vals[finite_mask])   # center on finite mean

            for j, (idx_row, idx_col) in enumerate(pair_cache):
                if len(idx_row) == 0:
                    continue
                # Only use pairs where both endpoints are finite
                valid = finite_mask[idx_row] & finite_mask[idx_col]
                if valid.sum() < 2:
                    continue
                diffs = vals[idx_col[valid]] - vals[idx_row[valid]]
                sv    = 0.5 * torch.mean(diffs ** 2).item()
                if np.isfinite(sv):
                    slot_sum[slot][j]   += sv
                    slot_count[slot][j] += 1

    # Average across days
    result = {}
    for slot in range(N_SLOTS):
        with np.errstate(invalid='ignore'):
            result[slot] = [
                float(slot_sum[slot][j] / slot_count[slot][j])
                if slot_count[slot][j] > 0 else float('nan')
                for j in range(len(deltas))
            ]
    return result, {slot: int(slot_count[slot][0]) for slot in range(N_SLOTS)}


@app.command()
def cli(
    years:  List[str] = typer.Option(['2022,2023,2024,2025'], help="Comma-separated years"),
    months: List[str] = typer.Option(['7'],                   help="Comma-separated months"),
    tag:    str       = typer.Option('',                      help="Extra tag for output filename"),
) -> None:

    years_list  = [y.strip() for y in years[0].split(',')]
    months_list = [int(m.strip()) for m in months[0].split(',')]
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Years : {years_list}")
    print(f"Months: {months_list}")

    data_loader = load_data_dynamic_processed(config.mac_data_load_path)

    # Accumulators across all (year, month) combinations
    lat_slot_sum   = {h: np.zeros(N_SHORT_LAGS) for h in range(N_SLOTS)}
    lat_slot_count = {h: np.zeros(N_SHORT_LAGS, dtype=int) for h in range(N_SLOTS)}
    lon_slot_sum   = {h: np.zeros(N_SHORT_LAGS) for h in range(N_SLOTS)}
    lon_slot_count = {h: np.zeros(N_SHORT_LAGS, dtype=int) for h in range(N_SLOTS)}

    for yr in years_list:
        for mo in months_list:
            print(f"\n{'='*55}")
            print(f"  Year={yr}  Month={mo:02d}")
            print(f"{'='*55}")

            df_map, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
                lat_lon_resolution=[1, 1],
                years_=[yr],
                months_=[mo],
                lat_range=LAT_RANGE,
                lon_range=LON_RANGE,
                is_whittle=True,
            )
            if not df_map:
                print(f"  No data for {yr}-{mo:02d}. Skipping.")
                continue

            # Convert DataFrames to numpy arrays, keyed by sorted key
            hourly_maps = {}
            for key in sorted(df_map.keys()):
                df = df_map[key]
                # Extract [Lat, Lon, O3_centered, Hours_elapsed] columns
                arr = df[['Latitude', 'Longitude', 'ColumnAmountO3', 'Hours_elapsed']].to_numpy(dtype=np.float32)
                hourly_maps[key] = arr

            if not hourly_maps:
                continue

            # Precompute spatial pairs (only once per month — coords same for all slots)
            first_key  = sorted(hourly_maps.keys())[0]
            coords     = hourly_maps[first_key][:, :2]   # (N, 2)
            N          = coords.shape[0]
            print(f"  Grid points N={N}")

            print("  Precomputing lat-direction pairs...")
            lat_cache = precompute_pairs(coords, SHORT_LAGS_LAT, TOLERANCE, device)
            print("  Precomputing lon-direction pairs...")
            lon_cache = precompute_pairs(coords, SHORT_LAGS_LON, TOLERANCE, device)

            # Compute slot-wise semivariograms for this (year, month)
            print("  Computing lat-direction slot semivariograms...")
            lat_result, lat_n = compute_slot_semivariogram(
                hourly_maps, lat_cache, SHORT_LAGS_LAT, device)

            print("  Computing lon-direction slot semivariograms...")
            lon_result, lon_n = compute_slot_semivariogram(
                hourly_maps, lon_cache, SHORT_LAGS_LON, device)

            # Accumulate
            for slot in range(N_SLOTS):
                for j in range(N_SHORT_LAGS):
                    if np.isfinite(lat_result[slot][j]):
                        lat_slot_sum[slot][j]   += lat_result[slot][j] * lat_n[slot]
                        lat_slot_count[slot][j] += lat_n[slot]
                    if np.isfinite(lon_result[slot][j]):
                        lon_slot_sum[slot][j]   += lon_result[slot][j] * lon_n[slot]
                        lon_slot_count[slot][j] += lon_n[slot]

            # Print progress for this (year, month)
            print(f"\n  Slot γ̂ lat-lag1  | {'  '.join(f'h{h}' for h in range(N_SLOTS))}")
            row = '  '.join(f'{lat_result[h][0]:.3f}' for h in range(N_SLOTS))
            print(f"  γ(lag1):           {row}")

    # ── Final pooled result ───────────────────────────────────────────────────
    lat_final = {}
    lon_final = {}
    for slot in range(N_SLOTS):
        lat_final[slot] = [
            float(lat_slot_sum[slot][j] / lat_slot_count[slot][j])
            if lat_slot_count[slot][j] > 0 else float('nan')
            for j in range(N_SHORT_LAGS)
        ]
        lon_final[slot] = [
            float(lon_slot_sum[slot][j] / lon_slot_count[slot][j])
            if lon_slot_count[slot][j] > 0 else float('nan')
            for j in range(N_SHORT_LAGS)
        ]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL POOLED SLOT SEMIVARIOGRAMS")
    print(f"{'='*60}")
    for direction, final, lags in [('LAT', lat_final, SHORT_LAGS_LAT),
                                    ('LON', lon_final, SHORT_LAGS_LON)]:
        print(f"\nDirection: {direction}")
        header = f"{'slot':>5} | " + " | ".join(f"lag{j+1}(Δ={lags[j][0 if direction=='LAT' else 1]:.3f}°)" for j in range(N_SHORT_LAGS))
        print(header)
        print("-" * len(header))
        for slot in range(N_SLOTS):
            row = " | ".join(f"{final[slot][j]:10.4f}" for j in range(N_SHORT_LAGS))
            print(f"  h{slot}   | {row}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        'lags_lat': SHORT_LAGS_LAT,
        'lags_lon': SHORT_LAGS_LON,
        'lat':      lat_final,
        'lon':      lon_final,
        'meta': {
            'years':  years_list,
            'months': months_list,
            'n_days_per_slot_lat': {h: int(lat_slot_count[h][0]) for h in range(N_SLOTS)},
            'n_days_per_slot_lon': {h: int(lon_slot_count[h][0]) for h in range(N_SLOTS)},
        }
    }

    mo_str   = '_'.join(f'm{m:02d}' for m in months_list)
    yr_str   = '_'.join(y[2:] for y in years_list)
    tag_str  = f'_{tag}' if tag else ''
    filename = f"diurnal_slot_sem_{yr_str}_{mo_str}{tag_str}.pkl"

    out_dir  = Path(config.mac_save_computed_semi_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    app()
