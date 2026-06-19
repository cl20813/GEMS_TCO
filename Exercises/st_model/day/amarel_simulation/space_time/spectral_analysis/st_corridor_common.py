"""Shared helpers for real July space-time corridor spectral diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
SPACE_TIME_DIR = HERE.parent

AMAREL_SRC = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SRC = AMAREL_SRC if AMAREL_SRC.exists() else LOCAL_SRC
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed


DTYPE = torch.double
ROUND_DECIMALS = 6
P_LABELS = [
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
]

DEFAULT_REAL_INIT_PHYSICAL = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.0,
}


def default_data_root() -> Path:
    amarel = Path(config.amarel_data_load_path)
    if amarel.exists():
        return amarel
    return Path(config.mac_data_load_path)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def save_rows(csv_path: Path, rows: list[dict[str, Any]] | pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(decimals)
    df.to_csv(csv_path, index=False, float_format=f"%.{decimals}f")
    return df


def parse_tokens(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def parse_int_tokens(values: Iterable[str]) -> list[int]:
    return [int(v) for v in parse_tokens(values)]


def parse_day_idxs(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(31))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end))
    return [int(p) for p in parts]


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.cuda_fallback == "error" or args.require_cuda:
        raise RuntimeError("CUDA is not available and CUDA fallback is disabled.")
    return torch.device("cpu")


def physical_to_log_phi(params: dict[str, float], nugget_mode: str) -> list[float]:
    sigmasq = float(params["sigmasq"])
    range_lat = float(params["range_lat"])
    range_lon = float(params["range_lon"])
    range_time = float(params["range_time"])
    phi2 = 1.0 / range_lon
    phi1 = sigmasq * phi2
    phi3 = (range_lon / range_lat) ** 2
    phi4 = (range_lon / range_time) ** 2
    raw = [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
    ]
    if nugget_mode == "estimated":
        raw.append(np.log(max(float(params["nugget"]), 1e-12)))
    return raw


def make_params_list(init_physical: dict[str, float], dtype: torch.dtype, device: torch.device, nugget_mode: str):
    return [
        Parameter(torch.tensor([val], dtype=dtype, device=device))
        for val in physical_to_log_phi(init_physical, nugget_mode=nugget_mode)
    ]


def backmap_params(out_params: list[float], nugget_mode: str) -> dict[str, float]:
    raw = [float(x) for x in out_params]
    phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(phi1 / phi2),
        "range_lat": float(range_lon / np.sqrt(phi3)),
        "range_lon": float(range_lon),
        "range_time": float(range_lon / np.sqrt(phi4)),
        "advec_lat": float(raw[4]),
        "advec_lon": float(raw[5]),
        "nugget": float(np.exp(raw[6])) if nugget_mode == "estimated" else 0.0,
    }


def count_valid(source_map: dict[str, torch.Tensor]) -> tuple[int, int, dict[str, int]]:
    n_valid = 0
    n_total = 0
    valid_by_t: dict[str, int] = {}
    for key, tensor in source_map.items():
        count = int((~torch.isnan(tensor[:, 2])).sum().item())
        n_valid += count
        n_total += int(tensor.shape[0])
        valid_by_t[str(key)] = count
    return n_valid, n_total, valid_by_t


def assert_grid_order_consistent(df_map: dict[str, pd.DataFrame], keys: list[str], base_coords: np.ndarray) -> None:
    for key in keys:
        coords = df_map[key][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coords.shape != base_coords.shape or not np.allclose(coords, base_coords, equal_nan=True):
            raise RuntimeError(f"Regular grid coordinate order differs at {key}.")


def load_real_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_root = args.data_root or default_data_root()
    data_loader = load_data_dynamic_processed(str(data_root))
    years = parse_int_tokens(args.real_years)
    days = parse_day_idxs(args.days)
    lat_lon_resolution = [int(x) for x in parse_pair(args.space, int)]
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    assets: list[dict[str, Any]] = []

    for year in years:
        print("\n" + "=" * 88, flush=True)
        print(f"Loading real July data for {year}-{args.month:02d}", flush=True)
        print("=" * 88, flush=True)
        df_map, _, _, monthly_mean = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=1,
            years_=[str(year)],
            months_=[int(args.month)],
            lat_range=lat_range,
            lon_range=lon_range,
            is_whittle=True,
        )
        key_idx = sorted(df_map)
        if not key_idx:
            raise RuntimeError(f"No data loaded for {year}-{args.month:02d} from {data_root}")
        base_grid_coords_np = df_map[key_idx[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        print("n hourly slots:", len(key_idx), "grid:", base_grid_coords_np.shape, "monthly_mean:", monthly_mean, flush=True)

        for day_idx in days:
            start, end = int(day_idx) * 8, (int(day_idx) + 1) * 8
            selected_keys = key_idx[start:end]
            day = f"{year}-{args.month:02d}-{int(day_idx) + 1:02d}"
            if len(selected_keys) != 8:
                print(f"Skipping {day}: expected 8 hourly slots, found {len(selected_keys)}", flush=True)
                continue
            assert_grid_order_consistent(df_map, selected_keys, base_grid_coords_np)
            source_map, _ = data_loader.load_working_data(
                df_map,
                monthly_mean,
                [start, end],
                ord_mm=None,
                dtype=DTYPE,
                keep_ori=bool(args.keep_exact_loc),
            )
            assets.append(
                {
                    "dataset": "real",
                    "year": int(year),
                    "month": int(args.month),
                    "day_idx": int(day_idx),
                    "day": day,
                    "day_keys": list(selected_keys),
                    "source_map": {k: v.contiguous() for k, v in source_map.items()},
                    "grid_coords_np": base_grid_coords_np.copy(),
                    "monthly_mean": float(monthly_mean),
                }
            )
    return assets
