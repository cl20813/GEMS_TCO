#!/usr/bin/env python3
"""Real July 2023-2025 ST corridor Vecchia Matérn-vs-Cauchy nugget-zero tradeoff test v2.

This is the focused comparison requested after the pure-space spectral checks:

  - Matérn with fixed smooth=0.3 and nugget fixed at zero;
  - year-specific generalized Cauchy candidates with fixed alpha/beta and nugget fixed at zero;
  - the same 4x4 corridor lag-643 Vecchia geometry;
  - max-min block-prefix fits at 100, 200, 400, 600, 800, and all blocks;
  - real July 2023, 2024, and 2025 data by default.

The output includes final Vecchia objective values from each fitted model, both
raw and normalized by the number of valid observations. Parameter-tracking plot
legends include the mean final negative likelihood loss by model/year.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
SPACE_TIME_DIR = HERE.parent
for path in [HERE, SPACE_TIME_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

AMAREL_SRC = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SRC = AMAREL_SRC if AMAREL_SRC.exists() else LOCAL_SRC
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME as VECCHIA_SPEC_NAME,
)
from GEMS_TCO.vecchia_st_generalized_cauchy import (
    RealDataCorridorWidth4x4Lag643GeneralizedCauchyFit,
    RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit,
)
from GEMS_TCO.vecchia_st_spline import (
    RealDataCorridorWidth4x4Lag643NoNuggetSplineFit,
    RealDataCorridorWidth4x4Lag643SplineFit,
)


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

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "matern_s03": {
        "family": "matern",
        "smooth": 0.3,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.3",
    },
    "gc_a075_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 1.0,
        "label": "GC a=0.75 b=1",
    },
    "gc_a08_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.8,
        "gc_beta": 1.0,
        "label": "GC a=0.8 b=1",
    },
    "gc_a075_b05": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 0.5,
        "label": "GC a=0.75 b=0.5",
    },
    "gc_a08_b05": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.8,
        "gc_beta": 0.5,
        "label": "GC a=0.8 b=0.5",
    },
}

YEAR_VARIANT_DEFAULTS: dict[int, list[str]] = {
    2023: ["matern_s03", "gc_a075_b05", "gc_a075_b1", "gc_a08_b1", "gc_a08_b05"],
    2024: ["matern_s03", "gc_a08_b05", "gc_a08_b1"],
    2025: ["matern_s03", "gc_a075_b1", "gc_a075_b05", "gc_a08_b1", "gc_a08_b05"],
}

DEFAULT_MODEL_VARIANTS = list(dict.fromkeys(v for vals in YEAR_VARIANT_DEFAULTS.values() for v in vals))

ALL_FITS_CSV = "real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_all_fits.csv"
PARAM_SUMMARY_CSV = "real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_monthly_param_summary.csv"
LOSS_SUMMARY_CSV = "real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_monthly_loss_summary.csv"
MISSING_CSV = "real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_missing.csv"
JSONL_NAME = "real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_all_fits.jsonl"


def default_data_root() -> Path:
    amarel = Path(config.amarel_data_load_path)
    if amarel.exists():
        return amarel
    return Path(config.mac_data_load_path)


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/exercise_output/summer/real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/real_july2023_2025_corridor_lag643_matern_cauchy_tradeoff_nugget0_prefix_v2_061526")


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


def parse_block_prefix_tokens(values: list[str]) -> list[int]:
    out: list[int] = []
    for token in parse_tokens(values):
        low = token.lower()
        out.append(-1 if low in {"all", "full"} else int(token))
    return out


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


def grid_row_col_maps(grid_coords_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    coords = np.asarray(grid_coords_np, dtype=np.float64)
    lat_key = np.round(coords[:, 0], ROUND_DECIMALS)
    lon_key = np.round(coords[:, 1], ROUND_DECIMALS)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_to_row = {v: i for i, v in enumerate(lats)}
    lon_to_col = {v: i for i, v in enumerate(lons)}
    rows = np.asarray([lat_to_row[v] for v in lat_key], dtype=np.int64)
    cols = np.asarray([lon_to_col[v] for v in lon_key], dtype=np.int64)
    return rows, cols, int(len(lats)), int(len(lons))


def blockmaxmin_indices(
    grid_coords_np: np.ndarray,
    n_blocks_requested: int,
    block_shape: tuple[int, int] = BLOCK_SHAPE,
) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(grid_coords_np, dtype=np.float64)
    rows, cols, n_lat, n_lon = grid_row_col_maps(coords)
    by_block: dict[tuple[int, int], list[int]] = {}
    block_y, block_x = int(block_shape[0]), int(block_shape[1])
    for idx, (row, col) in enumerate(zip(rows, cols)):
        by_block.setdefault((int(row) // block_y, int(col) // block_x), []).append(int(idx))

    block_keys = sorted(by_block)
    centers = np.asarray([coords[np.asarray(by_block[key], dtype=np.int64)].mean(axis=0) for key in block_keys])
    if centers.size == 0:
        raise ValueError("No 4x4 blocks could be built from grid_coords_np.")
    lon_lat = np.column_stack([centers[:, 1], centers[:, 0]])
    order = np.asarray(orderings.maxmin_cpp(lon_lat), dtype=np.int64)
    if order.size and order.min() == 1 and order.max() == len(block_keys):
        order = order - 1
    if order.size != len(block_keys):
        raise RuntimeError(f"max-min order length {order.size} != block count {len(block_keys)}")

    requested = int(n_blocks_requested)
    n_use = len(block_keys) if requested <= 0 else min(requested, len(block_keys))
    selected_blocks = [block_keys[int(i)] for i in order[:n_use]]
    selected = np.concatenate([np.asarray(by_block[key], dtype=np.int64) for key in selected_blocks])
    selected = np.asarray(sorted(set(int(i) for i in selected)), dtype=np.int64)
    meta = {
        "n_lat_grid": int(n_lat),
        "n_lon_grid": int(n_lon),
        "block_shape": f"{block_y}x{block_x}",
        "n_blocks_full": int(len(block_keys)),
        "n_blocks_requested": int(requested),
        "n_blocks_used": int(n_use),
        "n_grid_selected": int(selected.size),
    }
    return selected, meta


def asset_for_block_prefix(asset: dict[str, Any], n_blocks_requested: int) -> dict[str, Any]:
    grid = np.asarray(asset["grid_coords_np"], dtype=np.float64)
    idx, meta = blockmaxmin_indices(grid, int(n_blocks_requested), BLOCK_SHAPE)
    source_map = {k: v[idx].contiguous() for k, v in asset["source_map"].items()}
    n_valid, n_total, valid_by_t = count_valid(source_map)
    return {
        **asset,
        "source_map": source_map,
        "grid_coords_np": grid[idx],
        "n_grid_full": int(grid.shape[0]),
        "n_block_prefix_requested": int(n_blocks_requested),
        "n_block_prefix_used": int(meta["n_blocks_used"]),
        "block_prefix_label": "all" if int(n_blocks_requested) <= 0 else f"B{int(meta['n_blocks_used'])}",
        "n_grid_block_prefix": int(idx.size),
        "n_valid_block_prefix": int(n_valid),
        "n_total_block_prefix": int(n_total),
        "valid_by_t_block_prefix": valid_by_t,
        **meta,
    }


def variant_spec(name: str) -> dict[str, Any]:
    if name not in VARIANT_SPECS:
        raise ValueError(f"Unknown model variant {name!r}. Known: {sorted(VARIANT_SPECS)}")
    return {**VARIANT_SPECS[name], "model_variant": name}


def variants_for_year(year: int, requested_variants: list[str]) -> list[str]:
    allowed = set(YEAR_VARIANT_DEFAULTS.get(int(year), requested_variants))
    return [name for name in requested_variants if name in allowed]


def build_model(spec: dict[str, Any], source_map: dict[str, torch.Tensor], grid_coords_np: np.ndarray, args: argparse.Namespace):
    family = str(spec["family"])
    nugget_mode = str(args.nugget_mode)
    if family == "matern":
        klass = RealDataCorridorWidth4x4Lag643SplineFit if nugget_mode == "estimated" else RealDataCorridorWidth4x4Lag643NoNuggetSplineFit
        return klass(
            smooth=float(spec["smooth"]),
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=float(args.real_reference_advec_lon_abs),
            daily_stride=int(args.daily_stride),
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
    if family == "cauchy":
        klass = (
            RealDataCorridorWidth4x4Lag643GeneralizedCauchyFit
            if nugget_mode == "estimated"
            else RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit
        )
        return klass(
            gc_alpha=float(spec["gc_alpha"]),
            gc_beta=float(spec["gc_beta"]),
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=float(args.real_reference_advec_lon_abs),
            daily_stride=int(args.daily_stride),
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
        )
    raise ValueError(f"Unknown family {family!r}")


def fit_one(
    asset: dict[str, Any],
    model_variant: str,
    block_prefix: int,
    device: torch.device,
    args: argparse.Namespace,
    fit_id: int,
) -> dict[str, Any]:
    spec = variant_spec(model_variant)
    sub = asset_for_block_prefix(asset, int(block_prefix))
    source_map = {
        k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for k, v in sub["source_map"].items()
    }
    grid_coords_np = np.asarray(sub["grid_coords_np"], dtype=np.float64)
    n_valid, n_total, valid_by_t = count_valid(source_map)
    params_list = make_params_list(DEFAULT_REAL_INIT_PHYSICAL, dtype=DTYPE, device=device, nugget_mode=args.nugget_mode)
    model = build_model(spec, source_map, grid_coords_np, args)

    t0 = time.time()
    model.precompute_conditioning_sets()
    precompute_s = time.time() - t0

    optimizer = model.set_optimizer(
        params_list,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )

    t1 = time.time()
    if args.suppress_fit_prints:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            out, steps_ran = model.fit_vecc_lbfgs(
                params_list,
                optimizer,
                max_steps=int(args.lbfgs_steps),
                grad_tol=float(args.grad_tol),
            )
    else:
        out, steps_ran = model.fit_vecc_lbfgs(
            params_list,
            optimizer,
            max_steps=int(args.lbfgs_steps),
            grad_tol=float(args.grad_tol),
        )
    fit_s = time.time() - t1

    est = backmap_params(out, nugget_mode=str(args.nugget_mode))
    loss = float(out[-1])
    nonfinite = [k for k in P_LABELS if not np.isfinite(float(est[k]))]
    if nonfinite or not np.isfinite(loss):
        raise RuntimeError(f"Non-finite fit result: loss={loss}, nonfinite_params={nonfinite}")
    cluster_summary = model.cluster_summary()

    row = {
        "fit_id": int(fit_id),
        "status": "ok",
        "error": "",
        "dataset": "real",
        "year": int(sub["year"]),
        "month": int(sub["month"]),
        "day_idx": int(sub["day_idx"]),
        "day": str(sub["day"]),
        "model_variant": str(model_variant),
        "model_family": str(spec["family"]),
        "model_label": str(spec["label"]),
        "smooth": float(spec["smooth"]) if pd.notna(spec["smooth"]) else np.nan,
        "gc_alpha": float(spec["gc_alpha"]) if pd.notna(spec["gc_alpha"]) else np.nan,
        "gc_beta": float(spec["gc_beta"]) if pd.notna(spec["gc_beta"]) else np.nan,
        "nugget_mode": str(args.nugget_mode),
        "block_prefix_requested": int(block_prefix),
        "block_prefix_used": int(sub["n_block_prefix_used"]),
        "block_prefix_label": str(sub["block_prefix_label"]),
        "block_prefix_order": {100: 0, 200: 1, 400: 2, 600: 3, 800: 4, -1: 5}.get(int(block_prefix), int(block_prefix)),
        "n_grid_full": int(sub["n_grid_full"]),
        "n_blocks_full": int(sub["n_blocks_full"]),
        "n_grid_block_prefix": int(sub["n_grid_block_prefix"]),
        "n_rows_total": int(n_total),
        "n_valid_o3": int(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else np.nan,
        "valid_by_t": json.dumps(valid_by_t, separators=(",", ":")),
        "monthly_mean": float(sub["monthly_mean"]),
        "first_slot": sub["day_keys"][0] if sub.get("day_keys") else "",
        "last_slot": sub["day_keys"][-1] if sub.get("day_keys") else "",
        "spec_name": VECCHIA_SPEC_NAME,
        "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
        "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
        "reference_advec_lon_abs": float(args.real_reference_advec_lon_abs),
        "loss": loss,
        "loss_per_valid": float(loss / n_valid) if n_valid else np.nan,
        "steps_raw": int(steps_ran),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "total_s": float(precompute_s + fit_s),
        **{f"est_{k}": float(est[k]) for k in P_LABELS},
        **cluster_summary,
    }
    del model, params_list, optimizer, source_map
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def finite_estimate_mask(df: pd.DataFrame) -> pd.Series:
    est_cols = [f"est_{p}" for p in P_LABELS if f"est_{p}" in df.columns]
    if not est_cols:
        return pd.Series(False, index=df.index)
    vals = df[est_cols].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(vals.to_numpy(dtype=float)).all(axis=1), index=df.index)


def make_param_summary(ok: pd.DataFrame) -> pd.DataFrame:
    if ok.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "dataset",
        "year",
        "month",
        "model_variant",
        "model_family",
        "model_label",
        "smooth",
        "gc_alpha",
        "gc_beta",
        "nugget_mode",
        "block_prefix_requested",
        "block_prefix_used",
        "block_prefix_label",
        "block_prefix_order",
    ]
    for param in P_LABELS:
        col = f"est_{param}"
        if col not in ok.columns:
            continue
        for keys, sub in ok.groupby(group_cols, dropna=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "parameter": param,
                    "n_days": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "p10": float(np.quantile(vals, 0.10)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_loss_summary(ok: pd.DataFrame) -> pd.DataFrame:
    if ok.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "dataset",
        "year",
        "month",
        "model_variant",
        "model_family",
        "model_label",
        "smooth",
        "gc_alpha",
        "gc_beta",
        "nugget_mode",
        "block_prefix_requested",
        "block_prefix_used",
        "block_prefix_label",
        "block_prefix_order",
    ]
    for keys, sub in ok.groupby(group_cols, dropna=False):
        loss = pd.to_numeric(sub["loss"], errors="coerce").dropna().to_numpy(dtype=float)
        loss_per = pd.to_numeric(sub["loss_per_valid"], errors="coerce").dropna().to_numpy(dtype=float)
        if loss.size == 0:
            continue
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_days": int(loss.size),
                "loss_mean": float(np.mean(loss)),
                "loss_median": float(np.median(loss)),
                "loss_sd": float(np.std(loss, ddof=1)) if loss.size > 1 else 0.0,
                "loss_per_valid_mean": float(np.mean(loss_per)) if loss_per.size else np.nan,
                "loss_per_valid_median": float(np.median(loss_per)) if loss_per.size else np.nan,
                "loss_per_valid_sd": float(np.std(loss_per, ddof=1)) if loss_per.size > 1 else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def model_style(model_variant: str):
    styles = {
        "matern_s03": ("tab:blue", "o", "-"),
        "gc_a075_b1": ("tab:green", "s", "-"),
        "gc_a08_b1": ("tab:orange", "P", "-"),
        "gc_a075_b05": ("tab:brown", "v", "--"),
        "gc_a08_b05": ("tab:purple", "^", "--"),
    }
    return styles.get(str(model_variant), ("0.25", "o", "-"))


def year_model_loss_lookup(loss_summary: pd.DataFrame | None) -> dict[tuple[int, str], float]:
    if loss_summary is None or loss_summary.empty:
        return {}
    lookup: dict[tuple[int, str], float] = {}
    for (year, model_variant), sub in loss_summary.groupby(["year", "model_variant"], dropna=False):
        vals = pd.to_numeric(sub["loss_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            lookup[(int(year), str(model_variant))] = float(np.mean(vals))
    return lookup


def label_with_loss(base_label: str, loss_mean: float | None) -> str:
    if loss_mean is None or not np.isfinite(float(loss_mean)):
        return base_label
    return f"{base_label} loss={float(loss_mean):.5f}"


def ordered_prefixes(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    if df.empty:
        return [], []
    order_df = (
        df[["block_prefix_requested", "block_prefix_label", "block_prefix_order"]]
        .drop_duplicates()
        .sort_values(["block_prefix_order", "block_prefix_requested"])
    )
    return (
        [int(x) for x in order_df["block_prefix_requested"].tolist()],
        [str(x) for x in order_df["block_prefix_label"].tolist()],
    )


def plot_param_summary(
    summary: pd.DataFrame,
    out_dir: Path,
    monthly_out_dir: Path | None = None,
    loss_summary: pd.DataFrame | None = None,
) -> None:
    if summary.empty:
        return
    loss_lookup = year_model_loss_lookup(loss_summary)
    plot_dirs = [out_dir / "monthly_average_plots"]
    if monthly_out_dir is not None:
        plot_dirs.append(monthly_out_dir)
    for plot_dir in plot_dirs:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(pd.to_numeric(summary["year"], errors="coerce").dropna().astype(int).unique()):
        sub_year = summary[summary["year"].astype(int) == int(year)].copy()
        prefixes, labels = ordered_prefixes(sub_year)
        if not prefixes:
            continue
        x_pos = np.arange(len(prefixes))
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        axes_flat = axes.ravel()
        for ax, param in zip(axes_flat, P_LABELS):
            sub_param = sub_year[sub_year["parameter"] == param].copy()
            for model_variant, sub_model in sub_param.groupby("model_variant", dropna=False):
                sub_model = sub_model.set_index("block_prefix_requested").reindex(prefixes)
                y = pd.to_numeric(sub_model["median"], errors="coerce").to_numpy(dtype=float)
                p10 = pd.to_numeric(sub_model["p10"], errors="coerce").to_numpy(dtype=float)
                p90 = pd.to_numeric(sub_model["p90"], errors="coerce").to_numpy(dtype=float)
                color, marker, ls = model_style(str(model_variant))
                base_label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
                label = label_with_loss(base_label, loss_lookup.get((int(year), str(model_variant))))
                finite = np.isfinite(y)
                ax.plot(x_pos[finite], y[finite], marker=marker, linestyle=ls, color=color, linewidth=1.7, label=label)
                if finite.sum() >= 2:
                    ax.fill_between(x_pos[finite], p10[finite], p90[finite], color=color, alpha=0.10, linewidth=0)
            ax.set_title(param)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.grid(alpha=0.25)
        for ax in axes_flat[len(P_LABELS) :]:
            ax.axis("off")
        axes_flat[0].legend(fontsize=7)
        fig.suptitle(f"Real July {year}: parameter monthly medians by block-prefix")
        fig.tight_layout()
        for plot_dir in plot_dirs:
            fig.savefig(plot_dir / f"real_{year}_parameter_median_by_blockmaxmin.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        axes_flat = axes.ravel()
        for ax, param in zip(axes_flat, P_LABELS):
            sub_param = sub_year[sub_year["parameter"] == param].copy()
            for model_variant, sub_model in sub_param.groupby("model_variant", dropna=False):
                sub_model = sub_model.set_index("block_prefix_requested").reindex(prefixes)
                y = pd.to_numeric(sub_model["median"], errors="coerce").to_numpy(dtype=float)
                p10 = pd.to_numeric(sub_model["p10"], errors="coerce").to_numpy(dtype=float)
                p90 = pd.to_numeric(sub_model["p90"], errors="coerce").to_numpy(dtype=float)
                color, marker, ls = model_style(str(model_variant))
                base_label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
                label = label_with_loss(base_label, loss_lookup.get((int(year), str(model_variant))))
                finite = np.isfinite(y)
                ax.plot(x_pos[finite], y[finite], marker=marker, linestyle=ls, color=color, linewidth=1.7, label=label)
                if finite.sum() >= 2:
                    ax.fill_between(x_pos[finite], p10[finite], p90[finite], color=color, alpha=0.10, linewidth=0)
            if param in {"sigmasq", "range_lat", "range_lon", "range_time", "nugget"}:
                ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7)
            ax.set_title(param)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.grid(alpha=0.25)
        for ax in axes_flat[len(P_LABELS) :]:
            ax.axis("off")
        axes_flat[0].legend(fontsize=7)
        fig.suptitle(f"Real July {year}: parameter monthly medians by block-prefix, symlog scale")
        fig.tight_layout()
        for plot_dir in plot_dirs:
            fig.savefig(plot_dir / f"real_{year}_parameter_median_by_blockmaxmin_symlog.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_loss_summary(loss_summary: pd.DataFrame, out_dir: Path, monthly_out_dir: Path | None = None) -> None:
    if loss_summary.empty:
        return
    plot_dirs = [out_dir / "monthly_average_plots"]
    if monthly_out_dir is not None:
        plot_dirs.append(monthly_out_dir)
    for plot_dir in plot_dirs:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(pd.to_numeric(loss_summary["year"], errors="coerce").dropna().astype(int).unique()):
        sub_year = loss_summary[loss_summary["year"].astype(int) == int(year)].copy()
        prefixes, labels = ordered_prefixes(sub_year)
        if not prefixes:
            continue
        x_pos = np.arange(len(prefixes))
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        metrics = [
            ("loss_per_valid_mean", "loss / valid obs mean"),
            ("loss_per_valid_median", "loss / valid obs median"),
            ("loss_mean", "raw final objective mean"),
            ("loss_median", "raw final objective median"),
        ]
        for ax, (metric, title) in zip(axes.ravel(), metrics):
            for model_variant, sub_model in sub_year.groupby("model_variant", dropna=False):
                sub_model = sub_model.set_index("block_prefix_requested").reindex(prefixes)
                y = pd.to_numeric(sub_model[metric], errors="coerce").to_numpy(dtype=float)
                color, marker, ls = model_style(str(model_variant))
                label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
                finite = np.isfinite(y)
                ax.plot(x_pos[finite], y[finite], marker=marker, linestyle=ls, color=color, linewidth=1.9, label=label)
            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.grid(alpha=0.25)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f"Real July {year}: final Vecchia objective by block-prefix")
        fig.tight_layout()
        for plot_dir in plot_dirs:
            fig.savefig(plot_dir / f"real_{year}_loss_mean_median_by_blockmaxmin.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    table = loss_summary.pivot_table(
        index="model_label",
        columns="block_prefix_label",
        values="loss_per_valid_median",
        aggfunc="median",
    )
    if not table.empty:
        col_order = (
            loss_summary[["block_prefix_label", "block_prefix_order"]]
            .drop_duplicates()
            .sort_values("block_prefix_order")["block_prefix_label"]
            .tolist()
        )
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
        fig, ax = plt.subplots(figsize=(1.35 * max(4, table.shape[1]), 1.0 * max(4, table.shape[0]) + 1.4))
        arr = table.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(table.shape[1]))
        ax.set_xticklabels(table.columns.astype(str))
        ax.set_yticks(np.arange(table.shape[0]))
        ax.set_yticklabels(table.index.astype(str))
        ax.set_xlabel("4x4 block-center max-min prefix")
        ax.set_title("Median final objective per valid observation, pooled across 2023-2025")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.4g}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for plot_dir in plot_dirs:
            fig.savefig(plot_dir / "real_2023_2025_loss_per_valid_median_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def make_missing_report(df: pd.DataFrame, requested_prefixes: list[int], model_variants: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["dataset", "year", "month", "day_idx", "model_variant", "nugget_mode"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        ok = sub[sub["status"] == "ok"].copy() if "status" in sub.columns else sub.iloc[0:0].copy()
        if not ok.empty:
            ok = ok[finite_estimate_mask(ok)]
        ok_prefixes = set(pd.to_numeric(ok["block_prefix_requested"], errors="coerce").dropna().astype(int).tolist())
        for prefix in requested_prefixes:
            if int(prefix) in ok_prefixes:
                continue
            sub_prefix = sub[pd.to_numeric(sub["block_prefix_requested"], errors="coerce") == int(prefix)]
            latest_error = ""
            if "error" in sub_prefix.columns and sub_prefix["error"].notna().any():
                latest_error = str(sub_prefix["error"].dropna().iloc[-1])
            rows.append(
                {
                    **key_map,
                    "block_prefix_requested": int(prefix),
                    "attempt_rows": int(len(sub_prefix)),
                    "error_rows": int((sub_prefix["status"] == "error").sum()) if "status" in sub_prefix.columns else 0,
                    "latest_error": latest_error,
                }
            )
    _ = model_variants
    return pd.DataFrame(rows)


def refresh_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    monthly_out_dir: Path | None,
    requested_prefixes: list[int],
    model_variants: list[str],
) -> None:
    if not rows:
        return
    df = save_rows(out_dir / ALL_FITS_CSV, rows)
    missing = make_missing_report(df, requested_prefixes, model_variants)
    save_rows(out_dir / MISSING_CSV, missing)

    status_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    ok = status_ok[finite_estimate_mask(status_ok)].copy() if not status_ok.empty else status_ok
    param_summary = make_param_summary(ok)
    loss_summary = make_loss_summary(ok)
    if not param_summary.empty:
        save_rows(out_dir / PARAM_SUMMARY_CSV, param_summary)
        plot_param_summary(param_summary, out_dir, monthly_out_dir, loss_summary)
    if not loss_summary.empty:
        save_rows(out_dir / LOSS_SUMMARY_CSV, loss_summary)
        plot_loss_summary(loss_summary, out_dir, monthly_out_dir)

    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)}",
        f"Status ok: {int((df['status'] == 'ok').sum()) if 'status' in df.columns else 0}",
        f"Finite completed: {int(len(ok))}",
        f"Errors: {int((df['status'] == 'error').sum()) if 'status' in df.columns else 0}",
        f"Missing ok combinations: {int(len(missing))}",
        "",
        df.tail(12).to_string(index=False),
    ]
    (out_dir / "running_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real July 2023-2025 ST Matérn-vs-generalized-Cauchy nugget-zero tradeoff prefix test v2.")
    parser.add_argument("--real-years", nargs="+", default=["2023", "2024", "2025"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,15", help="'0,15' means day_idx 0..14; use '0,31' for all July.")
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--model-variants", nargs="+", default=DEFAULT_MODEL_VARIANTS)
    parser.add_argument("--block-prefixes", nargs="+", default=["100", "200", "400", "600", "800", "all"])
    parser.add_argument("--nugget-mode", choices=["zero"], default="zero")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=32)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--monthly-out-dir", type=Path, default=None)
    return parser


def load_existing(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / ALL_FITS_CSV
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[int, int, str, int, str]]:
    keys = set()
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        keys.add(
            (
                int(row["year"]),
                int(row["day_idx"]),
                str(row["model_variant"]),
                int(row["block_prefix_requested"]),
                str(row.get("nugget_mode", "zero")),
            )
        )
    return keys


def main() -> None:
    args = build_arg_parser().parse_args()
    model_variants = parse_tokens(args.model_variants)
    for name in model_variants:
        variant_spec(name)
    block_prefixes = parse_block_prefix_tokens(args.block_prefixes)
    device = resolve_device(args)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_out_dir = args.monthly_out_dir
    if monthly_out_dir is not None:
        monthly_out_dir.mkdir(parents=True, exist_ok=True)

    print("SRC:", SRC, flush=True)
    print("device:", device, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("years:", parse_int_tokens(args.real_years), "days:", parse_day_idxs(args.days), flush=True)
    print("model_variants:", model_variants, flush=True)
    print("year_variant_defaults:", YEAR_VARIANT_DEFAULTS, flush=True)
    print("block_prefixes:", block_prefixes, flush=True)
    print("nugget_mode:", args.nugget_mode, flush=True)

    rows = load_existing(out_dir) if args.skip_existing else []
    done = completed_keys(rows) if args.skip_existing else set()
    fit_id = int(max([0] + [int(r.get("fit_id", 0)) for r in rows]))

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "src": str(SRC),
        "device": str(device),
        "args": clean_json_value(vars(args)),
        "model_variants": {name: clean_json_value(variant_spec(name)) for name in model_variants},
        "year_variant_defaults": YEAR_VARIANT_DEFAULTS,
        "block_prefixes": block_prefixes,
        "density_definition": "first K max-min ordered 4x4 regular-grid target blocks; all cells inside selected blocks are kept",
        "init_physical": DEFAULT_REAL_INIT_PHYSICAL,
    }
    (out_dir / "run_config.json").write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    assets = load_real_assets(args)
    for asset in assets:
        asset_model_variants = variants_for_year(int(asset["year"]), model_variants)
        if not asset_model_variants:
            print(f"Skipping {asset['day']}: no requested variants are enabled for year {asset['year']}", flush=True)
            continue
        for block_prefix in block_prefixes:
            for model_variant in asset_model_variants:
                key = (
                    int(asset["year"]),
                    int(asset["day_idx"]),
                    str(model_variant),
                    int(block_prefix),
                    str(args.nugget_mode),
                )
                if args.skip_existing and key in done:
                    print(f"Skipping existing ok fit: {key}", flush=True)
                    continue
                fit_id += 1
                base = {
                    "fit_id": int(fit_id),
                    "status": "error",
                    "dataset": "real",
                    "year": int(asset["year"]),
                    "month": int(asset["month"]),
                    "day_idx": int(asset["day_idx"]),
                    "day": str(asset["day"]),
                    "model_variant": str(model_variant),
                    "block_prefix_requested": int(block_prefix),
                    "block_prefix_label": "all" if int(block_prefix) <= 0 else f"B{int(block_prefix)}",
                    "nugget_mode": str(args.nugget_mode),
                }
                print("\n" + "-" * 100, flush=True)
                print(
                    f"fit_id={fit_id} year={asset['year']} day={asset['day']} "
                    f"variant={model_variant} block_prefix={block_prefix} nugget={args.nugget_mode}",
                    flush=True,
                )
                print("-" * 100, flush=True)
                try:
                    row = fit_one(
                        asset=asset,
                        model_variant=str(model_variant),
                        block_prefix=int(block_prefix),
                        device=device,
                        args=args,
                        fit_id=int(fit_id),
                    )
                    print(
                        pd.Series(
                            {
                                k: row.get(k)
                                for k in [
                                    "day",
                                    "model_label",
                                    "block_prefix_label",
                                    "n_grid_block_prefix",
                                    "loss",
                                    "loss_per_valid",
                                    "fit_s",
                                    "est_sigmasq",
                                    "est_range_lat",
                                    "est_range_lon",
                                    "est_range_time",
                                    "est_advec_lat",
                                    "est_advec_lon",
                                    "est_nugget",
                                ]
                            }
                        ).to_string(),
                        flush=True,
                    )
                except Exception as exc:
                    row = {**base, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=10)}
                    print(f"ERROR: {row['error']}", flush=True)
                    traceback.print_exc()
                rows.append(clean_json_value(row))
                append_jsonl(out_dir / JSONL_NAME, row)
                if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                    refresh_outputs(out_dir, rows, monthly_out_dir, requested_prefixes=block_prefixes, model_variants=model_variants)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    refresh_outputs(out_dir, rows, monthly_out_dir, requested_prefixes=block_prefixes, model_variants=model_variants)
    print("\nDone.", flush=True)
    print("fits:", out_dir / ALL_FITS_CSV, flush=True)
    print("param summary:", out_dir / PARAM_SUMMARY_CSV, flush=True)
    print("loss summary:", out_dir / LOSS_SUMMARY_CSV, flush=True)


if __name__ == "__main__":
    main()
