#!/usr/bin/env python3
"""Real July 2023-2025 ST Vecchia conditional diagnostics.

This is the real-data counterpart of
``vecchia_conditional_eigen_sort_common_engine_061926.py``.
It reuses that script's fit/diagnostic engine, but builds day assets from the real
GEMS July pickles and runs two domain families:

  - full: the whole x1 July grid;
  - tile_2x4: eight spatial tiles, each fitted and diagnosed separately.

The default comparison is year-specific:

  - 2023: Matérn smooth=0.3, GC a=0.75 b=1, day-specific fine-tuned GC;
  - 2024: Matérn smooth=0.3, GC a=0.8 b=1;
  - 2025: Matérn smooth=0.3, GC a=0.75 b=1.

For each fitted model/domain/day, both diagnostic orderings are computed from
the same fitted Vecchia object:

  - max-min ordered: max-min target-block prefix, local eigen-rank, hours pooled;
  - eigen-sorted: pooled local conditional eigenvalue order.

Daily comparison plots are written as a single two-ordering panel for the full
domain by default.  Monthly average plots are refreshed after every completed
domain/day for the max-min ordering, including a combined 2x4 tile panel.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPACE_TIME_DIR = HERE.parent
for path in [HERE, SPACE_TIME_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_conditional_eigen_sort_common_engine_061926 as sim_eig  # noqa: E402

from GEMS_TCO import configuration as config  # noqa: E402
from GEMS_TCO import orderings  # noqa: E402
from GEMS_TCO.data_loader import load_data_dynamic_processed  # noqa: E402


DTYPE = sim_eig.DTYPE
ROUND_DECIMALS = sim_eig.ROUND_DECIMALS
LOSS_DECIMALS = sim_eig.LOSS_DECIMALS
BROWN_BRIDGE_Q95 = sim_eig.BROWN_BRIDGE_Q95
RUN_STEM = "real_july2023_2025_vecchia_conditional_dual_order_matern_gc_year_specific_domains_061926"
FINE_TUNED_VARIANT = "fine_tuned_gc"
MAXMIN_ORDER_LABEL = "max-min block -> local eigen rank -> hour"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "matern_s03": {
        "family": "matern",
        "smooth": 0.3,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.3 nugget0",
        "color": "#1f77b4",
    },
    "gc_a075_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 1.0,
        "label": "Baseline GC a=0.75 b=1 nugget0",
        "color": "#d62728",
    },
    "gc_a08_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.8,
        "gc_beta": 1.0,
        "label": "Baseline GC a=0.8 b=1 nugget0",
        "color": "#ff7f0e",
    },
    FINE_TUNED_VARIANT: {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 0.5,
        "label": "Fine tuned GC nugget0",
        "color": "#2ca02c",
    },
}

YEAR_MODEL_DEFAULTS: dict[int, list[str]] = {
    2023: ["matern_s03", "gc_a075_b1", FINE_TUNED_VARIANT],
    2024: ["matern_s03", "gc_a08_b1"],
    2025: ["matern_s03", "gc_a075_b1"],
}

FINE_TUNED_GC_BY_DAY_IDX: dict[int, tuple[float, float]] = {
    0: (0.75, 0.5),
    1: (0.75, 4.0),
    2: (0.75, 0.5),
    3: (0.75, 4.0),
    4: (0.8, 0.5),
    5: (0.9, 0.5),
    6: (0.9, 0.5),
    7: (0.75, 1.0),
    8: (1.0, 0.5),
    9: (0.75, 0.5),
    10: (0.9, 1.0),
    11: (0.8, 3.0),
    12: (0.9, 0.5),
    13: (1.0, 0.5),
    14: (0.8, 0.5),
    15: (0.8, 1.0),
    16: (0.8, 0.5),
    17: (0.75, 0.5),
    18: (0.8, 1.0),
    19: (0.9, 0.5),
    20: (0.9, 0.5),
    21: (0.75, 0.5),
    22: (0.8, 1.0),
    23: (0.75, 0.5),
    24: (0.9, 0.5),
    25: (0.75, 0.5),
    26: (0.75, 0.5),
    27: (0.9, 0.5),
    28: (0.8, 0.5),
    29: (0.9, 0.5),
}

sim_eig.RUN_STEM = RUN_STEM
sim_eig.MODEL_SPECS = MODEL_SPECS
sim_eig.TRUE_INIT_PHYSICAL = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.0,
}


@dataclass(frozen=True)
class DomainMeta:
    domain_group: str
    domain_label: str
    domain_title: str
    domain_order: int
    indices: np.ndarray
    tile_row: int = 0
    tile_col: int = 0
    tile_n_rows: int = 0
    tile_n_cols: int = 0
    lat_min: float = np.nan
    lat_max: float = np.nan
    lon_min: float = np.nan
    lon_max: float = np.nan
    n_blocks_used: int = 0
    n_blocks_full: int = 0


def default_data_root() -> Path:
    amarel = Path(getattr(config, "amarel_data_load_path", "/home/jl2815/tco/data"))
    if amarel.exists():
        return amarel
    return Path(getattr(config, "mac_data_load_path", "/Users/joonwonlee/Documents/GEMS_DATA"))


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path(f"/home/jl2815/tco/exercise_output/summer/real_data/{RUN_STEM}")
    return Path(f"/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/{RUN_STEM}")


def clean_json_value(value: Any) -> Any:
    return sim_eig.clean_json_value(value)


def parse_tokens(values: Iterable[str] | str) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return sim_eig.parse_tokens(values)


def parse_int_tokens(values: Iterable[str] | str) -> list[int]:
    if isinstance(values, str):
        values = [values]
    return [int(v) for v in parse_tokens(values)]


def parse_pair(text: str, cast=float) -> list[Any]:
    return sim_eig.parse_pair(text, cast)


def parse_day_idxs(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(31))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end))
    return [int(p) for p in parts]


def parse_grid_shape(text: str) -> tuple[int, int]:
    vals = str(text).lower().replace(",", "x").split("x")
    if len(vals) != 2:
        raise argparse.ArgumentTypeError(f"Expected grid shape like '2x4', got {text!r}")
    return int(vals[0]), int(vals[1])


def variants_for_year(year: int, requested: list[str]) -> list[str]:
    if not requested or "year_default" in requested:
        return list(YEAR_MODEL_DEFAULTS[int(year)])
    return [name for name in requested if name in MODEL_SPECS]


def fine_tuned_spec_for_day(day_idx: int) -> dict[str, Any]:
    if int(day_idx) not in FINE_TUNED_GC_BY_DAY_IDX:
        raise ValueError(f"No fine-tuned GC spec for day_idx={day_idx}")
    alpha, beta = FINE_TUNED_GC_BY_DAY_IDX[int(day_idx)]
    return {
        **MODEL_SPECS[FINE_TUNED_VARIANT],
        "gc_alpha": float(alpha),
        "gc_beta": float(beta),
        "label": "Fine tuned GC day-specific nugget0",
        "selected_model_label": f"Fine tuned GC a={alpha:g} b={beta:g} nugget0",
    }


def prepare_model_spec_for_asset(model_variant: str, asset: sim_eig.DayAsset) -> None:
    if model_variant != FINE_TUNED_VARIANT:
        return
    MODEL_SPECS[FINE_TUNED_VARIANT] = fine_tuned_spec_for_day(int(asset.day_idx))
    sim_eig.MODEL_SPECS = MODEL_SPECS


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
    block_shape: tuple[int, int],
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
    return selected, {
        "n_lat_grid": int(n_lat),
        "n_lon_grid": int(n_lon),
        "block_shape": f"{block_y}x{block_x}",
        "n_blocks_full": int(len(block_keys)),
        "n_blocks_requested": int(requested),
        "n_blocks_used": int(n_use),
        "n_grid_selected": int(selected.size),
    }


def tile_domain_metas(args: argparse.Namespace, grid_coords_np: np.ndarray) -> list[DomainMeta]:
    tile_y, tile_x = parse_grid_shape(args.tile_grid)
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    lat_edges = np.linspace(float(lat_range[0]), float(lat_range[1]), tile_y + 1)
    lon_edges = np.linspace(float(lon_range[0]), float(lon_range[1]), tile_x + 1)
    lat = np.asarray(grid_coords_np[:, 0], dtype=np.float64)
    lon = np.asarray(grid_coords_np[:, 1], dtype=np.float64)
    metas: list[DomainMeta] = []
    for iy in range(tile_y):
        for ix in range(tile_x):
            lat_lo, lat_hi = float(lat_edges[iy]), float(lat_edges[iy + 1])
            lon_lo, lon_hi = float(lon_edges[ix]), float(lon_edges[ix + 1])
            lat_mask = (lat >= lat_lo) & ((lat <= lat_hi) if iy == tile_y - 1 else (lat < lat_hi))
            lon_mask = (lon >= lon_lo) & ((lon <= lon_hi) if ix == tile_x - 1 else (lon < lon_hi))
            idx = np.flatnonzero(lat_mask & lon_mask).astype(np.int64)
            if idx.size == 0:
                continue
            label = f"tile_y{iy + 1:02d}_x{ix + 1:02d}"
            title = f"tile y{iy + 1}, x{ix + 1} [{lat_lo:.2f},{lat_hi:.2f}] x [{lon_lo:.2f},{lon_hi:.2f}]"
            metas.append(
                DomainMeta(
                    domain_group=f"tile_{tile_y}x{tile_x}",
                    domain_label=label,
                    domain_title=title,
                    domain_order=100 + iy * tile_x + ix,
                    indices=idx,
                    tile_row=iy + 1,
                    tile_col=ix + 1,
                    tile_n_rows=tile_y,
                    tile_n_cols=tile_x,
                    lat_min=lat_lo,
                    lat_max=lat_hi,
                    lon_min=lon_lo,
                    lon_max=lon_hi,
                )
            )
    return metas


def build_domain_metas(args: argparse.Namespace, grid_coords_np: np.ndarray) -> list[DomainMeta]:
    modes = set(parse_tokens(args.domain_modes))
    metas: list[DomainMeta] = []
    n_full = int(np.asarray(grid_coords_np).shape[0])
    if "full" in modes:
        metas.append(
            DomainMeta(
                domain_group="full",
                domain_label="full",
                domain_title="full x1 grid",
                domain_order=0,
                indices=np.arange(n_full, dtype=np.int64),
            )
        )
    if "center400" in modes:
        idx, meta = blockmaxmin_indices(
            grid_coords_np,
            int(args.center_block_prefix),
            tuple(int(x) for x in sim_eig.BLOCK_SHAPE),
        )
        n_used = int(meta["n_blocks_used"])
        metas.append(
            DomainMeta(
                domain_group="center400",
                domain_label=f"center_b{n_used}",
                domain_title=f"first {n_used} max-min 4x4 blocks",
                domain_order=10,
                indices=idx,
                n_blocks_used=n_used,
                n_blocks_full=int(meta["n_blocks_full"]),
            )
        )
    if "tile_2x4" in modes or f"tile_{args.tile_grid}" in modes:
        metas.extend(tile_domain_metas(args, grid_coords_np))
    if not metas:
        raise ValueError(f"No domain metas built from --domain-modes={args.domain_modes!r}")
    return sorted(metas, key=lambda m: m.domain_order)


def asset_for_domain(base_asset: sim_eig.DayAsset, meta: DomainMeta) -> sim_eig.DayAsset:
    idx = np.asarray(meta.indices, dtype=np.int64)
    source_map = {k: v[idx].contiguous() for k, v in base_asset.source_map.items()}
    n_valid, n_total, _ = sim_eig.count_valid(source_map)
    asset = sim_eig.DayAsset(
        year=int(base_asset.year),
        month=int(base_asset.month),
        day_idx=int(base_asset.day_idx),
        day_label=str(base_asset.day_label),
        keys=list(base_asset.keys),
        source_map=source_map,
        grid_coords_np=np.asarray(base_asset.grid_coords_np, dtype=np.float64)[idx].copy(),
        monthly_mean=float(base_asset.monthly_mean),
        n_rows_per_time=int(idx.size),
        n_valid=int(n_valid),
        n_total=int(n_total),
    )
    for key, value in meta.__dict__.items():
        if key != "indices":
            setattr(asset, key, value)
    setattr(asset, "dataset", "real")
    setattr(asset, "n_grid_full", int(base_asset.n_rows_per_time))
    return asset


def load_real_domain_assets(args: argparse.Namespace) -> list[sim_eig.DayAsset]:
    data_root = args.data_root or default_data_root()
    data_loader = load_data_dynamic_processed(str(data_root))
    years = parse_int_tokens(args.years)
    days = parse_day_idxs(args.days)
    lat_lon_resolution = [int(x) for x in parse_pair(args.space, int)]
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    assets: list[sim_eig.DayAsset] = []

    for year in years:
        print("\n" + "=" * 92, flush=True)
        print(f"Loading real July data for {year}-{int(args.month):02d}", flush=True)
        print("=" * 92, flush=True)
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
            raise RuntimeError(f"No real data loaded for {year}-{int(args.month):02d} from {data_root}")
        base_grid_coords_np = df_map[key_idx[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        sim_eig.assert_grid_order_consistent(df_map, key_idx, base_grid_coords_np)
        domain_metas = build_domain_metas(args, base_grid_coords_np)
        print(
            f"n hourly slots={len(key_idx)} grid={base_grid_coords_np.shape} "
            f"monthly_mean={float(monthly_mean):.6f} domains={len(domain_metas)}",
            flush=True,
        )

        for day_idx in days:
            start, end = int(day_idx) * int(args.hours_per_day), (int(day_idx) + 1) * int(args.hours_per_day)
            selected_keys = key_idx[start:end]
            if len(selected_keys) != int(args.hours_per_day):
                print(
                    f"Skipping year={year} day_idx={day_idx}: expected {args.hours_per_day} keys, got {len(selected_keys)}",
                    flush=True,
                )
                continue
            day = f"{year}-{int(args.month):02d}-{int(day_idx) + 1:02d}"
            source_map, _ = data_loader.load_working_data(
                df_map,
                float(monthly_mean),
                [start, end],
                ord_mm=None,
                dtype=DTYPE,
                keep_ori=bool(args.keep_exact_loc),
            )
            base_n_valid, base_n_total, _ = sim_eig.count_valid(source_map)
            base_asset = sim_eig.DayAsset(
                year=int(year),
                month=int(args.month),
                day_idx=int(day_idx),
                day_label=day,
                keys=list(selected_keys),
                source_map={k: v.contiguous() for k, v in source_map.items()},
                grid_coords_np=base_grid_coords_np.copy(),
                monthly_mean=float(monthly_mean),
                n_rows_per_time=int(base_grid_coords_np.shape[0]),
                n_valid=int(base_n_valid),
                n_total=int(base_n_total),
            )
            for meta in domain_metas:
                assets.append(asset_for_domain(base_asset, meta))
    return assets


def domain_fields(asset: sim_eig.DayAsset) -> dict[str, Any]:
    keys = [
        "dataset",
        "domain_group",
        "domain_label",
        "domain_title",
        "domain_order",
        "tile_row",
        "tile_col",
        "tile_n_rows",
        "tile_n_cols",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "n_blocks_used",
        "n_blocks_full",
        "n_grid_full",
    ]
    return {key: getattr(asset, key, np.nan) for key in keys}


def safe_path_token(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def maxmin_order_label(mode: str) -> str:
    labels = {
        "block_rank": "max-min block -> local eigen rank, hours pooled",
        "block_rank_hour": "max-min block -> local eigen rank -> hour",
        "block_hour_rank": "max-min block -> hour -> local eigen rank",
        "hour_block_rank": "hour -> max-min block -> local eigen rank",
    }
    return labels.get(str(mode).strip().lower(), str(mode))


def model_label(model_variant: str, summary: pd.DataFrame | None = None) -> str:
    return sim_eig.model_label(model_variant, summary)


def _curve_y(curve: pd.DataFrame, residual_df: float | None = None) -> np.ndarray:
    if "scaled_cumsum" in curve.columns:
        return curve["scaled_cumsum"].to_numpy(dtype=np.float64)
    if residual_df is None:
        residual_df = float(curve["expected"].iloc[-1]) if "expected" in curve.columns and len(curve) else 1.0
    return curve["cumsum_y2"].to_numpy(dtype=np.float64) / max(float(residual_df), sim_eig.EPS)


def plot_dual_daily_comparison(
    maxmin_curves: dict[str, pd.DataFrame],
    eigen_curves: dict[str, pd.DataFrame],
    summary_rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    if not maxmin_curves and not eigen_curves:
        return
    summary = pd.DataFrame([r for r in summary_rows if r.get("status") == "ok"])
    styles = {
        "matern_s03": {"color": "#1f77b4", "linewidth": 2.35, "linestyle": "-", "alpha": 0.85},
        "gc_a075_b1": {"color": "#d62728", "linewidth": 2.15, "linestyle": "--", "alpha": 0.92},
        "gc_a08_b1": {"color": "#ff7f0e", "linewidth": 2.15, "linestyle": "--", "alpha": 0.92},
        FINE_TUNED_VARIANT: {"color": "#2ca02c", "linewidth": 1.95, "linestyle": "-.", "alpha": 0.96},
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 8.0),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    specs = [
        ("maxmin", "Max-min ordered", maxmin_curves, "maxmin_D"),
        ("eigen", "Eigenvalue sorted pooling", eigen_curves, "eigen_D"),
    ]
    for col, (_, subtitle, curves, d_col) in enumerate(specs):
        ax = axes[0, col]
        ax_dev = axes[1, col]
        y_max = 1.04
        for model_variant, curve in curves.items():
            if curve.empty:
                continue
            curve = curve.sort_values("frac_index")
            row_df = summary[summary["model_variant"].astype(str) == str(model_variant)]
            row = row_df.iloc[-1] if not row_df.empty else pd.Series(dtype=object)
            x = curve["frac_index"].to_numpy(dtype=np.float64)
            residual_df = row.get("residual_df", np.nan)
            y = _curve_y(curve, residual_df if pd.notna(residual_df) else None)
            if np.isfinite(y).any():
                y_max = max(y_max, float(np.nanmax(y)) * 1.04)
            style = styles.get(
                str(model_variant),
                {
                    "color": MODEL_SPECS.get(str(model_variant), {}).get("color"),
                    "linewidth": 2.0,
                    "linestyle": "-",
                    "alpha": 0.9,
                },
            )
            label = model_label(str(model_variant), summary)
            if d_col in row and pd.notna(row[d_col]):
                label += f", D={float(row[d_col]):.2f}"
            ax.plot(x, y, label=label, **style)
            ax_dev.plot(x, y - x, label=str(model_variant), **style)
        grid = np.linspace(0.0, 1.0, 200)
        ax.plot(grid, grid, color="0.25", linewidth=1.25, linestyle=":", alpha=0.9, label="reference y=x")
        ax_dev.axhline(0.0, color="0.25", linewidth=1.1, linestyle=":", alpha=0.9)
        ax.set_title(subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax_dev.set_ylim(-0.025, 0.060)
        ax.grid(alpha=0.18)
        ax_dev.grid(alpha=0.18)
        ax_dev.set_xlabel("projected expected df fraction")
        if col == 0:
            ax.set_ylabel("cumulative squared score / residual df")
            ax_dev.set_ylabel("curve - reference")
        ax.legend(fontsize=7.4, framealpha=0.88, loc="upper left")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_comparison(monthly: pd.DataFrame, summary: pd.DataFrame, out_path: Path, title: str) -> None:
    if monthly.empty:
        return
    fig, ax = plt.subplots(figsize=(6.7, 5.25))
    y_max = 1.05
    for model_variant, g in monthly.groupby("model_variant", sort=True):
        color = MODEL_SPECS.get(str(model_variant), {}).get("color")
        g = g.sort_values("frac_index")
        x = g["frac_index"].to_numpy(dtype=np.float64)
        y = g["scaled_cumsum_mean"].to_numpy(dtype=np.float64)
        sd = g["scaled_cumsum_sd"].fillna(0.0).to_numpy(dtype=np.float64)
        y_max = max(y_max, float(np.nanmax(y + sd)) * 1.04)
        ax.plot(x, y, color=color, linewidth=2.0, label=model_label(str(model_variant), summary))
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, linewidth=0)
    grid = np.linspace(0.0, 1.0, 200)
    ax.plot(grid, grid, color="0.45", linewidth=1.1)
    residual_df_mean = pd.to_numeric(monthly["residual_df_mean"], errors="coerce").dropna()
    if not residual_df_mean.empty:
        band = BROWN_BRIDGE_Q95 * math.sqrt(2.0 / max(float(residual_df_mean.mean()), 1.0))
        ax.plot(grid, grid - band, color="0.65", linestyle=(0, (4, 4)), linewidth=0.85)
        ax.plot(grid, grid + band, color="0.65", linestyle=(0, (4, 4)), linewidth=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(f"projected expected df fraction, ordered by {MAXMIN_ORDER_LABEL}")
    ax.set_ylabel("monthly mean projected cumulative squared score / residual df")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def monthly_table_for_domain(all_avg: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        all_avg.groupby(keys + ["model_variant", "frac_index"], as_index=False)
        .agg(
            scaled_cumsum_mean=("scaled_cumsum", "mean"),
            scaled_cumsum_sd=("scaled_cumsum", "std"),
            n_days=("scaled_cumsum", "count"),
            n_scores_mean=("n_scores", "mean"),
            residual_df_mean=("residual_df", "mean"),
        )
        .sort_values(keys + ["model_variant", "frac_index"])
    )


def monthly_roots_for_domain(out_dir: Path, domain_group: str) -> tuple[Path, Path]:
    group = str(domain_group)
    if group == "full":
        return out_dir / "monthly_average_full", out_dir / "monthly_average_plots_full"
    if group.startswith("tile_"):
        return out_dir / f"monthly_average_{group}", out_dir / f"monthly_average_plots_{group}"
    return out_dir / "monthly_average_other", out_dir / "monthly_average_plots_other"


def plot_tile_daily_panel_outputs(all_avg: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    if all_avg.empty:
        return
    tile_avg = all_avg[all_avg["domain_group"].astype(str).str.startswith("tile_")].copy()
    if tile_avg.empty:
        return
    for (year, day_idx, day, domain_group), sub_day_group in tile_avg.groupby(
        ["year", "day_idx", "day", "domain_group"],
        sort=True,
    ):
        tile_meta = (
            sub_day_group[["domain_label", "domain_title", "tile_row", "tile_col", "tile_n_rows", "tile_n_cols"]]
            .drop_duplicates()
            .copy()
        )
        if tile_meta.empty:
            continue
        n_rows = int(pd.to_numeric(tile_meta["tile_n_rows"], errors="coerce").dropna().max())
        n_cols = int(pd.to_numeric(tile_meta["tile_n_cols"], errors="coerce").dropna().max())
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.25 * n_rows), sharex=True, sharey=True)
        axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
        y_max = 1.05
        for _, meta_row in tile_meta.iterrows():
            r = int(meta_row["tile_row"]) - 1
            c = int(meta_row["tile_col"]) - 1
            label = str(meta_row["domain_label"])
            ax = axes_arr[r, c]
            tile_daily = sub_day_group[sub_day_group["domain_label"].astype(str) == label]
            tile_summary = summary[
                (summary["year"].astype(int) == int(year))
                & (summary["day_idx"].astype(int) == int(day_idx))
                & (summary["domain_label"].astype(str) == label)
            ]
            for model_variant, g in tile_daily.groupby("model_variant", sort=True):
                color = MODEL_SPECS.get(str(model_variant), {}).get("color")
                g = g.sort_values("frac_index")
                x = g["frac_index"].to_numpy(dtype=np.float64)
                y = g["scaled_cumsum"].to_numpy(dtype=np.float64)
                if np.isfinite(y).any():
                    y_max = max(y_max, float(np.nanmax(y)) * 1.04)
                ax.plot(x, y, color=color, linewidth=1.65, label=model_label(str(model_variant), tile_summary))
            grid = np.linspace(0.0, 1.0, 100)
            ax.plot(grid, grid, color="0.48", linewidth=0.9)
            ax.set_title(str(meta_row["domain_title"]), fontsize=9)
            ax.grid(alpha=0.20)
        for ax in axes_arr.flat:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_max)
        handles, labels = axes_arr.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 3), fontsize=8)
        fig.supxlabel(f"projected expected df fraction, ordered by {MAXMIN_ORDER_LABEL}")
        fig.supylabel("cumulative squared score / residual df")
        fig.suptitle(f"Real July {int(year)} day_idx={int(day_idx)} ({day}) {domain_group}: conditional max-min ordered daily panel")
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        out_path = (
            out_dir
            / f"daily_plots_{domain_group}"
            / f"year_{int(year)}"
            / f"real_{int(year)}_day{int(day_idx) + 1:02d}_{domain_group}_vecchia_conditional_maxmin_order_panel.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def refresh_monthly_outputs(avg_rows: list[pd.DataFrame], summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not avg_rows:
        return
    nonempty_avg = [df for df in avg_rows if not df.empty]
    if not nonempty_avg:
        return
    all_avg = pd.concat(nonempty_avg, ignore_index=True)
    if all_avg.empty:
        return
    summary = pd.DataFrame([r for r in summary_rows if r.get("status") == "ok"])
    if summary.empty:
        summary = pd.DataFrame(columns=["year", "domain_group", "domain_label", "model_variant", "vecchia_loss_per_obs"])
    group_cols = ["year", "domain_group", "domain_label"]
    monthly_all = monthly_table_for_domain(all_avg, group_cols)
    plot_tile_daily_panel_outputs(all_avg, summary, out_dir)

    for (year, domain_group, domain_label), monthly in monthly_all.groupby(group_cols, sort=True):
        token = safe_path_token(str(domain_label))
        monthly_root, plot_root = monthly_roots_for_domain(out_dir, str(domain_group))
        domain_dir = monthly_root / f"year_{int(year)}" / str(domain_group) / token
        domain_dir.mkdir(parents=True, exist_ok=True)
        daily = all_avg[
            (all_avg["year"].astype(int) == int(year))
            & (all_avg["domain_group"].astype(str) == str(domain_group))
            & (all_avg["domain_label"].astype(str) == str(domain_label))
        ].copy()
        daily.round(ROUND_DECIMALS).to_csv(
            domain_dir / f"real_{int(year)}_{token}_daily_resampled_curves.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )
        monthly.round(ROUND_DECIMALS).to_csv(
            domain_dir / f"real_{int(year)}_{token}_monthly_average_curves.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )
        sub_summary = summary[
            (summary["year"].astype(int) == int(year))
            & (summary["domain_group"].astype(str) == str(domain_group))
            & (summary["domain_label"].astype(str) == str(domain_label))
        ].copy()
        title = f"Real July {int(year)} {domain_label}: monthly average Vecchia conditional max-min ordered diagnostic"
        plot_monthly_comparison(
            monthly,
            sub_summary,
            plot_root / f"year_{int(year)}" / str(domain_group) / f"real_{int(year)}_{token}_monthly_average_vecchia_conditional_maxmin_order_comparison.png",
            title,
        )

    for year, sub_year in monthly_all[monthly_all["domain_group"].astype(str).str.startswith("tile_")].groupby("year", sort=True):
        tile_meta = (
            all_avg[all_avg["year"].astype(int) == int(year)]
            [["domain_group", "domain_label", "domain_title", "tile_row", "tile_col", "tile_n_rows", "tile_n_cols"]]
            .drop_duplicates()
        )
        if tile_meta.empty:
            continue
        for domain_group, sub_group in sub_year.groupby("domain_group", sort=True):
            _, plot_root = monthly_roots_for_domain(out_dir, str(domain_group))
            meta_group = tile_meta[tile_meta["domain_group"].astype(str) == str(domain_group)]
            if meta_group.empty:
                continue
            n_rows = int(pd.to_numeric(meta_group["tile_n_rows"], errors="coerce").dropna().max())
            n_cols = int(pd.to_numeric(meta_group["tile_n_cols"], errors="coerce").dropna().max())
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.25 * n_rows), sharex=True, sharey=True)
            axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
            y_max = 1.05
            for _, meta_row in meta_group.iterrows():
                r = int(meta_row["tile_row"]) - 1
                c = int(meta_row["tile_col"]) - 1
                label = str(meta_row["domain_label"])
                ax = axes_arr[r, c]
                tile_monthly = sub_group[sub_group["domain_label"].astype(str) == label]
                tile_summary = summary[
                    (summary["year"].astype(int) == int(year))
                    & (summary["domain_label"].astype(str) == label)
                ]
                for model_variant, g in tile_monthly.groupby("model_variant", sort=True):
                    color = MODEL_SPECS.get(str(model_variant), {}).get("color")
                    g = g.sort_values("frac_index")
                    x = g["frac_index"].to_numpy(dtype=np.float64)
                    y = g["scaled_cumsum_mean"].to_numpy(dtype=np.float64)
                    y_max = max(y_max, float(np.nanmax(y)) * 1.04)
                    ax.plot(x, y, color=color, linewidth=1.65, label=model_label(str(model_variant), tile_summary))
                grid = np.linspace(0.0, 1.0, 100)
                ax.plot(grid, grid, color="0.48", linewidth=0.9)
                ax.set_title(str(meta_row["domain_title"]), fontsize=9)
                ax.grid(alpha=0.20)
            for ax in axes_arr.flat:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, y_max)
            handles, labels = axes_arr.flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 3), fontsize=8)
            fig.supxlabel(f"projected expected df fraction, ordered by {MAXMIN_ORDER_LABEL}")
            fig.supylabel("monthly mean cumulative squared score / residual df")
            fig.suptitle(f"Real July {int(year)} {domain_group}: Vecchia conditional max-min ordered monthly averages")
            fig.tight_layout(rect=(0, 0.06, 1, 0.96))
            out_path = plot_root / f"year_{int(year)}" / f"real_{int(year)}_{domain_group}_monthly_average_vecchia_conditional_maxmin_order_comparison.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)


def save_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    return sim_eig.save_summary(summary_rows, out_dir)


def write_eigenvalue_stability_summaries(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    rank_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    base_cols = [
        "year",
        "month",
        "day_idx",
        "day",
        "domain_group",
        "domain_label",
        "model_variant",
        "model_label",
        "vecchia_loss_per_obs",
        "mean_y2",
        "max_abs_bridge_scaled",
    ]
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        base = {col: row.get(col, np.nan) for col in base_cols}
        max_rank = int(row.get("max_local_eigen_rank", 16) or 16)
        for rank in range(1, max_rank + 1):
            prefix = f"local_rank{rank:02d}"
            median_key = f"{prefix}_lambda_median"
            if median_key not in row:
                continue
            rank_rows.append(
                {
                    **base,
                    "local_eigen_rank": int(rank),
                    "n_scores": row.get(f"{prefix}_n", np.nan),
                    "mean_y2_over_expected": row.get(f"{prefix}_mean_y2", np.nan),
                    "lambda_mean": row.get(f"{prefix}_lambda_mean", np.nan),
                    "lambda_sd": row.get(f"{prefix}_lambda_sd", np.nan),
                    "lambda_min": row.get(f"{prefix}_lambda_min", np.nan),
                    "lambda_p10": row.get(f"{prefix}_lambda_p10", np.nan),
                    "lambda_p25": row.get(f"{prefix}_lambda_p25", np.nan),
                    "lambda_median": row.get(median_key, np.nan),
                    "lambda_p75": row.get(f"{prefix}_lambda_p75", np.nan),
                    "lambda_p90": row.get(f"{prefix}_lambda_p90", np.nan),
                    "lambda_max": row.get(f"{prefix}_lambda_max", np.nan),
                }
            )
        for rank in range(1, max_rank):
            gap_key = f"local_rank{rank:02d}_{rank + 1:02d}_lambda_iqr_gap"
            if gap_key not in row:
                continue
            gap_rows.append(
                {
                    **base,
                    "rank_hi": int(rank),
                    "rank_lo": int(rank + 1),
                    "lambda_iqr_gap": row.get(gap_key, np.nan),
                    "iqr_overlaps": bool(float(row.get(gap_key, np.nan)) <= 0.0),
                }
            )

    if rank_rows:
        rank_df = pd.DataFrame(rank_rows)
        rank_df.round(ROUND_DECIMALS).to_csv(
            out_dir / f"{RUN_STEM}_local_rank_eigenvalue_stability_daily.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )
        group_cols = ["year", "domain_group", "domain_label", "model_variant", "local_eigen_rank"]
        monthly = (
            rank_df.groupby(group_cols, as_index=False)
            .agg(
                n_days=("day_idx", "nunique"),
                n_scores_mean=("n_scores", "mean"),
                mean_y2_over_expected_mean=("mean_y2_over_expected", "mean"),
                lambda_median_mean=("lambda_median", "mean"),
                lambda_p25_mean=("lambda_p25", "mean"),
                lambda_p75_mean=("lambda_p75", "mean"),
                lambda_p10_mean=("lambda_p10", "mean"),
                lambda_p90_mean=("lambda_p90", "mean"),
            )
            .sort_values(group_cols)
        )
        monthly.round(ROUND_DECIMALS).to_csv(
            out_dir / f"{RUN_STEM}_local_rank_eigenvalue_stability_monthly.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )
    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        gap_df.round(ROUND_DECIMALS).to_csv(
            out_dir / f"{RUN_STEM}_adjacent_local_rank_iqr_gaps_daily.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )
        group_cols = ["year", "domain_group", "domain_label", "model_variant", "rank_hi", "rank_lo"]
        gap_monthly = (
            gap_df.groupby(group_cols, as_index=False)
            .agg(
                n_days=("day_idx", "nunique"),
                lambda_iqr_gap_mean=("lambda_iqr_gap", "mean"),
                overlap_rate=("iqr_overlaps", "mean"),
            )
            .sort_values(group_cols)
        )
        gap_monthly.round(ROUND_DECIMALS).to_csv(
            out_dir / f"{RUN_STEM}_adjacent_local_rank_iqr_gaps_monthly.csv",
            index=False,
            float_format=f"%.{ROUND_DECIMALS}f",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real July 2023-2025 ST Vecchia dual-order conditional diagnostics.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--years", nargs="+", default=["2023", "2024", "2025"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,30", help="'0,30' means day_idx 0..29; use 'all' for day_idx 0..30.")
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--domain-modes", default="full,tile_2x4")
    parser.add_argument("--center-block-prefix", type=int, default=400)
    parser.add_argument("--tile-grid", default="2x4")
    parser.add_argument("--daily-plot-domains", default="full")
    parser.add_argument("--model-variants", nargs="+", default=["year_default"])
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=sim_eig.REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=32)
    parser.add_argument("--diag-chunk-size", type=int, default=64)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=8)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    parser.add_argument(
        "--maxmin-order-mode",
        choices=["block_rank", "block_rank_hour", "block_hour_rank", "hour_block_rank"],
        default="block_rank",
        help=(
            "Ordering for max-min cumulative curve. Default pools spatial prefix first: "
            "max-min block -> local eigen rank, with hours pooled inside each cell."
        ),
    )
    parser.add_argument("--resample-grid", type=int, default=200)
    parser.add_argument("--save-daily-curves", action="store_true")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    global MAXMIN_ORDER_LABEL
    MAXMIN_ORDER_LABEL = maxmin_order_label(args.maxmin_order_mode)
    requested_variants = parse_tokens(args.model_variants)
    known_requested = [v for v in requested_variants if v != "year_default"]
    for variant in known_requested:
        if variant not in MODEL_SPECS:
            raise ValueError(f"Unknown model variant {variant!r}. Known: {sorted(MODEL_SPECS)} plus year_default")
    device = sim_eig.resolve_device(args)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "engine_script": str((HERE / "vecchia_conditional_eigen_sort_common_engine_061926.py").resolve()),
        "src": str(sim_eig.SRC),
        "device": str(device),
        "args": clean_json_value(vars(args)),
        "model_specs": clean_json_value(MODEL_SPECS),
        "year_model_defaults": clean_json_value(YEAR_MODEL_DEFAULTS),
        "fine_tuned_gc_by_day_idx": clean_json_value(FINE_TUNED_GC_BY_DAY_IDX),
        "init_physical": clean_json_value(sim_eig.TRUE_INIT_PHYSICAL),
        "domain_definition": {
            "full": "whole x1 real July grid",
            "tile_2x4": "2 latitude by 4 longitude half-open tiles, inclusive on outer upper edges",
        },
        "diagnostic_definition": (
            "Vecchia conditional target-block covariance eigenbasis; conditional-eigen scores are projected "
            "off the fitted mean-design column space via P=I-Z(Z'Z)^-1Z'. For each fitted model, "
            "two curves are computed from the same fitted parameters: max-min ordered by "
            f"{MAXMIN_ORDER_LABEL}, and pooled sorted by conditional eigenvalue."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    print("SRC:", sim_eig.SRC, flush=True)
    print("device:", device, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("requested_variants:", requested_variants, flush=True)
    print("year defaults:", YEAR_MODEL_DEFAULTS, flush=True)

    assets = load_real_domain_assets(args)
    daily_plot_domains = set(parse_tokens(args.daily_plot_domains))
    summary_rows: list[dict[str, Any]] = []
    avg_rows: list[pd.DataFrame] = []

    for asset in assets:
        model_variants = variants_for_year(int(asset.year), requested_variants)
        if not model_variants:
            print(f"Skipping asset year={asset.year}: no model variants selected.", flush=True)
            continue
        print("\n" + "=" * 104, flush=True)
        print(
            f"Real day year={asset.year} day_idx={asset.day_idx} day={asset.day_label} "
            f"domain={getattr(asset, 'domain_group', '')}/{getattr(asset, 'domain_label', '')} "
            f"rows/time={asset.n_rows_per_time:,} valid={asset.n_valid:,}/{asset.n_total:,} "
            f"models={model_variants}",
            flush=True,
        )
        print("=" * 104, flush=True)
        day_curves: dict[str, pd.DataFrame] = {}
        day_eigen_curves: dict[str, pd.DataFrame] = {}
        day_summary_rows: list[dict[str, Any]] = []

        for model_variant in model_variants:
            print(f"\n--- Fitting and diagnosing {model_variant} ---", flush=True)
            model = None
            try:
                prepare_model_spec_for_asset(str(model_variant), asset)
                fit_row, model, beta = sim_eig.fit_one_model(asset, model_variant, device, args)
                params = torch.as_tensor(
                    sim_eig.physical_to_log_phi(
                        {
                            "sigmasq": fit_row["est_sigmasq"],
                            "range_lat": fit_row["est_range_lat"],
                            "range_lon": fit_row["est_range_lon"],
                            "range_time": fit_row["est_range_time"],
                            "advec_lat": fit_row["est_advec_lat"],
                            "advec_lon": fit_row["est_advec_lon"],
                            "nugget": 0.0,
                        }
                    ),
                    device=device,
                    dtype=DTYPE,
                )
                t_diag = time.time()
                curve, diag_summary = sim_eig.conditional_maxmin_order_curve(model, params, beta, args)
                diag_s = time.time() - t_diag
                t_eigen_diag = time.time()
                eigen_curve, eigen_diag_summary = sim_eig.conditional_eigen_curve(model, params, beta, args)
                eigen_diag_s = time.time() - t_eigen_diag
                row = {
                    **fit_row,
                    **domain_fields(asset),
                    **diag_summary,
                    "diag_s": float(diag_s),
                    "maxmin_diag_s": float(diag_s),
                    "eigen_diag_s": float(eigen_diag_s),
                    "maxmin_mean_y2": float(diag_summary.get("mean_y2", np.nan)),
                    "maxmin_D": float(diag_summary.get("max_abs_bridge_scaled", np.nan)),
                    "maxmin_n_scores": int(diag_summary.get("n_conditional_scores", 0)),
                    "eigen_mean_y2": float(eigen_diag_summary.get("mean_y2", np.nan)),
                    "eigen_D": float(eigen_diag_summary.get("max_abs_bridge_scaled", np.nan)),
                    "eigen_n_scores": int(eigen_diag_summary.get("n_conditional_scores", 0)),
                    "eigen_conditional_loss_per_score": float(eigen_diag_summary.get("conditional_loss_per_score", np.nan)),
                }
                if str(model_variant) == FINE_TUNED_VARIANT:
                    row["selected_model_label"] = str(MODEL_SPECS[FINE_TUNED_VARIANT].get("selected_model_label", ""))
                print(
                    pd.Series(
                        {
                            "model": row["model_label"],
                            "loss/obs": f"{row['vecchia_loss_per_obs']:.{LOSS_DECIMALS}f}",
                            "conditional_loss/score": f"{row['conditional_loss_per_score']:.{LOSS_DECIMALS}f}",
                            "mean_y2": f"{row['mean_y2']:.5f}",
                            "D": f"{row['max_abs_bridge_scaled']:.5f}",
                            "eigen_D": f"{row['eigen_D']:.5f}",
                            "n_scores": row["n_conditional_scores"],
                            "diag_s max/eig": f"{diag_s:.1f}/{eigen_diag_s:.1f}",
                        }
                    ).to_string(),
                    flush=True,
                )
                if args.save_daily_curves:
                    token = safe_path_token(str(getattr(asset, "domain_label", "domain")))
                    curve_path = (
                        out_dir
                        / "daily_curves"
                        / f"year_{asset.year}"
                        / str(getattr(asset, "domain_group", "domain"))
                        / token
                        / f"real_{asset.year}_day{asset.day_idx + 1:02d}_{token}_{model_variant}_conditional_maxmin_order_curve.csv"
                    )
                    curve_path.parent.mkdir(parents=True, exist_ok=True)
                    curve.round(ROUND_DECIMALS).to_csv(curve_path, index=False, float_format=f"%.{ROUND_DECIMALS}f")
                    eigen_curve_path = (
                        out_dir
                        / "daily_curves"
                        / f"year_{asset.year}"
                        / str(getattr(asset, "domain_group", "domain"))
                        / token
                        / f"real_{asset.year}_day{asset.day_idx + 1:02d}_{token}_{model_variant}_conditional_eigen_sorted_curve.csv"
                    )
                    eigen_curve.round(ROUND_DECIMALS).to_csv(eigen_curve_path, index=False, float_format=f"%.{ROUND_DECIMALS}f")
                day_curves[model_variant] = curve
                day_eigen_curves[model_variant] = eigen_curve
                avg_rows.append(
                    sim_eig.resample_curve(curve, int(args.resample_grid)).assign(
                        year=int(asset.year),
                        month=int(asset.month),
                        day_idx=int(asset.day_idx),
                        day=str(asset.day_label),
                        model_variant=str(model_variant),
                        **domain_fields(asset),
                    )
                )
            except Exception as exc:
                row = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                    "year": int(asset.year),
                    "month": int(asset.month),
                    "day_idx": int(asset.day_idx),
                    "day": str(asset.day_label),
                    "model_variant": str(model_variant),
                    **domain_fields(asset),
                }
                print(f"ERROR for {model_variant}: {row['error']}", flush=True)
                traceback.print_exc()
            finally:
                if model is not None:
                    del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            summary_rows.append(clean_json_value(row))
            day_summary_rows.append(clean_json_value(row))
            save_summary(summary_rows, out_dir)
            write_eigenvalue_stability_summaries(summary_rows, out_dir)

        domain_label = str(getattr(asset, "domain_label", ""))
        domain_group = str(getattr(asset, "domain_group", ""))
        if day_curves and (domain_label in daily_plot_domains or domain_group in daily_plot_domains):
            token = safe_path_token(domain_label)
            plot_dual_daily_comparison(
                day_curves,
                day_eigen_curves,
                day_summary_rows,
                out_dir
                / f"daily_plots_{domain_group}"
                / f"year_{asset.year}"
                / f"real_{asset.year}_day{asset.day_idx + 1:02d}_{token}_vecchia_conditional_dual_order_comparison.png",
                (
                    f"Real July {asset.year} day_idx={asset.day_idx} ({asset.day_label}) "
                    f"{domain_label}: Vecchia conditional diagnostics"
                ),
            )

        refresh_monthly_outputs(avg_rows, summary_rows, out_dir)

    summary = save_summary(summary_rows, out_dir)
    write_eigenvalue_stability_summaries(summary_rows, out_dir)
    refresh_monthly_outputs(avg_rows, summary_rows, out_dir)
    print("\nDone.", flush=True)
    print("summary:", out_dir / f"{RUN_STEM}_summary.csv", flush=True)
    print("ok rows:", int((summary["status"] == "ok").sum()) if not summary.empty and "status" in summary.columns else 0, flush=True)


if __name__ == "__main__":
    main()
